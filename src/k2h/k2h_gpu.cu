// src/k2h/k2h_gpu.cu - include/strata/k2h/gpu.hpp: K2 Horizon's weights in VRAM and their matrix products on the GPU,
// through Strata's native MMVQ kernels (src/kernels/cuda/native_mmvq.cu, llama.cpp's CUDA MMVQ pinned).  The simplest
// GPU side that can work: each product copies its activations to the GPU and its outputs back, synchronously.  Built
// only with CUDA (STRATA_K2H_GPU), opt-in at run time (strata-k2h --gpu).
#include "strata/k2h/gpu.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <unordered_map>
#include <vector>

namespace strata::k2h {

namespace {
using F32Fn = void (*)(const void*, const float*, void*, float*, int, int, int, void*);

/// The native MMVQ entry for a ggml type, and the block it needs n_in to be a multiple of.
F32Fn kernel_for(uint32_t type, int& block) {
    namespace K = strata::kernels;
    switch (type) {
    case 2: block = 32; return K::native_q4_0_f32;
    case 6: block = 32; return K::native_q5_0_f32;
    case 8: block = 32; return K::native_q8_0_f32;
    case 20: block = 32; return K::native_iq4_nl_f32;
    case 11: block = 256; return K::native_q3_k_f32;
    case 12: block = 256; return K::native_q4_k_f32;
    case 13: block = 256; return K::native_q5_k_f32;
    case 14: block = 256; return K::native_q6_k_f32;
    case 23: block = 256; return K::native_iq4_xs_f32;
    default: block = 0; return nullptr;
    }
}

bool ok(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "strata-k2h: CUDA %s: %s\n", what, cudaGetErrorString(e));
    return false;
}
}  // namespace

struct GpuImpl {
    int device = 0;
    cudaStream_t stream = nullptr;
    std::unordered_map<const strata::TensorInfo*, uint8_t*> held;   // the GGUF directory entry -> its device copy
    size_t budget = 0, used = 0;
    float* dx = nullptr;
    float* dy = nullptr;
    void* dq = nullptr;
    size_t cap_x = 0, cap_y = 0, cap_q = 0;   // floats, floats, bytes

    bool grow(void** p, size_t& cap, size_t want) {
        if (want <= cap) return true;
        if (*p) cudaFree(*p);
        *p = nullptr;
        cap = 0;
        if (!ok(cudaMalloc(p, want), "cudaMalloc (scratch)")) return false;
        cap = want;
        return true;
    }
};

std::unique_ptr<Gpu> Gpu::open(int device, std::string& err) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) {
        err = "no CUDA device";
        return nullptr;
    }
    if (device < 0 || device >= n) {
        err = "no CUDA device " + std::to_string(device) + " (" + std::to_string(n) + " found)";
        return nullptr;
    }
    if (cudaSetDevice(device) != cudaSuccess) {
        err = "cudaSetDevice failed";
        return nullptr;
    }
    auto* p = new GpuImpl;
    p->device = device;
    if (cudaStreamCreateWithFlags(&p->stream, cudaStreamNonBlocking) != cudaSuccess) {
        delete p;
        err = "cudaStreamCreate failed";
        return nullptr;
    }
    return std::unique_ptr<Gpu>(new Gpu(p));
}

Gpu::Gpu(GpuImpl* p) : p_(p) {}

Gpu::~Gpu() {
    if (!p_) return;
    cudaSetDevice(p_->device);
    for (auto& [info, d] : p_->held) cudaFree(d);
    if (p_->dx) cudaFree(p_->dx);
    if (p_->dy) cudaFree(p_->dy);
    if (p_->dq) cudaFree(p_->dq);
    if (p_->stream) cudaStreamDestroy(p_->stream);
    delete p_;
}

bool Gpu::takes(uint32_t type, int64_t n_in) {
    int block = 0;
    return kernel_for(type, block) != nullptr && n_in > 0 && n_in % block == 0 && n_in <= (1 << 30);
}

void Gpu::set_budget(size_t bytes) { p_->budget = bytes; }
size_t Gpu::held_bytes() const { return p_->used; }

size_t Gpu::free_vram() const {
    size_t fr = 0, tot = 0;
    cudaSetDevice(p_->device);
    cudaMemGetInfo(&fr, &tot);
    return fr;
}

std::string Gpu::device_name() const {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, p_->device);
    return prop.name;
}

bool Gpu::hold(const TensorRef& t) {
    if (!t || p_->held.count(t.info)) return (bool) t;
    if (!takes(t.type(), (int64_t) t.info->shape[0])) return false;
    const size_t bytes = (size_t) t.bytes();
    if (p_->used + bytes > p_->budget) return false;
    uint8_t* d = nullptr;
    if (cudaMalloc((void**) &d, bytes) != cudaSuccess) {
        cudaGetLastError();   // (the failed allocation is not an error of the run: the weight stays on the CPU)
        return false;
    }
    if (!ok(cudaMemcpy(d, t.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy (weights)")) {
        cudaFree(d);
        return false;
    }
    p_->held[t.info] = d;
    p_->used += bytes;
    return true;
}

bool Gpu::mv(const TensorRef& W, int64_t row0, int64_t n_out, const float* x, int64_t T, float* y) {
    const auto it = p_->held.find(W.info);
    if (it == p_->held.end()) return false;
    int block = 0;
    const F32Fn fn = kernel_for(W.type(), block);
    const int64_t n_in = (int64_t) W.info->shape[0];
    const int64_t row_bytes = (int64_t) (strata::tensor_payload_bytes(*W.info) / (W.info->elements() / (uint64_t) n_in));
    const int64_t cols = std::min<int64_t>(T, 8);
    if (!p_->grow((void**) &p_->dx, p_->cap_x, (size_t) (cols * n_in) * sizeof(float)) ||
        !p_->grow((void**) &p_->dy, p_->cap_y, (size_t) (cols * n_out) * sizeof(float)) ||
        !p_->grow(&p_->dq, p_->cap_q, strata::kernels::native_q8_1_bytes((int) n_in, (int) cols)))
        return false;
    const uint8_t* w = it->second + row0 * row_bytes;
    for (int64_t t0 = 0; t0 < T; t0 += 8) {   // the kernels take 1 .. 8 columns
        const int64_t c = std::min<int64_t>(8, T - t0);
        cudaMemcpyAsync(p_->dx, x + t0 * n_in, (size_t) (c * n_in) * sizeof(float), cudaMemcpyHostToDevice, p_->stream);
        fn(w, p_->dx, p_->dq, p_->dy, (int) n_in, (int) n_out, (int) c, p_->stream);
        cudaMemcpyAsync(y + t0 * n_out, p_->dy, (size_t) (c * n_out) * sizeof(float), cudaMemcpyDeviceToHost, p_->stream);
    }
    return ok(cudaStreamSynchronize(p_->stream), W.name().c_str());
}

}  // namespace strata::k2h
