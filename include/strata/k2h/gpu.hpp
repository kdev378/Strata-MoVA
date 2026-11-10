// include/strata/k2h/gpu.hpp - strata-k2h's GPU side: weights copied to VRAM and their matrix products run there with
// Strata's native MMVQ kernels (strata/kernels/native_mmvq.hpp: llama.cpp's CUDA MMVQ, activations as Q8_1).  Only in a
// CUDA build (STRATA_K2H_GPU); the engine asks it first for every product and computes on the CPU what it does not hold.
//
// No CUDA header here: the engine (a plain C++ file) includes it; src/k2h/k2h_gpu.cu implements it.
#pragma once

#include "strata/k2h/layout.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace strata::k2h {

struct GpuImpl;

class Gpu {
public:
    /// The CUDA device `device`, nullptr with `err` when there is none (or this build has no GPU side).
    static std::unique_ptr<Gpu> open(int device, std::string& err);
    ~Gpu();

    /// Whether a weight of this ggml type and row length can be held (a native MMVQ kernel exists for it).
    static bool takes(uint32_t type, int64_t n_in);
    /// Copies `t` to VRAM if it fits in the budget (bytes still free for weights); false otherwise (it stays on the CPU).
    bool hold(const TensorRef& t);
    /// y[t * n_out + j] = W[row0 + j] . x[t] for j < n_out and T tokens, on the GPU when W is held; false (nothing done)
    /// when it is not.  x and y are host memory.
    bool mv(const TensorRef& W, int64_t row0, int64_t n_out, const float* x, int64_t T, float* y);

    void set_budget(size_t bytes);    ///< bytes the held weights may take
    size_t held_bytes() const;
    size_t free_vram() const;         ///< cudaMemGetInfo's free bytes now
    std::string device_name() const;

private:
    explicit Gpu(GpuImpl* p);
    GpuImpl* p_;
};

}  // namespace strata::k2h
