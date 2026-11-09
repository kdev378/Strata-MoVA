// include/strata/k2h/cpu_ref.hpp - K2 Horizon's CPU reference forward: plain F32 arithmetic over the dequantized weights,
// one operation after another in llama.cpp b11454's order (src/models/k2-horizon.cpp), to check the model's numbers before
// any fast path exists.  Not a serving path: no SIMD, a thread per output slice, the weights dequantized row by row on
// every call.
//
// Each operation follows the ggml CPU op it stands for (ggml/src/ggml-cpu at b11454):
//   * the grouped RMS norm: n_embd / groups slices, each x * 1/sqrtf(mean(x^2) + eps) with the sum of squares in double
//     (ggml_compute_forward_rms_norm_f32), then one full-width weight (k2_horizon_group_rms_norm);
//   * RoPE, NEOX: pairs (i, i + n_rot/2), theta_i = pos * theta_scale^i accumulated in float with theta_scale =
//     powf(base, -2/n_rot) (ggml_rope_cache_init), cosf / sinf;
//   * attention: softmax(Q K^T / sqrt(head_dim)) V over the cached positions <= the query's (causal), GQA (query head h
//     reads KV head h / (n_head / n_head_kv));
//   * the output gate: log2(1 + 2^x) = softplus(x ln 2) / ln 2, softplus(y) = y > 20 ? y : logf(1 + expf(y)) (op_softplus),
//     times the attention output, before attn_output;
//   * the FFN: down(silu(gate(x)) * up(x)), silu(x) = x / (1 + expf(-x)).
//
// The matrix products: llama.cpp's CPU matmul rounds the activations to the weight type's vec_dot_type before its dot
// product (ggml-cpu.c type_traits_cpu: Q8_K for the K-quants and IQ4_XS, Q8_0 for Q4_0 / Q5_0 / Q8_0 / IQ4_NL, F16 for
// F16, none for F32).  ActRound::LlamaCpu (the default) does the same rounding (quantize_row_q8_K_ref, which x86 uses as
// is; quantize_row_q8_0_ref) and then takes the dot product of the rounded activations with the dequantized weights in
// double: the same sum as ggml's integer dot product up to float rounding.  ActRound::None keeps the activations F32.
// With the rounding off, the layer outputs differ from llama.cpp by several percent after two layers (measured on
// K2-Horizon-MoVA-36B-A4B Q4_K_M, docs/K2_HORIZON.md), so the comparison needs it on.
//
// Steps done: the dense layers (0 .. leading_dense_block_count - 1) and the attention of every layer except MoVA.  A MoE
// or MoVA layer is refused (steps 5 and 6 of the port).
#pragma once

#include "strata/k2h/dequant_rows.hpp"
#include "strata/k2h/geometry.hpp"
#include "strata/k2h/layout.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace strata::k2h::ref {

enum class ActRound { None, LlamaCpu };

/// F32 -> F16 bits, round to nearest even (as x86's F16C conversion that ggml uses for GGML_FP32_TO_FP16).
inline uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t ax = x & 0x7fffffffu;
    if (ax >= 0x7f800000u) return (uint16_t) (sign | 0x7c00u | (ax > 0x7f800000u ? 0x200u : 0u));   // inf / nan
    if (ax >= 0x477ff000u) return (uint16_t) (sign | 0x7c00u);                                        // overflow
    if (ax < 0x38800000u) {   // a subnormal (or zero) half
        if (ax < 0x33000000u) return (uint16_t) sign;
        const uint32_t m = (ax & 0x7fffffu) | 0x800000u;
        const int shift = 126 - (int) (ax >> 23);   // 14 .. 24
        const uint32_t half = m >> shift, rest = m & ((1u << shift) - 1u), mid = 1u << (shift - 1);
        return (uint16_t) (sign | (half + ((rest > mid || (rest == mid && (half & 1u))) ? 1u : 0u)));
    }
    const uint32_t h = ((ax - 0x38000000u) >> 13), rest = ax & 0x1fffu;
    return (uint16_t) (sign | (h + ((rest > 0x1000u || (rest == 0x1000u && (h & 1u))) ? 1u : 0u)));
}

/// ggml's nearest_int: round half to even through the float mantissa (ggml-quants.c).
inline int nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i;
    std::memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

/// x (n, a multiple of 256) rounded as quantize_row_q8_K_ref rounds it, written back as the values the dot product sees.
inline void round_q8_K(float* x, int64_t n) {
    for (int64_t b = 0; b < n; b += 256) {
        float max = 0, amax = 0;
        for (int j = 0; j < 256; ++j) {
            const float ax = fabsf(x[b + j]);
            if (ax > amax) {
                amax = ax;
                max = x[b + j];
            }
        }
        if (!amax) {
            for (int j = 0; j < 256; ++j) x[b + j] = 0;
            continue;
        }
        const float iscale = -127.f / max;
        const float d = 1 / iscale;
        for (int j = 0; j < 256; ++j) x[b + j] = d * (float) std::min(127, nearest_int(iscale * x[b + j]));
    }
}

/// x (n, a multiple of 32) rounded as quantize_row_q8_0_ref rounds it (the scale stored as F16).
inline void round_q8_0(float* x, int64_t n) {
    for (int64_t b = 0; b < n; b += 32) {
        float amax = 0;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, fabsf(x[b + j]));
        const float d = amax / 127.f, id = d ? 1.0f / d : 0.0f;
        const float d16 = strata::fp16_to_fp32(fp32_to_fp16(d));
        for (int j = 0; j < 32; ++j) x[b + j] = d16 * (float) (int8_t) roundf(x[b + j] * id);
    }
}

/// The activations a weight of `type` meets in llama.cpp's CPU matmul (in place, one row of n).
inline void round_for(uint32_t type, float* x, int64_t n) {
    switch (type) {
    case 10: case 11: case 12: case 13: case 14: case 23:   // Q2_K .. Q6_K, IQ4_XS: Q8_K
        round_q8_K(x, n);
        break;
    case 2: case 6: case 8: case 20:                        // Q4_0, Q5_0, Q8_0, IQ4_NL: Q8_0
        round_q8_0(x, n);
        break;
    case 1:                                                 // F16
        for (int64_t i = 0; i < n; ++i) x[i] = strata::fp16_to_fp32(fp32_to_fp16(x[i]));
        break;
    default:                                                // F32 (and anything else): as it is
        break;
    }
}

/// y[t * n_out + j] = sum_i W[j][i] x[t * n_in + i] for T tokens; W is a GGUF matrix [n_in, n_out] (row j contiguous).
inline void matmul(const TensorRef& W, const float* x_in, int64_t T, float* y, int threads, ActRound act = ActRound::LlamaCpu) {
    const int64_t n_in = (int64_t) W.info->shape[0], n_out = (int64_t) W.info->shape[1];
    std::vector<float> rounded;
    const float* x = x_in;
    if (act == ActRound::LlamaCpu) {
        rounded.assign(x_in, x_in + T * n_in);
        for (int64_t t = 0; t < T; ++t) round_for(W.type(), rounded.data() + t * n_in, n_in);
        x = rounded.data();
    }
    int be = 0, bb = 0;
    if (!strata::block_geometry(W.type(), be, bb)) throw std::runtime_error(W.name() + ": unknown type");
    const int64_t row_bytes = n_in / be * bb;
    auto work = [&](int64_t j0, int64_t j1) {
        std::vector<float> row((size_t) n_in);
        for (int64_t j = j0; j < j1; ++j) {
            if (!dequantize_rows(W.type(), W.data() + j * row_bytes, row.data(), n_in))
                throw std::runtime_error(W.name() + ": no dequantizer for " + W.info->type_name());
            for (int64_t t = 0; t < T; ++t) {
                const float* xt = x + t * n_in;
                double acc = 0;   // a double sum: the reference's own rounding stays far below the Q8_K difference
                for (int64_t i = 0; i < n_in; ++i) acc += (double) row[(size_t) i] * xt[i];
                y[t * n_out + j] = (float) acc;
            }
        }
    };
    const int64_t n_thr = std::max<int64_t>(1, std::min<int64_t>(threads, n_out / 64));
    if (n_thr == 1) {
        work(0, n_out);
        return;
    }
    std::vector<std::thread> pool;
    const int64_t per = (n_out + n_thr - 1) / n_thr;
    for (int64_t k = 0; k < n_thr; ++k) pool.emplace_back(work, k * per, std::min(n_out, (k + 1) * per));
    for (auto& th : pool) th.join();
}

/// A 1-D weight (norms, biases) as F32.
inline std::vector<float> vector_of(const TensorRef& t) {
    std::vector<float> v(t.info->elements());
    if (!dequantize_rows(t.type(), t.data(), v.data(), (int64_t) v.size())) throw std::runtime_error(t.name() + ": unreadable");
    return v;
}

/// The grouped RMS norm of one token's n values (k2_horizon_group_rms_norm), times `w` when given.
inline void group_rms_norm(const float* x, const float* w, float* y, int64_t n, int64_t groups, float eps) {
    const int64_t g = n / groups;
    for (int64_t k = 0; k < groups; ++k) {
        double sum = 0;
        for (int64_t i = 0; i < g; ++i) sum += (double) (x[k * g + i] * x[k * g + i]);
        const float mean = (float) (sum / (double) g);
        const float scale = 1.0f / sqrtf(mean + eps);
        for (int64_t i = 0; i < g; ++i) y[k * g + i] = x[k * g + i] * scale;
    }
    if (w)
        for (int64_t i = 0; i < n; ++i) y[i] *= w[i];
}

/// NEOX RoPE in place over `heads` heads of width `hd` at position `pos`: the first n_rot dims, pairs (i, i + n_rot/2).
inline void rope_neox(float* x, int64_t heads, int64_t hd, int64_t n_rot, int64_t pos, float base) {
    const float theta_scale = powf(base, -2.0f / (float) n_rot);
    std::vector<float> cs((size_t) n_rot);
    float theta = (float) pos;
    for (int64_t i0 = 0; i0 < n_rot; i0 += 2) {
        cs[(size_t) i0] = cosf(theta);
        cs[(size_t) i0 + 1] = sinf(theta);
        theta *= theta_scale;
    }
    const int64_t half = n_rot / 2;
    for (int64_t h = 0; h < heads; ++h) {
        float* v = x + h * hd;
        for (int64_t i = 0; i < half; ++i) {
            const float c = cs[(size_t) (2 * i)], s = cs[(size_t) (2 * i + 1)];
            const float x0 = v[i], x1 = v[i + half];
            v[i] = x0 * c - x1 * s;
            v[i + half] = x0 * s + x1 * c;
        }
    }
}

/// log2(1 + 2^x), as the graph builds it: scale by ln 2, op_softplus, scale by 1/ln 2.
inline float softplus2(float x) {
    const float y = x * 0.6931471805599453f;
    const float sp = (y > 20.0f) ? y : logf(1.0f + expf(y));
    return sp * 1.4426950408889634f;
}

inline float silu(float x) { return x / (1.0f + expf(-x)); }

/// Called with each intermediate tensor: its llama.cpp graph name ("attn_norm-0", "Qcur-0", ...), the values (T tokens,
/// `per_token` each, in llama.cpp's element order) and the count per token.
using Hook = std::function<void(const std::string& name, const float* data, int64_t T, int64_t per_token)>;

class Forward {
public:
    Forward(const Geometry& g, const Weights& w, int64_t n_ctx, int threads, ActRound act = ActRound::LlamaCpu)
        : g_(g), w_(w), n_ctx_(n_ctx), threads_(threads), act_(act) {
        k_.assign((size_t) g.n_layers, std::vector<float>());
        v_.assign((size_t) g.n_layers, std::vector<float>());
        for (int64_t l = 0; l < g.n_layers; ++l) {
            k_[(size_t) l].resize((size_t) (n_ctx * g.n_embd_kv()));
            v_[(size_t) l].resize((size_t) (n_ctx * g.n_embd_kv()));
        }
    }

    /// Runs `tokens` at positions pos0 .. through layers 0 .. n_run - 1 and returns the residual stream after them
    /// ([T][n_embd]).  The KV cache of those layers is filled for these positions.
    std::vector<float> run(const std::vector<int32_t>& tokens, int64_t pos0, int64_t n_run, const Hook& hook = {}) {
        const int64_t T = (int64_t) tokens.size(), N = g_.n_embd;
        if (pos0 + T > n_ctx_) throw std::runtime_error("the context is full");
        std::vector<float> x((size_t) (T * N));
        embed(tokens, x.data());
        for (int64_t l = 0; l < n_run; ++l) run_layer(l, x, T, pos0, hook);
        return x;
    }

    /// One layer over the residual stream `x` ([T][n_embd], updated in place) of T tokens at positions pos0 ..
    void run_layer(int64_t l, std::vector<float>& x, int64_t T, int64_t pos0, const Hook& hook = {}) {
        if (g_.is_moe_layer(l) || g_.is_mova_layer(l))
            throw std::runtime_error("layer " + std::to_string(l) + " is a MoE/MoVA layer: not in the CPU reference yet");
        layer(l, x, T, pos0, hook);
    }

    void embed(const std::vector<int32_t>& tokens, float* x) const {
        const TensorRef& E = w_.token_embd;
        int be = 0, bb = 0;
        strata::block_geometry(E.type(), be, bb);
        const int64_t N = g_.n_embd, row_bytes = N / be * bb;
        for (size_t t = 0; t < tokens.size(); ++t) {
            if (tokens[t] < 0 || tokens[t] >= g_.n_vocab) throw std::runtime_error("token id out of range");
            dequantize_rows(E.type(), E.data() + (int64_t) tokens[t] * row_bytes, x + (int64_t) t * N, N);
        }
    }

private:
    void emit(const Hook& hook, const char* what, int64_t l, const std::vector<float>& v, int64_t T) const {
        if (hook) hook(std::string(what) + "-" + std::to_string(l), v.data(), T, (int64_t) v.size() / T);
    }

    void layer(int64_t l, std::vector<float>& x, int64_t T, int64_t pos0, const Hook& hook) {
        const LayerWeights& L = w_.layers[(size_t) l];
        const int64_t N = g_.n_embd, HD = g_.head_dim, H = g_.n_head, HKV = g_.n_head_kv, KV = g_.n_embd_kv(), HQ = H * HD;
        const float eps = (float) g_.rms_eps;

        // attention
        std::vector<float> h((size_t) (T * N));
        const std::vector<float> an = vector_of(L.attn_norm);
        for (int64_t t = 0; t < T; ++t) group_rms_norm(&x[(size_t) (t * N)], an.data(), &h[(size_t) (t * N)], N, g_.n_norm_groups, eps);
        emit(hook, "attn_norm", l, h, T);
        std::vector<float> q((size_t) (T * HQ)), k((size_t) (T * KV)), v((size_t) (T * KV));
        matmul(L.attn_q, h.data(), T, q.data(), threads_, act_);
        matmul(L.attn_k, h.data(), T, k.data(), threads_, act_);
        matmul(L.attn_v, h.data(), T, v.data(), threads_, act_);
        if (L.attn_q_norm || L.attn_k_norm) throw std::runtime_error("per-head Q/K norms are not in the CPU reference yet");
        for (int64_t t = 0; t < T; ++t) {
            rope_neox(&q[(size_t) (t * HQ)], H, HD, g_.n_rot, pos0 + t, (float) g_.rope_freq_base);
            rope_neox(&k[(size_t) (t * KV)], HKV, HD, g_.n_rot, pos0 + t, (float) g_.rope_freq_base);
        }
        emit(hook, "Qcur", l, q, T);
        emit(hook, "Kcur", l, k, T);
        emit(hook, "Vcur", l, v, T);
        std::vector<float>& kc = k_[(size_t) l];
        std::vector<float>& vc = v_[(size_t) l];
        std::copy(k.begin(), k.end(), kc.begin() + pos0 * KV);
        std::copy(v.begin(), v.end(), vc.begin() + pos0 * KV);

        std::vector<float> att((size_t) (T * HQ));
        const float kq_scale = 1.0f / sqrtf((float) HD);
        const int64_t group = H / HKV;
        std::vector<float> sc;
        for (int64_t t = 0; t < T; ++t) {
            const int64_t n_pos = pos0 + t + 1;
            sc.resize((size_t) n_pos);
            for (int64_t hh = 0; hh < H; ++hh) {
                const float* qv = &q[(size_t) (t * HQ + hh * HD)];
                const int64_t kvh = hh / group;
                float mx = -INFINITY;
                for (int64_t p = 0; p < n_pos; ++p) {
                    const float* kv = &kc[(size_t) (p * KV + kvh * HD)];
                    double dot = 0;
                    for (int64_t i = 0; i < HD; ++i) dot += (double) qv[i] * kv[i];
                    sc[(size_t) p] = (float) dot * kq_scale;
                    mx = std::max(mx, sc[(size_t) p]);
                }
                double sum = 0;
                for (int64_t p = 0; p < n_pos; ++p) {
                    sc[(size_t) p] = expf(sc[(size_t) p] - mx);
                    sum += sc[(size_t) p];
                }
                float* o = &att[(size_t) (t * HQ + hh * HD)];
                for (int64_t i = 0; i < HD; ++i) {
                    double acc = 0;
                    for (int64_t p = 0; p < n_pos; ++p) acc += (double) sc[(size_t) p] * vc[(size_t) (p * KV + kvh * HD + i)];
                    o[i] = (float) (acc / sum);
                }
            }
        }
        emit(hook, "kqv_out", l, att, T);
        if (L.attn_gate) {
            std::vector<float> gate((size_t) (T * HQ));
            matmul(L.attn_gate, h.data(), T, gate.data(), threads_, act_);
            for (size_t i = 0; i < gate.size(); ++i) att[i] *= softplus2(gate[i]);
        }
        std::vector<float> o((size_t) (T * N));
        matmul(L.attn_output, att.data(), T, o.data(), threads_, act_);
        for (size_t i = 0; i < x.size(); ++i) x[i] += o[i];
        emit(hook, "ffn_inp", l, x, T);

        // dense FFN
        const std::vector<float> fn = vector_of(L.ffn_norm);
        for (int64_t t = 0; t < T; ++t) group_rms_norm(&x[(size_t) (t * N)], fn.data(), &h[(size_t) (t * N)], N, g_.n_norm_groups, eps);
        emit(hook, "ffn_norm", l, h, T);
        const int64_t F = g_.n_ff_dense;
        std::vector<float> gt((size_t) (T * F)), up((size_t) (T * F));
        matmul(L.ffn_gate, h.data(), T, gt.data(), threads_, act_);
        matmul(L.ffn_up, h.data(), T, up.data(), threads_, act_);
        for (size_t i = 0; i < gt.size(); ++i) gt[i] = silu(gt[i]) * up[i];
        matmul(L.ffn_down, gt.data(), T, o.data(), threads_, act_);
        emit(hook, "ffn_out", l, o, T);
        for (size_t i = 0; i < x.size(); ++i) x[i] += o[i];
        emit(hook, "l_out", l, x, T);
    }

    const Geometry& g_;
    const Weights& w_;
    int64_t n_ctx_;
    int threads_;
    ActRound act_;
    std::vector<std::vector<float>> k_, v_;   // [layer][pos * n_embd_kv]
};

}  // namespace strata::k2h::ref
