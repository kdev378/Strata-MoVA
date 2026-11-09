// include/strata/k2h/dequant_rows.hpp - one call that dequantizes a run of whole blocks of any type a K2 Horizon GGUF
// stores (IFM's quants: F32, F16, BF16, Q8_0, Q5_0, Q4_K, Q5_K, Q6_K; the rest of strata/artifact/dequant.hpp too).
//
// The per-type code is strata/artifact/dequant.hpp's scalar reference dequantizers (transcribed from ggml).  This is the
// K2 path's reference read: the load check uses it to read every byte of every tensor, and the CPU reference forward
// (later steps) uses it for its matrices.  No SIMD, no state.
#pragma once

#include "strata/artifact/dequant.hpp"
#include "strata/artifact/gguf_reader.hpp"

#include <cstdint>

namespace strata::k2h {

/// Whether `type` can be dequantized here.
inline bool can_dequantize(uint32_t type) {
    switch (type) {
    case 0: case 1: case 30:            // F32, F16, BF16
    case 2: case 6: case 7: case 8:     // Q4_0, Q5_0, Q5_1, Q8_0
    case 11: case 12: case 13: case 14: // Q3_K, Q4_K, Q5_K, Q6_K
    case 20: case 23:                   // IQ4_NL, IQ4_XS
        return true;
    default:
        return false;
    }
}

/// Dequantizes `n` elements (a whole number of `type`'s blocks) from `src` into `out`.  False when the type is not one
/// `can_dequantize` names or `n` is not whole blocks.
inline bool dequantize_rows(uint32_t type, const uint8_t* src, float* out, int64_t n) {
    int be = 0, bb = 0;
    if (!can_dequantize(type) || !strata::block_geometry(type, be, bb) || n % be) return false;
    const int64_t blocks = n / be;
    switch (type) {
    case 0: strata::dequantize_f32(src, out, (int) n); return true;
    case 1: strata::dequantize_f16(src, out, (int) n); return true;
    case 30: strata::dequantize_bf16(src, out, (int) n); return true;
    default: break;
    }
    for (int64_t b = 0; b < blocks; ++b) {
        const uint8_t* p = src + b * bb;
        float* o = out + b * be;
        switch (type) {
        case 2: strata::dequantize_q4_0(p, o); break;
        case 6: strata::dequantize_q5_0(p, o); break;
        case 7: strata::dequantize_q5_1(p, o); break;
        case 8: strata::dequantize_q8_0(p, o); break;
        case 11: strata::dequantize_q3_K(p, o); break;
        case 12: strata::dequantize_q4_K(p, o); break;
        case 13: strata::dequantize_q5_K(p, o); break;
        case 14: strata::dequantize_q6_K(p, o); break;
        case 20: strata::dequantize_iq4_nl(p, o); break;
        case 23: strata::dequantize_iq4_xs(p, o); break;
        default: return false;
        }
    }
    return true;
}

}  // namespace strata::k2h
