// include/strata/k2h/geometry.hpp - K2 Horizon's geometry (GGUF architecture "k2-horizon"), read from a GGUF's metadata.
//
// The reference is llama.cpp b11454 (src/models/k2-horizon.cpp: load_arch_hparams, load_arch_tensors and the graph;
// conversion/k2_horizon.py for where each key comes from in the HF config).  Keys llama.cpp reads as optional keep
// llama.cpp's defaults here, so a file llama.cpp loads gets the same numbers:
//
//   * attention.group_norm_groups (layernorm_num_groups): 1 when absent.  The RMS norm runs over n_embd / groups
//     slices and then one full-width weight (k2_horizon_group_rms_norm), on attn_norm, ffn_norm and output_norm.
//   * leading_dense_block_count: 0.  Layers below it are dense (ffn_up/gate/down, feed_forward_length wide, a plain
//     attn_v); the rest are MoE (n_expert > 0) and, with value experts, MoVA layers.  llama.cpp ignores
//     moe_every_n_layers for this family, so a file whose value is not 1 is refused rather than run another way.
//   * expert_gating_func: sigmoid when absent or 0 (LLAMA_EXPERT_GATING_FUNC_TYPE_NONE); the FFN router and the MoVA
//     value router share it, and share expert_weights_norm (norm_topk_prob) and expert_weights_scale
//     (router_scaling_factor; 0 = no scaling).
//   * attention.value_expert_count / value_expert_used_count (mova_num_experts / _per_tok): 0 = no MoVA.
//   * attention.key_length / value_length: n_embd / n_head; they must be equal (the graph asserts it).
//   * rope.dimension_count: the head width.  RoPE is NEOX-style (llama_model_rope_type).
//
// Header-only and GPU-free, like core/family_layout.hpp: the engine, the tools and tests/core/k2h_geometry_test.cpp
// share it.  Nothing here runs the model: model_family_not_supported still refuses K2 Horizon in every build.
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/model_family.hpp"

#include <cstdint>
#include <cstdio>
#include <string>

namespace strata::k2h {

/// llama.cpp's llama_expert_gating_func_type values, as the GGUF stores them.
enum class Gating : int64_t { Softmax = 1, Sigmoid = 2 };

struct Geometry {
    int64_t n_layers = 0;          ///< block_count
    int64_t n_embd = 0;            ///< embedding_length
    int64_t n_head = 0;            ///< attention.head_count
    int64_t n_head_kv = 0;         ///< attention.head_count_kv
    int64_t head_dim = 0;          ///< attention.key_length (= attention.value_length)
    int64_t n_rot = 0;             ///< rope.dimension_count
    double rope_freq_base = 0;     ///< rope.freq_base
    double rms_eps = 0;            ///< attention.layer_norm_rms_epsilon
    int64_t n_norm_groups = 1;     ///< attention.group_norm_groups
    int64_t n_vocab = 0;           ///< the length of tokenizer.ggml.tokens

    int64_t n_ff_dense = 0;        ///< feed_forward_length: the dense layers' FFN width
    int64_t n_dense_lead = 0;      ///< leading_dense_block_count

    int64_t n_expert = 0;          ///< expert_count (0: a dense model)
    int64_t n_expert_used = 0;     ///< expert_used_count
    int64_t n_ff_exp = 0;          ///< expert_feed_forward_length
    int64_t n_expert_shared = 0;   ///< expert_shared_count
    int64_t n_ff_shexp = 0;        ///< expert_shared_feed_forward_length (llama.cpp: n_ff_exp * n_expert_shared when 0)
    bool expert_weights_norm = false;   ///< expert_weights_norm
    double expert_weights_scale = 0;    ///< expert_weights_scale (0 or 1: no scaling)
    Gating gating = Gating::Sigmoid;    ///< expert_gating_func

    int64_t n_value_expert = 0;        ///< attention.value_expert_count (MoVA)
    int64_t n_value_expert_used = 0;   ///< attention.value_expert_used_count

    int64_t n_embd_kv() const { return n_head_kv * head_dim; }   ///< K and V width (the MoVA experts' output)
    bool is_moe_layer(int64_t l) const { return n_expert > 0 && l >= n_dense_lead; }
    bool is_mova_layer(int64_t l) const { return is_moe_layer(l) && n_value_expert > 0; }
};

/// Fills `g` from the metadata shard `meta`.  False with `err` (the file and the first problem) when the guard refuses
/// the file, a key has the wrong type, or the values do not describe a model llama.cpp's graph can build.
inline bool geometry_from_gguf(const strata::GgufFile& meta, Geometry& g, std::string& err) {
    err = strata::check_architecture(meta);
    if (err.empty() && strata::model_family(meta) != strata::ModelFamily::K2Horizon)
        err = "not a K2 Horizon file (architecture '" +
              std::string(meta.get("general.architecture") ? meta.get("general.architecture")->s : "") + "')";
    if (!err.empty()) {
        err = meta.path() + ": " + err;
        return false;
    }
    g = Geometry{};
    const std::string p = "k2-horizon.";
    std::string bad_key;
    // an integer key into `dst` (absent: `dst` keeps its default)
    auto num = [&](const char* key, int64_t& dst) {
        const strata::MetaValue* v = meta.get(p + key);
        if (!v) return;
        if (!v->is_num() || v->type == strata::MetaType::F32 || v->type == strata::MetaType::F64) {
            if (bad_key.empty()) bad_key = p + key + " is not an integer";
            return;
        }
        dst = (int64_t) v->u;
    };
    auto real = [&](const char* key, double& dst) {
        const strata::MetaValue* v = meta.get(p + key);
        if (!v) return;
        if (!v->is_num()) {
            if (bad_key.empty()) bad_key = p + key + " is not a number";
            return;
        }
        dst = v->num();
    };
    num("block_count", g.n_layers);
    num("embedding_length", g.n_embd);
    num("attention.head_count", g.n_head);
    num("attention.head_count_kv", g.n_head_kv);
    real("rope.freq_base", g.rope_freq_base);
    real("attention.layer_norm_rms_epsilon", g.rms_eps);
    num("attention.group_norm_groups", g.n_norm_groups);
    num("feed_forward_length", g.n_ff_dense);
    num("expert_count", g.n_expert);
    num("expert_used_count", g.n_expert_used);
    num("attention.value_expert_count", g.n_value_expert);
    num("attention.value_expert_used_count", g.n_value_expert_used);

    char buf[320];
    auto bad = [&](const std::string& what) {
        std::snprintf(buf, sizeof buf, "%s: %s", meta.path().c_str(), what.c_str());
        err = buf;
        return false;
    };
    if (g.n_head <= 0) return bad("k2-horizon.attention.head_count must be positive");
    int64_t key_length = g.n_head > 0 ? g.n_embd / g.n_head : 0, value_length = key_length;
    num("attention.key_length", key_length);
    num("attention.value_length", value_length);
    g.head_dim = key_length;
    g.n_rot = key_length;
    num("rope.dimension_count", g.n_rot);
    if (g.n_norm_groups == 0) g.n_norm_groups = 1;   // llama.cpp: 0 (or absent) means one group

    if (g.n_expert > 0) {
        if (const strata::MetaValue* v = meta.get(p + "expert_feed_forward_length"))
            if (v->type == strata::MetaType::ARRAY)
                return bad("k2-horizon.expert_feed_forward_length is a per-layer array; Strata reads one width only");
        num("expert_feed_forward_length", g.n_ff_exp);
        num("leading_dense_block_count", g.n_dense_lead);
        num("expert_shared_count", g.n_expert_shared);
        num("expert_shared_feed_forward_length", g.n_ff_shexp);
        int64_t every = 1, gating = 0, norm = 0;
        num("moe_every_n_layers", every);
        num("expert_gating_func", gating);
        if (const strata::MetaValue* v = meta.get(p + "expert_weights_norm")) {
            if (!v->is_num()) return bad("k2-horizon.expert_weights_norm is not a bool");
            norm = (int64_t) v->u;
        }
        real("expert_weights_scale", g.expert_weights_scale);
        g.expert_weights_norm = norm != 0;
        if (gating == 0) gating = (int64_t) Gating::Sigmoid;   // llama.cpp: NONE -> SIGMOID for this family
        if (gating != (int64_t) Gating::Softmax && gating != (int64_t) Gating::Sigmoid)
            return bad("k2-horizon.expert_gating_func = " + std::to_string(gating) + ", only 1 (softmax) and 2 (sigmoid) are known");
        g.gating = (Gating) gating;
        if (every != 1)
            return bad("k2-horizon.moe_every_n_layers = " + std::to_string(every) +
                       ": llama.cpp's K2 Horizon graph makes every layer from leading_dense_block_count on MoE, so only 1 is read");
        if (g.n_expert_shared > 0 && g.n_ff_shexp == 0) g.n_ff_shexp = g.n_ff_exp * g.n_expert_shared;
    }
    if (const strata::MetaValue* v = meta.get("tokenizer.ggml.tokens"))
        if (v->type == strata::MetaType::ARRAY) g.n_vocab = (int64_t) v->count;
    if (!bad_key.empty()) return bad(bad_key);

    // the shapes llama.cpp's graph needs (its GGML_ASSERTs and create_tensor shapes), refused here by key
    if (g.n_layers <= 0 || g.n_embd <= 0 || g.n_head_kv <= 0 || g.head_dim <= 0 || g.n_vocab <= 0 || g.rms_eps <= 0 ||
        g.rope_freq_base <= 0)
        return bad("a geometry key is zero or out of range");
    if (value_length != key_length) return bad("k2-horizon.attention.key_length != attention.value_length");
    if (g.n_head % g.n_head_kv) return bad("attention.head_count is not a multiple of attention.head_count_kv");
    if (g.n_rot <= 0 || g.n_rot > g.head_dim || g.n_rot % 2)
        return bad("k2-horizon.rope.dimension_count must be even and at most the head width");
    if (g.n_norm_groups < 0 || g.n_embd % g.n_norm_groups)
        return bad("k2-horizon.attention.group_norm_groups does not divide embedding_length");
    if (g.n_expert > 0) {
        if (g.n_expert_used <= 0 || g.n_expert_used > g.n_expert)
            return bad("k2-horizon.expert_used_count must be 1 .. expert_count");
        if (g.n_ff_exp <= 0) return bad("K2 Horizon MoE layer requires expert_feed_forward_length");
        if (g.n_dense_lead < 0 || g.n_dense_lead > g.n_layers)
            return bad("k2-horizon.leading_dense_block_count is larger than block_count");
    } else if (g.n_expert < 0) {
        return bad("k2-horizon.expert_count is negative");
    }
    if (g.n_dense_lead > 0 || g.n_expert == 0)
        if (g.n_ff_dense <= 0) return bad("k2-horizon.feed_forward_length must be positive for the dense layers");
    if (g.n_value_expert > 0) {
        if (g.n_value_expert_used <= 0 || g.n_value_expert_used > g.n_value_expert)
            return bad("k2-horizon.attention.value_expert_used_count must be 1 .. value_expert_count");
    } else if (g.n_value_expert_used != 0 || g.n_value_expert < 0) {
        return bad("k2-horizon.attention.value_expert_used_count without value experts");
    }
    return true;
}

/// One line for logs: the geometry as llama.cpp would print it.
inline std::string describe(const Geometry& g) {
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "K2 Horizon: %lld layers (%lld dense, FF %lld), hidden %lld (norm groups %lld, eps %g), attention %lld/%lld x "
                  "%lld (rope %lld, base %g), MoE %lld experts top-%lld FF %lld + %lld shared FF %lld (%s, norm %d, scale %g), "
                  "MoVA %lld value experts top-%lld, vocab %lld",
                  (long long) g.n_layers, (long long) (g.n_expert > 0 ? g.n_dense_lead : g.n_layers), (long long) g.n_ff_dense,
                  (long long) g.n_embd, (long long) g.n_norm_groups, g.rms_eps, (long long) g.n_head, (long long) g.n_head_kv,
                  (long long) g.head_dim, (long long) g.n_rot, g.rope_freq_base, (long long) g.n_expert,
                  (long long) g.n_expert_used, (long long) g.n_ff_exp, (long long) g.n_expert_shared, (long long) g.n_ff_shexp,
                  g.gating == Gating::Sigmoid ? "sigmoid" : "softmax", g.expert_weights_norm ? 1 : 0, g.expert_weights_scale,
                  (long long) g.n_value_expert, (long long) g.n_value_expert_used, (long long) g.n_vocab);
    return buf;
}

}  // namespace strata::k2h
