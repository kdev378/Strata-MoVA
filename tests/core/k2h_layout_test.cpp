// tests/core/k2h_layout_test.cpp - K2 Horizon's layout table and weight mapping (include/strata/k2h/layout.hpp) on a tiny
// synthetic model (every tensor F32, a few KB; no model, no GPU), and the dequantization dispatch of k2h/dequant_rows.hpp.
//
//   * a complete file maps: every tensor lands in its slot, dense layers get attn_v + ffn_*, MoE layers get the experts,
//     the shared expert and (MoVA) attn_v_gate / attn_v_exps;
//   * optional tensors may be absent (output.weight -> tied to token_embd, attn_gate, the norms, the biases), and the
//     per-head norms are accepted in both shapes llama.cpp takes;
//   * a missing required tensor, a wrong shape, a plain attn_v on a MoVA layer and a tensor the table does not name are
//     refused, naming the tensor.
#include "gguf_fixture.hpp"

#include "strata/k2h/dequant_rows.hpp"
#include "strata/k2h/layout.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-100s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-k2h-layout-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

using fixture::Kv;
using fixture::Tensor;

// a tiny K2 Horizon: 4 layers (1 dense), hidden 64 in 2 norm groups, 4 Q / 2 KV heads x 16, 4 experts top-2 FF 16 + 1 shared,
// 4 value experts top-2, vocab 32
std::vector<Kv> tiny_metadata() {
    const std::string p = "k2-horizon.";
    std::vector<Kv> kv = {
        fixture::str("general.architecture", "k2-horizon"),   fixture::u32(p + "block_count", 4),
        fixture::u32(p + "embedding_length", 64),              fixture::u32(p + "feed_forward_length", 48),
        fixture::u32(p + "attention.head_count", 4),           fixture::u32(p + "attention.head_count_kv", 2),
        fixture::f32(p + "rope.freq_base", 10000.0f),          fixture::f32(p + "attention.layer_norm_rms_epsilon", 1e-6f),
        fixture::u32(p + "attention.key_length", 16),          fixture::u32(p + "attention.value_length", 16),
        fixture::u32(p + "attention.group_norm_groups", 2),    fixture::u32(p + "expert_count", 4),
        fixture::u32(p + "expert_used_count", 2),              fixture::u32(p + "expert_feed_forward_length", 16),
        fixture::u32(p + "leading_dense_block_count", 1),      fixture::u32(p + "expert_shared_count", 1),
        fixture::boolean(p + "expert_weights_norm", true),     fixture::f32(p + "expert_weights_scale", 2.5f),
        fixture::u32(p + "expert_gating_func", 2),             fixture::u32(p + "attention.value_expert_count", 4),
        fixture::u32(p + "attention.value_expert_used_count", 2),
    };
    std::vector<std::string> tokens;
    for (int i = 0; i < 32; ++i) tokens.push_back("t" + std::to_string(i));
    kv.push_back(fixture::str_array("tokenizer.ggml.tokens", tokens));
    return kv;
}

/// Every entry of the table (optional ones too), in its first shape, as F32.
std::vector<Tensor> tiny_tensors(const strata::k2h::Geometry& g) {
    strata::k2h::Weights w;
    std::vector<Tensor> ts;
    uint8_t seed = 1;
    for (const auto& want : strata::k2h::tensor_table(g, w)) {
        std::vector<uint64_t> s(want.shapes.front().begin(), want.shapes.front().end());
        ts.push_back({want.name, s, 0, seed++});
    }
    return ts;
}

struct Mapped {
    std::string err;
    strata::k2h::Weights w;
    strata::k2h::MapReport r;
};

Mapped map_with(const fs::path& dir, const std::function<void(std::vector<Tensor>&)>& mutate) {
    const fs::path f = dir / "k2-tiny.gguf";
    const auto kv = tiny_metadata();
    fixture::write(f, kv, {});
    strata::k2h::Geometry g;
    Mapped out;
    if (!strata::k2h::geometry_from_gguf(strata::GgufFile(f.string()), g, out.err)) return out;
    std::vector<Tensor> ts = tiny_tensors(g);
    if (mutate) mutate(ts);
    fixture::write(f, kv, ts);
    const strata::GgufModel m({f.string()});
    if (!strata::k2h::map_weights(m, g, out.w, out.err, &out.r)) return out;
    // the refs must outlive m only for these checks: read what is needed now
    out.err.clear();
    return out;
}

void erase_named(std::vector<Tensor>& ts, const std::string& name) {
    for (auto it = ts.begin(); it != ts.end(); ++it)
        if (it->name == name) { ts.erase(it); return; }
}
}  // namespace

int main() {
    std::printf("k2h_layout_test\n");
    TempDir d;
    {
        const fs::path f = d.path / "k2-full.gguf";
        const auto kv = tiny_metadata();
        fixture::write(f, kv, {});
        strata::k2h::Geometry g;
        std::string err;
        check(strata::k2h::geometry_from_gguf(strata::GgufFile(f.string()), g, err), "the tiny model's geometry reads " + err);
        const std::vector<Tensor> ts = tiny_tensors(g);
        fixture::write(f, kv, ts);
        const strata::GgufModel m({f.string()});
        strata::k2h::Weights w;
        strata::k2h::MapReport r;
        const bool ok = strata::k2h::map_weights(m, g, w, err, &r);
        check(ok && r.found == ts.size() && r.optional_absent == 0, "a complete file maps every tensor (" +
                                                                         std::to_string(r.found) + " of " + std::to_string(ts.size()) + ") " + err);
        const auto& L0 = w.layers[0];
        const auto& L2 = w.layers[2];
        check(L0.attn_v && !L0.attn_v_exps && L0.ffn_up && !L0.ffn_up_exps, "layer 0 (dense): attn_v and the dense FFN");
        check(!L2.attn_v && L2.attn_v_exps && L2.attn_v_gate && L2.attn_v_gate_b && L2.ffn_up_exps && L2.ffn_gate_inp &&
                  L2.ffn_exp_probs_b && L2.ffn_down_shexp && !L2.ffn_up,
              "layer 2 (MoE + MoVA): value experts, their router and bias, the experts, the selection bias, the shared expert");
        check(L2.attn_v_exps.name() == "blk.2.attn_v_exps.weight" && L2.attn_v_exps.info->shape[2] == 4 &&
                  L2.attn_v_exps.info->shape[1] == 32,
              "blk.2.attn_v_exps is [64, 32, 4]: hidden -> 2 KV heads x 16, for 4 value experts");
        check(!w.output_tied && w.output.name() == "output.weight", "output.weight is its own");
        bool same = false;
        for (const Tensor& t : ts)
            if (t.name == "blk.2.attn_v_exps.weight") {
                const uint8_t* p = L2.attn_v_exps.data();
                same = L2.attn_v_exps.bytes() == 64ull * 32 * 4 * 4;
                for (uint64_t i = 0; same && i < L2.attn_v_exps.bytes(); ++i) same = p[i] == fixture::pattern(t.seed, i);
            }
        check(same, "a mapped tensor's bytes are the file's, to the last byte");
    }
    {
        Mapped r = map_with(d.path, [](std::vector<Tensor>& ts) {
            for (const char* n : {"output.weight", "blk.1.attn_gate.weight", "blk.1.attn_q_norm.weight", "blk.1.attn_k_norm.weight",
                                  "blk.1.attn_v_gate.bias", "blk.1.exp_probs_b.bias"})
                erase_named(ts, n);
        });
        check(r.err.empty() && r.r.optional_absent == 6 && r.w.output_tied && !r.w.layers[1].attn_gate &&
                  !r.w.layers[1].attn_v_gate_b,
              "optional tensors may be absent; no output.weight ties the head to token_embd " + r.err);
        r = map_with(d.path, [](std::vector<Tensor>& ts) {
            for (auto& t : ts)
                if (t.name == "blk.3.attn_q_norm.weight") t.shape = {64};   // [16, 4] stored flat
        });
        check(r.err.empty() && r.w.layers[3].attn_q_norm, "a per-head norm stored flat is accepted, as llama.cpp reshapes it " + r.err);
    }
    {
        auto refused = [&](const std::function<void(std::vector<Tensor>&)>& mutate, const std::string& needle, const std::string& what) {
            const Mapped r = map_with(d.path, mutate);
            check(!r.err.empty() && r.err.find(needle) != std::string::npos, what + " (" + (r.err.empty() ? "accepted" : r.err) + ")");
        };
        refused([](auto& ts) { erase_named(ts, "blk.2.attn_v_exps.weight"); }, "missing blk.2.attn_v_exps.weight",
                "a missing value-expert tensor");
        refused([](auto& ts) { erase_named(ts, "token_embd.weight"); }, "missing token_embd.weight", "a missing token embedding");
        refused([](auto& ts) { erase_named(ts, "blk.0.ffn_down.weight"); }, "missing blk.0.ffn_down.weight", "a missing dense FFN tensor");
        refused(
            [](auto& ts) {
                for (auto& t : ts)
                    if (t.name == "blk.3.ffn_down_exps.weight") t.shape = {64, 16, 4};
            },
            "blk.3.ffn_down_exps.weight is [64, 16, 4]", "a transposed expert tensor, naming both shapes");
        refused([](auto& ts) { ts.push_back({"blk.2.attn_v.weight", {64, 32}, 0, 99}); }, "blk.2.attn_v.weight is not in",
                "a plain attn_v on a MoVA layer");
        refused([](auto& ts) { ts.push_back({"rope_freqs.weight", {8}, 0, 98}); }, "rope_freqs.weight is not in",
                "a tensor the table does not name");
    }
    {
        // the dequantization dispatch: one Q8_0 block (d = 0.5, q = -16 .. 15) and F32
        uint8_t blk[34] = {0x00, 0x38};   // fp16 0.5
        for (int i = 0; i < 32; ++i) blk[2 + i] = (uint8_t) (int8_t) (i - 16);
        float out[32] = {};
        bool ok = strata::k2h::dequantize_rows(8, blk, out, 32);
        for (int i = 0; ok && i < 32; ++i) ok = out[i] == 0.5f * (float) (i - 16);
        check(ok, "dequantize_rows: a Q8_0 block reads as d * q");
        const float in[4] = {1.5f, -2.0f, 0.0f, 3.25f};
        float o4[4] = {};
        check(strata::k2h::dequantize_rows(0, (const uint8_t*) in, o4, 4) && o4[0] == 1.5f && o4[3] == 3.25f, "dequantize_rows: F32");
        check(!strata::k2h::dequantize_rows(12, blk, out, 32), "dequantize_rows: a count that is not whole blocks is refused");
        check(strata::k2h::can_dequantize(12) && strata::k2h::can_dequantize(14) && !strata::k2h::can_dequantize(42),
              "can_dequantize: Q4_K and Q6_K yes, Q2_0 no");
    }
    std::printf(g_fail ? "FAILED (%d)\n" : "ALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
