// tests/core/k2h_geometry_test.cpp - K2 Horizon (GGUF architecture "k2-horizon") recognised: the guards and the geometry
// from metadata, on synthetic GGUFs (no model, no GPU).
//
//   * model_family / the architecture guard accept "k2-horizon"; the engine guard refuses it by name (recognised, no
//     engine path yet), so every build keeps refusing the file before anything runs;
//   * strata::k2h::geometry_from_gguf reads IFM/K2-Horizon-MoVA-36B-A4B-GGUF's metadata (the values below are the
//     Q4_K_M file's header, read 2026-11-09) and fills llama.cpp b11454's defaults for the keys a file leaves out;
//   * inconsistent values are refused, naming the key.
//
// `k2h_geometry_test --real FILE`: the same view of a real file's header (the file may be cut after its tensor
// directory: only the metadata is read).
#include "gguf_fixture.hpp"

#include "strata/core/family_layout.hpp"
#include "strata/k2h/geometry.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
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
               ("strata-k2h-geometry-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

using fixture::Kv;

/// K2-Horizon-MoVA-36B-A4B-Q4_K_M.gguf's metadata (the tokenizer cut to `vocab` tokens).
std::vector<Kv> k2_metadata(uint64_t vocab) {
    const std::string p = "k2-horizon.";
    std::vector<Kv> kv = {
        fixture::str("general.architecture", "k2-horizon"),
        fixture::u32(p + "block_count", 48),
        fixture::u32(p + "context_length", 524288),
        fixture::u32(p + "embedding_length", 2560),
        fixture::u32(p + "feed_forward_length", 6144),
        fixture::u32(p + "attention.head_count", 32),
        fixture::u32(p + "attention.head_count_kv", 8),
        fixture::f32(p + "rope.freq_base", 10000000.0f),
        fixture::f32(p + "attention.layer_norm_rms_epsilon", 1e-6f),
        fixture::u32(p + "expert_count", 100),
        fixture::u32(p + "expert_used_count", 8),
        fixture::u32(p + "attention.key_length", 128),
        fixture::u32(p + "attention.value_length", 128),
        fixture::u32(p + "attention.group_norm_groups", 2),
        fixture::u32(p + "rope.dimension_count", 128),
        fixture::u32(p + "expert_feed_forward_length", 768),
        fixture::u32(p + "leading_dense_block_count", 3),
        fixture::u32(p + "moe_every_n_layers", 1),
        fixture::u32(p + "expert_shared_count", 1),
        fixture::boolean(p + "expert_weights_norm", true),
        fixture::u32(p + "expert_shared_feed_forward_length", 768),
        fixture::f32(p + "expert_weights_scale", 2.5f),
        fixture::u32(p + "expert_gating_func", 2),
        fixture::u32(p + "attention.value_expert_count", 64),
        fixture::u32(p + "attention.value_expert_used_count", 4),
    };
    std::vector<std::string> tokens;
    for (uint64_t i = 0; i < vocab; ++i) tokens.push_back("t" + std::to_string(i));
    kv.push_back(fixture::str_array("tokenizer.ggml.tokens", tokens));
    return kv;
}

std::vector<Kv> without(std::vector<Kv> kv, const std::string& key) {
    for (auto it = kv.begin(); it != kv.end(); ++it)
        if (it->key == key) { kv.erase(it); break; }
    return kv;
}

std::vector<Kv> with_u32(std::vector<Kv> kv, const std::string& key, uint64_t v) {
    kv = without(kv, key);
    kv.push_back(fixture::u32(key, v));
    return kv;
}

std::string geometry_error(const fs::path& dir, const std::vector<Kv>& kv, strata::k2h::Geometry* out = nullptr) {
    const fs::path f = dir / "k2.gguf";
    fixture::write(f, kv, {});
    strata::k2h::Geometry g;
    std::string err;
    const bool ok = strata::k2h::geometry_from_gguf(strata::GgufFile(f.string()), g, err);
    if (out) *out = g;
    return ok ? std::string() : err;
}

int real_file(const std::string& path) {
    try {
        const strata::GgufFile f(path);
        const std::string arch = strata::check_architecture(f);
        std::printf("  architecture guard: %s\n  engine guard: %s\n", arch.empty() ? "ok" : arch.c_str(),
                    strata::check_engine_architecture(f).c_str());
        strata::k2h::Geometry g;
        std::string err;
        if (!strata::k2h::geometry_from_gguf(f, g, err)) {
            std::printf("  geometry: %s\n", err.c_str());
            return 1;
        }
        std::printf("  %s\n", strata::k2h::describe(g).c_str());
        return 0;
    } catch (const std::exception& e) {
        std::printf("  refused: %s\n", e.what());
        return 1;
    }
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--real") return real_file(argv[2]);
    std::printf("k2h_geometry_test\n");
    TempDir d;
    const std::string p = "k2-horizon.";
    {
        // the guards
        const fs::path f = d.path / "guard.gguf";
        fixture::write(f, k2_metadata(8), {});
        const strata::GgufFile g(f.string());
        check(strata::model_family(g) == strata::ModelFamily::K2Horizon, "model_family: k2-horizon is K2 Horizon");
        check(std::string(strata::model_family_arch(strata::ModelFamily::K2Horizon)) == "k2-horizon",
              "the family's architecture string and key prefix is k2-horizon");
        check(strata::check_architecture(g).empty(), "the architecture guard accepts k2-horizon");
        const std::string e = strata::check_engine_architecture(g);
        check(e.find("K2 Horizon") != std::string::npos && e.find("recognised") != std::string::npos &&
                  e.find("no build of Strata runs it yet") != std::string::npos,
              "the engine guard refuses it: recognised, not run by any build yet");
        const fs::path m = d.path / "miss.gguf";
        fixture::write(m, without(k2_metadata(8), p + "rope.freq_base"), {});
        check(strata::check_architecture(strata::GgufFile(m.string())).find("rope.freq_base") != std::string::npos,
              "a k2-horizon file without rope.freq_base is refused, naming the key");
        strata::core::ModelGeometry qg;
        int64_t k = 0;
        std::string qe;
        check(!strata::core::geometry_from_gguf(g, qg, k, qe) && qe.find("strata::k2h::geometry_from_gguf") != std::string::npos,
              "the Qwen geometry reader refuses a K2 file and names K2's own reader");
    }
    {
        // K2-Horizon-MoVA-36B-A4B's geometry
        strata::k2h::Geometry g;
        const std::string e = geometry_error(d.path, k2_metadata(250624), &g);
        check(e.empty(), "K2-Horizon-MoVA-36B-A4B's metadata reads" + (e.empty() ? std::string() : ": " + e));
        check(g.n_layers == 48 && g.n_embd == 2560 && g.n_vocab == 250624, "48 layers, hidden 2560, vocab 250624");
        check(g.n_head == 32 && g.n_head_kv == 8 && g.head_dim == 128 && g.n_rot == 128 && g.n_embd_kv() == 1024,
              "attention: 32 Q / 8 KV heads x 128, RoPE over all 128, K/V width 1024");
        check(g.n_norm_groups == 2 && g.rms_eps > 0.99e-6 && g.rms_eps < 1.01e-6 && g.rope_freq_base == 1e7,
              "RMS norm in 2 groups, eps 1e-6, RoPE base 1e7");
        check(g.n_dense_lead == 3 && g.n_ff_dense == 6144 && !g.is_moe_layer(2) && g.is_moe_layer(3) && g.is_moe_layer(47),
              "layers 0-2 dense (FF 6144), 3-47 MoE");
        check(g.n_expert == 100 && g.n_expert_used == 8 && g.n_ff_exp == 768 && g.n_expert_shared == 1 && g.n_ff_shexp == 768,
              "MoE: 100 experts top-8, FF 768, one shared expert FF 768");
        check(g.gating == strata::k2h::Gating::Sigmoid && g.expert_weights_norm && g.expert_weights_scale == 2.5,
              "router: sigmoid, top-k weights renormalised, scale 2.5");
        check(g.n_value_expert == 64 && g.n_value_expert_used == 4 && !g.is_mova_layer(2) && g.is_mova_layer(3),
              "MoVA: 64 value experts top-4 on the MoE layers only");
        std::printf("  %s\n", strata::k2h::describe(g).c_str());
    }
    {
        // llama.cpp's defaults for the optional keys
        std::vector<Kv> kv = k2_metadata(8);
        for (const char* key : {"attention.group_norm_groups", "attention.key_length", "attention.value_length",
                                "rope.dimension_count", "expert_gating_func", "expert_shared_feed_forward_length",
                                "attention.value_expert_count", "attention.value_expert_used_count"})
            kv = without(kv, p + key);
        strata::k2h::Geometry g;
        const std::string e = geometry_error(d.path, kv, &g);
        check(e.empty() && g.n_norm_groups == 1 && g.head_dim == 80 && g.n_rot == 80 && g.gating == strata::k2h::Gating::Sigmoid &&
                  g.n_ff_shexp == 768 && g.n_value_expert == 0 && !g.is_mova_layer(3),
              "absent keys: 1 norm group, head width n_embd / n_head, sigmoid, shared FF = FF x count, no MoVA" +
                  (e.empty() ? std::string() : ": " + e));
        kv = with_u32(k2_metadata(8), p + "expert_gating_func", 0);
        check(geometry_error(d.path, kv, &g).empty() && g.gating == strata::k2h::Gating::Sigmoid,
              "expert_gating_func 0 (none) is sigmoid, as llama.cpp sets it");
    }
    {
        // refusals
        auto refused = [&](const std::vector<Kv>& kv, const std::string& needle, const std::string& what) {
            const std::string e = geometry_error(d.path, kv);
            check(!e.empty() && e.find(needle) != std::string::npos, what + (e.empty() ? " (accepted)" : " (" + e.substr(e.rfind(": ") + 2) + ")"));
        };
        refused(with_u32(k2_metadata(8), p + "moe_every_n_layers", 2), "moe_every_n_layers", "moe_every_n_layers 2 is refused");
        refused(with_u32(k2_metadata(8), p + "attention.value_expert_used_count", 65), "value_expert_used_count",
                "more value experts used than there are");
        refused(with_u32(k2_metadata(8), p + "expert_used_count", 101), "expert_used_count", "more experts used than there are");
        refused(with_u32(k2_metadata(8), p + "attention.group_norm_groups", 3), "group_norm_groups",
                "norm groups that do not divide the hidden size");
        refused(with_u32(k2_metadata(8), p + "attention.value_length", 64), "value_length", "key and value widths that differ");
        refused(with_u32(k2_metadata(8), p + "attention.head_count_kv", 5), "head_count_kv", "Q heads not a multiple of KV heads");
        refused(with_u32(k2_metadata(8), p + "rope.dimension_count", 129), "rope.dimension_count", "RoPE wider than the head");
        refused(with_u32(k2_metadata(8), p + "leading_dense_block_count", 49), "leading_dense_block_count",
                "more dense layers than layers");
        refused(with_u32(k2_metadata(8), p + "expert_gating_func", 3), "expert_gating_func", "an unknown gating function");
        refused(without(k2_metadata(8), p + "expert_feed_forward_length"), "expert_feed_forward_length",
                "a MoE file without expert_feed_forward_length");
        std::vector<Kv> kv = without(k2_metadata(8), p + "attention.value_expert_count");
        refused(kv, "value_expert_used_count", "value experts used without a value expert count");
        kv = without(k2_metadata(8), p + "block_count");
        kv.push_back(fixture::f32(p + "block_count", 48.0f));
        refused(kv, "block_count", "a float where an integer belongs");
        const fs::path q = d.path / "qwen.gguf";
        fixture::write(q, {fixture::str("general.architecture", "llama")}, {});
        strata::k2h::Geometry g;
        std::string e;
        check(!strata::k2h::geometry_from_gguf(strata::GgufFile(q.string()), g, e) && e.find("'llama'") != std::string::npos,
              "another architecture is refused by name");
    }
    std::printf(g_fail ? "FAILED (%d)\n" : "ALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
