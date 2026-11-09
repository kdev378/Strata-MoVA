// src/k2h/k2h_check.cpp - strata-k2h-check: a K2 Horizon GGUF, read the way the K2 engine path will read it.
//
//   strata-k2h-check MODEL.gguf            the guard, the geometry, and every tensor mapped through the layout table
//                                          (include/strata/k2h/layout.hpp): present, its shape, inside its file, no
//                                          tensor the table does not name
//   strata-k2h-check MODEL.gguf --read     and then every byte of every tensor dequantized on the CPU
//                                          (include/strata/k2h/dequant_rows.hpp), each value checked finite; a table per
//                                          tensor kind (count, types, bytes, min, max, mean |x|) goes to stdout
//   --threads N                            readers (default: the hardware's)
//
// Exit 0 when everything maps (and, with --read, reads and is finite); 1 otherwise.  CPU only, no GPU: the step 3 check
// of the K2-Horizon-MoVA-36B-A4B work, before any layer runs.
#include "strata/k2h/dequant_rows.hpp"
#include "strata/k2h/geometry.hpp"
#include "strata/k2h/layout.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Stats {
    uint64_t tensors = 0, elements = 0, bytes = 0, nonfinite = 0;
    double min = INFINITY, max = -INFINITY, abs_sum = 0;
    std::map<std::string, uint64_t> types;
    void add(const Stats& o) {
        tensors += o.tensors;
        elements += o.elements;
        bytes += o.bytes;
        nonfinite += o.nonfinite;
        min = std::min(min, o.min);
        max = std::max(max, o.max);
        abs_sum += o.abs_sum;
        for (const auto& [k, v] : o.types) types[k] += v;
    }
};

/// Every element of `t` dequantized in pieces of whole rows (at most ~4M elements); false when the type cannot be read.
bool read_tensor(const strata::k2h::TensorRef& t, Stats& s, std::string& err) {
    const strata::TensorInfo& info = *t.info;
    const int64_t row = (int64_t) info.shape[0];
    const int64_t rows = (int64_t) (info.elements() / (uint64_t) row);
    int be = 0, bb = 0;
    if (!strata::block_geometry(info.type, be, bb) || !strata::k2h::can_dequantize(info.type)) {
        err = info.name + ": no dequantizer for " + info.type_name();
        return false;
    }
    const int64_t row_bytes = row / be * bb;
    const int64_t per = std::max<int64_t>(1, (int64_t) (4 << 20) / row);
    std::vector<float> buf((size_t) (per * row));
    const uint8_t* src = t.data();
    for (int64_t r0 = 0; r0 < rows; r0 += per) {
        const int64_t n = std::min(per, rows - r0);
        if (!strata::k2h::dequantize_rows(info.type, src + r0 * row_bytes, buf.data(), n * row)) {
            err = info.name + ": dequantization refused";
            return false;
        }
        for (int64_t i = 0; i < n * row; ++i) {
            const float v = buf[(size_t) i];
            if (!std::isfinite(v)) {
                ++s.nonfinite;
                continue;
            }
            s.min = std::min(s.min, (double) v);
            s.max = std::max(s.max, (double) v);
            s.abs_sum += std::fabs(v);
        }
    }
    s.tensors = 1;
    s.elements = info.elements();
    s.bytes = t.bytes();
    s.types[info.type_name()] = 1;
    return true;
}

std::string kind_of(const std::string& name) { return std::regex_replace(name, std::regex("^blk\\.\\d+\\."), "blk.N."); }

void usage() {
    std::fprintf(stderr, "usage: strata-k2h-check MODEL.gguf [--read] [--threads N]\n");
}
}  // namespace

int main(int argc, char** argv) {
    std::string path;
    bool read = false;
    unsigned threads = std::max(1u, std::thread::hardware_concurrency());
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--read") read = true;
        else if (a == "--threads" && i + 1 < argc) threads = (unsigned) std::max(1, std::atoi(argv[++i]));
        else if (a == "--help" || a == "-h") { usage(); return 0; }
        else if (path.empty() && a[0] != '-') path = a;
        else { usage(); return 1; }
    }
    if (path.empty()) { usage(); return 1; }
    try {
        const auto t0 = std::chrono::steady_clock::now();
        const strata::GgufModel m = strata::GgufModel::open(path);
        uint64_t file_bytes = 0;
        size_t file_tensors = 0;
        for (size_t s = 0; s < m.size(); ++s) {
            file_bytes += m.shard(s).file_size();
            file_tensors += m.shard(s).tensors().size();
        }
        std::printf("k2h-check: %s: %zu shard(s), %zu tensors, %.3f GB\n", path.c_str(), m.size(), file_tensors, file_bytes / 1e9);
        std::printf("  engine guard: %s\n", strata::check_engine_architecture(m.meta()).c_str());
        strata::k2h::Geometry g;
        std::string err;
        if (!strata::k2h::geometry_from_gguf(m.meta(), g, err)) {
            std::printf("  geometry: FAIL - %s\n", err.c_str());
            return 1;
        }
        std::printf("  geometry: %s\n", strata::k2h::describe(g).c_str());
        strata::k2h::Weights w;
        strata::k2h::MapReport r;
        if (!strata::k2h::map_weights(m, g, w, err, &r)) {
            std::printf("  layout: FAIL - %s\n", err.c_str());
            return 1;
        }
        std::string types;
        for (const auto& [k, v] : r.types) types += (types.empty() ? "" : ", ") + k + " " + std::to_string(v);
        std::printf("  layout: ok - %zu of %zu tensors mapped (%zu optional entries absent), %.3f GiB of weights; %s\n",
                    r.found, file_tensors, r.optional_absent, r.bytes / 1073741824.0, types.c_str());
        size_t mova = 0, moe = 0;
        for (int64_t l = 0; l < g.n_layers; ++l) {
            mova += w.layers[(size_t) l].attn_v_exps ? 1 : 0;
            moe += w.layers[(size_t) l].ffn_up_exps ? 1 : 0;
        }
        const strata::k2h::LayerWeights& L3 = w.layers[(size_t) std::min<int64_t>(g.n_dense_lead, g.n_layers - 1)];
        std::printf("  mapping: %zu MoE layers, %zu MoVA layers, output %s, attention gate %s, Q/K norm %s, value router bias %s, "
                    "expert selection bias %s\n",
                    moe, mova, w.output_tied ? "tied to token_embd" : "its own", L3.attn_gate ? "present" : "absent",
                    L3.attn_q_norm ? "present" : "absent", L3.attn_v_gate_b ? "present" : "absent",
                    L3.ffn_exp_probs_b ? "present" : "absent");
        if (r.found != file_tensors) {
            std::printf("  layout: FAIL - %zu tensors in the file, %zu mapped\n", file_tensors, r.found);
            return 1;
        }
        if (!read) return 0;

        // every mapped tensor, read to its last byte
        std::vector<strata::k2h::TensorRef> all;
        auto push = [&](const strata::k2h::TensorRef& t) {
            if (t) all.push_back(t);
        };
        push(w.token_embd);
        push(w.output_norm);
        if (!w.output_tied) push(w.output);
        for (const auto& L : w.layers)
            for (const strata::k2h::TensorRef* t :
                 {&L.attn_norm, &L.attn_q, &L.attn_k, &L.attn_v, &L.attn_output, &L.attn_gate, &L.attn_q_norm, &L.attn_k_norm,
                  &L.attn_v_gate, &L.attn_v_gate_b, &L.attn_v_exps, &L.ffn_norm, &L.ffn_gate, &L.ffn_up, &L.ffn_down,
                  &L.ffn_gate_inp, &L.ffn_exp_probs_b, &L.ffn_gate_exps, &L.ffn_up_exps, &L.ffn_down_exps, &L.ffn_gate_shexp,
                  &L.ffn_up_shexp, &L.ffn_down_shexp})
                push(*t);
        // largest first, so the threads finish together
        std::sort(all.begin(), all.end(), [](const auto& a, const auto& b) { return a.bytes() > b.bytes(); });
        std::map<std::string, Stats> kinds;
        std::mutex mu;
        std::atomic<size_t> next{0}, done{0};
        std::string first_err;
        std::vector<std::thread> pool;
        for (unsigned i = 0; i < threads; ++i)
            pool.emplace_back([&] {
                for (size_t k; (k = next.fetch_add(1)) < all.size();) {
                    Stats s;
                    std::string e;
                    const bool ok = read_tensor(all[k], s, e);
                    std::lock_guard<std::mutex> lock(mu);
                    if (!ok && first_err.empty()) first_err = e;
                    if (ok) kinds[kind_of(all[k].name())].add(s);
                    ++done;
                }
            });
        for (auto& t : pool) t.join();
        if (!first_err.empty()) {
            std::printf("  read: FAIL - %s\n", first_err.c_str());
            return 1;
        }
        std::printf("  read: every tensor dequantized (%zu tensors, %u threads):\n", all.size(), threads);
        std::printf("    %-32s %7s %10s %12s %12s %12s %11s  %s\n", "tensor", "count", "GiB", "min", "max", "mean |x|", "non-finite",
                    "types");
        Stats total;
        for (const auto& [k, s] : kinds) {
            std::string ty;
            for (const auto& [t, n] : s.types) ty += (ty.empty() ? "" : ", ") + t + " " + std::to_string(n);
            std::printf("    %-32s %7llu %10.4f %12.5g %12.5g %12.5g %11llu  %s\n", k.c_str(), (unsigned long long) s.tensors,
                        s.bytes / 1073741824.0, s.min, s.max, s.abs_sum / (double) std::max<uint64_t>(1, s.elements - s.nonfinite),
                        (unsigned long long) s.nonfinite, ty.c_str());
            total.add(s);
        }
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("  read: %llu tensors, %.3f G weights, %.3f GiB, %llu non-finite values, %.1f s\n",
                    (unsigned long long) total.tensors, total.elements / 1e9, total.bytes / 1073741824.0,
                    (unsigned long long) total.nonfinite, secs);
        if (total.nonfinite) {
            std::printf("  read: FAIL - non-finite weights\n");
            return 1;
        }
        std::printf("  read: ok\n");
        return 0;
    } catch (const std::exception& e) {
        std::printf("k2h-check: refused: %s\n", e.what());
        return 1;
    }
}
