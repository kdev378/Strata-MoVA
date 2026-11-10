// src/k2h/k2h_ref.cpp - strata-k2h-ref: K2 Horizon's CPU reference forward (include/strata/k2h/cpu_ref.hpp), compared
// with llama.cpp tensor by tensor.
//
//   strata-k2h-ref MODEL.gguf --ref DIR [--layers N] [--threads N]
//       runs the token ids of DIR/tokens.txt (written by tools/k2h/llama_dump.cpp) through layers 0 .. N-1 and compares
//       every intermediate tensor DIR/index.txt names with Strata's: the largest |difference|, that over the largest
//       |llama.cpp value| ("rel"), and the RMS of the difference over the RMS of the values ("rms rel")
//   --isolate: each layer l > 0 starts from llama.cpp's own l_out-(l-1) instead of Strata's, so a layer's difference is
//       its own and not the earlier layers' carried forward
//   --split K: tokens 0 .. K-1 as one batch, then every later token alone against the KV cache (the decode path); the
//       outputs are joined and compared as one
//   --logits: every layer and the head; result_norm / result_output are the last token's (as llama.cpp's default)
//   strata-k2h-ref MODEL.gguf --tokens 0,864,7169 [--layers N]
//       the same forward without a reference: each tensor's RMS and its non-finite count
//
// Exit 0 when no tensor holds a non-finite value (and, with --ref, every tensor the reference has was compared); 1
// otherwise.  CPU only.
#include "strata/k2h/cpu_ref.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct RefTensor {
    std::vector<float> data;
    int64_t elements = 0;
};

std::vector<int32_t> parse_ids(const std::string& s) {
    std::vector<int32_t> v;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
        if (!item.empty()) v.push_back((int32_t) std::strtol(item.c_str(), nullptr, 10));
    return v;
}

std::map<std::string, RefTensor> read_reference(const std::string& dir, std::vector<int32_t>& tokens) {
    std::map<std::string, RefTensor> out;
    std::ifstream tf(dir + "/tokens.txt");
    std::string line;
    if (!std::getline(tf, line)) throw std::runtime_error(dir + "/tokens.txt: missing");
    tokens = parse_ids(line);
    std::ifstream idx(dir + "/index.txt");
    std::string name, type;
    int64_t ne[4];
    while (idx >> name >> type >> ne[0] >> ne[1] >> ne[2] >> ne[3]) {
        RefTensor r;
        r.elements = ne[0] * ne[1] * ne[2] * ne[3];
        r.data.resize((size_t) r.elements);
        std::ifstream f(dir + "/" + name + ".f32", std::ios::binary);
        if (!f.read((char*) r.data.data(), (std::streamsize) (r.elements * 4))) throw std::runtime_error(dir + "/" + name + ".f32: short");
        out[name] = std::move(r);
    }
    return out;
}
}  // namespace

int main(int argc, char** argv) {
    std::string path, ref_dir, ids;
    int64_t layers = -1;
    bool isolate = false;
    int64_t split = 0;
    bool logits = false;
    int threads = (int) std::max(1u, std::thread::hardware_concurrency());
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--ref" && i + 1 < argc) ref_dir = argv[++i];
        else if (a == "--tokens" && i + 1 < argc) ids = argv[++i];
        else if (a == "--layers" && i + 1 < argc) layers = std::atoll(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--isolate") isolate = true;
        else if (a == "--logits") logits = true;
        else if (a == "--split" && i + 1 < argc) split = std::atoll(argv[++i]);
        else if (path.empty() && a[0] != '-') path = a;
        else {
            std::fprintf(stderr, "usage: strata-k2h-ref MODEL.gguf (--ref DIR | --tokens IDS) [--layers N] [--threads N]\n");
            return 1;
        }
    }
    if (path.empty() || (ref_dir.empty() && ids.empty())) {
        std::fprintf(stderr, "usage: strata-k2h-ref MODEL.gguf (--ref DIR | --tokens IDS) [--layers N] [--threads N]\n");
        return 1;
    }
    try {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<int32_t> tokens = parse_ids(ids);
        std::map<std::string, RefTensor> ref;
        if (!ref_dir.empty()) ref = read_reference(ref_dir, tokens);
        const strata::GgufModel m = strata::GgufModel::open(path);
        strata::k2h::Geometry g;
        strata::k2h::Weights w;
        std::string err;
        if (!strata::k2h::geometry_from_gguf(m.meta(), g, err) || !strata::k2h::map_weights(m, g, w, err)) {
            std::printf("k2h-ref: %s\n", err.c_str());
            return 1;
        }
        if (logits) layers = g.n_layers;
        if (layers < 0) layers = g.n_dense_lead;
        std::string tl;
        for (size_t i = 0; i < tokens.size(); ++i) tl += (i ? "," : "") + std::to_string(tokens[i]);
        if (isolate && ref.empty()) {
            std::printf("k2h-ref: --isolate needs --ref\n");
            return 1;
        }
        std::printf("k2h-ref: %lld layers of %s, tokens [%s] at positions 0 .. %zu, %d threads%s%s\n", (long long) layers,
                    path.c_str(), tl.c_str(), tokens.size() - 1, threads, ref.empty() ? "" : (", against " + ref_dir).c_str(),
                    isolate ? ", each layer from llama.cpp's input (--isolate)" : "");
        if (split > 0 && split < (int64_t) tokens.size())
            std::printf("  tokens 0 .. %lld as one batch, then one at a time (--split)\n", (long long) split - 1);
        std::printf("  %-14s %10s %12s %12s %10s %10s %6s\n", "tensor", "elements", "max |ref|", "max |diff|", "rel", "rms rel",
                    "nan");
        size_t compared = 0, nonfinite_total = 0;
        double worst_rel = 0;
        std::string worst;
        strata::k2h::ref::Forward fwd(g, w, (int64_t) tokens.size(), threads);
        // every intermediate tensor, its tokens in order across the pieces the prompt is run in
        std::map<std::string, std::vector<float>> got;
        std::vector<std::string> order;
        const strata::k2h::ref::Hook hook = [&](const std::string& name, const float* d, int64_t T, int64_t per) {
            auto& v = got[name];
            if (v.empty()) order.push_back(name);
            v.insert(v.end(), d, d + T * per);
        };
        const int64_t T = (int64_t) tokens.size();
        // the pieces: tokens [0, split) together, then one at a time (decoding against the KV cache); all at once by default
        std::vector<std::pair<int64_t, int64_t>> pieces;
        if (split <= 0 || split >= T) pieces.push_back({0, T});
        else {
            pieces.push_back({0, split});
            for (int64_t t = split; t < T; ++t) pieces.push_back({t, 1});
        }
        std::string stopped;
        try {
        for (const auto& [p0, n] : pieces) {
            std::vector<int32_t> part(tokens.begin() + p0, tokens.begin() + p0 + n);
            std::vector<float> x((size_t) (n * g.n_embd));
            fwd.embed(part, x.data());
            for (int64_t l = 0; l < layers; ++l) {
                if (isolate && l > 0) {
                    const auto it = ref.find("l_out-" + std::to_string(l - 1));
                    if (it == ref.end() || it->second.elements != T * g.n_embd) {
                        std::printf("k2h-ref: --isolate: the reference has no l_out-%lld\n", (long long) (l - 1));
                        return 1;
                    }
                    x.assign(it->second.data.begin() + p0 * g.n_embd, it->second.data.begin() + (p0 + n) * g.n_embd);
                }
                fwd.run_layer(l, x, n, p0, hook);
            }
            if (logits && p0 + n == (int64_t) tokens.size()) fwd.head(x, n, hook);
        }
        } catch (const std::exception& e) {
            stopped = e.what();   // compare what was computed before the stop, then fail
            std::printf("  the forward stopped: %s\n", stopped.c_str());
        }
        for (const std::string& name : order) {
            const std::vector<float>& v = got[name];
            int64_t n = (int64_t) v.size();
            const float* d = v.data();
            int64_t rows = T;   // the tokens this comparison covers
            // llama.cpp keeps only the last token's row from the last layer's ffn_inp on (inp_out_ids): compare that row
            if (const auto rt = ref.find(name); rt != ref.end() && rt->second.elements * T == n && T > 1) {
                d += n - rt->second.elements;
                n = rt->second.elements;
                rows = 1;
            }
            size_t nonfinite = 0;
            double ss = 0;
            for (int64_t i = 0; i < n; ++i) {
                if (!std::isfinite(d[i])) ++nonfinite;
                else ss += (double) d[i] * d[i];
            }
            nonfinite_total += nonfinite;
            const auto it = ref.find(name);
            if (it == ref.end()) {
                std::printf("  %-14s %10lld %12s %12s %10s %10s %6zu   rms %.5g\n", name.c_str(), (long long) n, "-", "-", "-", "-",
                            nonfinite, std::sqrt(ss / (double) n));
                continue;
            }
            const RefTensor& r = it->second;
            if (r.elements != n) {
                std::printf("  %-14s element count %lld, llama.cpp's %lld: not compared\n", name.c_str(), (long long) n,
                            (long long) r.elements);
                continue;
            }
            double max_ref = 0, max_diff = 0, ssd = 0, ssr = 0;
            for (int64_t i = 0; i < n; ++i) {
                const double a = d[i], b = r.data[(size_t) i];
                max_ref = std::max(max_ref, std::fabs(b));
                max_diff = std::max(max_diff, std::fabs(a - b));
                ssd += (a - b) * (a - b);
                ssr += b * b;
            }
            const double rel = max_ref > 0 ? max_diff / max_ref : max_diff;
            const double rms_rel = ssr > 0 ? std::sqrt(ssd / ssr) : std::sqrt(ssd);
            std::string sets;
            if (name == "result_output") {   // the next-token logits: argmax and the top 5 of each side
                const std::vector<int32_t> mine = strata::k2h::ref::argsort_desc(d, n), theirs = strata::k2h::ref::argsort_desc(r.data.data(), n);
                sets = "   argmax " + std::to_string(mine[0]) + (mine[0] == theirs[0] ? " = " : " != ") + std::to_string(theirs[0]) +
                       "; top 5 Strata";
                for (int i = 0; i < 5; ++i) sets += " " + std::to_string(mine[(size_t) i]);
                sets += " / llama.cpp";
                for (int i = 0; i < 5; ++i) sets += " " + std::to_string(theirs[(size_t) i]);
            }
            if (name.find("_topk-") != std::string::npos) {   // the routers' choices: tokens whose chosen set differs
                const int64_t k = n / rows;
                int64_t differ = 0, order = 0;
                for (int64_t t = 0; t < rows; ++t) {
                    std::vector<float> a(d + t * k, d + (t + 1) * k), b(r.data.begin() + t * k, r.data.begin() + (t + 1) * k);
                    order += a != b ? 1 : 0;
                    std::sort(a.begin(), a.end());
                    std::sort(b.begin(), b.end());
                    differ += a != b ? 1 : 0;
                }
                sets = "   chosen set differs on " + std::to_string(differ) + " of " + std::to_string(rows) + " tokens, order on " +
                       std::to_string(order);
            }
            std::printf("  %-14s %10lld %12.5g %12.5g %10.3g %10.3g %6zu%s\n", name.c_str(), (long long) n, max_ref, max_diff, rel,
                        rms_rel, nonfinite, sets.c_str());
            ++compared;
            if (rel > worst_rel) {
                worst_rel = rel;
                worst = name;
            }
        }
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("k2h-ref: %zu tensors compared with llama.cpp (of %zu it holds), %zu non-finite values; worst rel %.3g (%s); %.1f s\n",
                    compared, ref.size(), nonfinite_total, worst_rel, worst.empty() ? "-" : worst.c_str(), secs);
        if (nonfinite_total || !stopped.empty()) return 1;
        if (!ref.empty() && compared != ref.size()) return 1;
        return 0;
    } catch (const std::exception& e) {
        std::printf("k2h-ref: %s\n", e.what());
        return 1;
    }
}
