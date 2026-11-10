// src/k2h/k2h_engine.cpp - strata-k2h: K2 Horizon (GGUF architecture k2-horizon) on the CPU.  The minimum viable engine of
// the K2-Horizon-MoVA-36B-A4B work (docs/K2_HORIZON.md): it runs the model, it is not fast.
//
// What it does, in llama.cpp b11454's order (src/models/k2-horizon.cpp; the CPU reference include/strata/k2h/cpu_ref.hpp
// is the same model written for checking, and strata-k2h-ref compares it with llama.cpp tensor by tensor):
//
//   * every matrix product through ggml-cpu's own kernels: the activations rounded to the weight type's vec_dot_type
//     with its from_float (Q8_K for Q4_K / Q6_K), then its vec_dot per weight row - the functions llama.cpp's CPU backend
//     calls, so a product here is llama.cpp's product;
//   * the weights read in place from the GGUF's mmap (include/strata/k2h/layout.hpp maps them); a routed expert's rows
//     are a slice of its layer's 3-D tensor, so no repack and no copy;
//   * the routers (MoVA's 64 / top 4 and the FFN's 100 / top 8) on the CPU with cpu_ref's route(): sigmoid, the selection
//     bias for the choice only, ggml's CPU top-k order (argsort_desc), renormalised, x 2.5;
//   * the tokens of a batch that chose the same expert go through it together (its rows read once);
//   * an F32 KV cache for --ctx positions; a request whose ids start with the previous request's reuses that prefix.
//
// Modes:
//   strata-k2h --model M.gguf --tokens IDS [--n N] [--ctx C] [--threads T]
//       the ids as a prompt, then N greedy tokens; prints the first step's top 5 and the generated ids with timings
//   strata-k2h --model M.gguf --serve [--ctx C] [--threads T] [--eos IDS]
//       serve/server.py's engine protocol: READY <ctx> stop; GEN <max_new> [key=value ...] <id,id,...> -> PP / T <id> / DONE;
//       STOP ends a generation.  Sampling keys: temperature, top_k, top_p, min_p, seed (greedy without a temperature).
//
// Not here (yet): the GPU, vision, batch slots, drafting, session files.
#include "strata/k2h/cpu_ref.hpp"
#include "strata/k2h/geometry.hpp"
#include "strata/k2h/layout.hpp"

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace R = strata::k2h::ref;
using strata::k2h::Geometry;
using strata::k2h::LayerWeights;
using strata::k2h::TensorRef;
using strata::k2h::Weights;
using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

// ------------------------------------------------------------------------------------------------ a fork-join pool
class Pool {
public:
    explicit Pool(int n) : n_(std::max(1, n)) {
        for (int i = 1; i < n_; ++i) workers_.emplace_back([this, i] { loop(i); });
    }
    ~Pool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            quit_ = true;
            ++epoch_;
        }
        cv_.notify_all();
        for (auto& w : workers_) w.join();
    }
    int size() const { return n_; }
    /// fn(i0, i1) over [0, n) in n_ slices; returns when every slice is done.
    void run(int64_t n, const std::function<void(int64_t, int64_t)>& fn) {
        if (n <= 0) return;
        if (n_ == 1 || n < 2) {
            fn(0, n);
            return;
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            job_ = &fn;
            n_items_ = n;
            pending_ = n_ - 1;
            ++epoch_;
        }
        cv_.notify_all();
        slice(0, n, fn);
        std::unique_lock<std::mutex> lk(mu_);
        done_cv_.wait(lk, [this] { return pending_ == 0; });
        job_ = nullptr;
    }

private:
    void slice(int id, int64_t n, const std::function<void(int64_t, int64_t)>& fn) const {
        const int64_t per = (n + n_ - 1) / n_, i0 = std::min<int64_t>(n, id * per), i1 = std::min<int64_t>(n, i0 + per);
        if (i0 < i1) fn(i0, i1);
    }
    void loop(int id) {
        uint64_t seen = 0;
        for (;;) {
            const std::function<void(int64_t, int64_t)>* job;
            int64_t n;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return epoch_ != seen; });
                seen = epoch_;
                if (quit_) return;
                job = job_;
                n = n_items_;
            }
            slice(id, n, *job);
            std::lock_guard<std::mutex> lk(mu_);
            if (--pending_ == 0) done_cv_.notify_one();
        }
    }
    int n_;
    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    const std::function<void(int64_t, int64_t)>* job_ = nullptr;
    int64_t n_items_ = 0;
    int pending_ = 0;
    uint64_t epoch_ = 0;
    bool quit_ = false;
};

// ------------------------------------------------------------------------------------------------ the model
class Engine {
public:
    Engine(const Geometry& g, const Weights& w, int64_t n_ctx, int threads) : g_(g), w_(w), n_ctx_(n_ctx), pool_(threads) {
        ggml_cpu_init();
        const size_t kv = (size_t) (n_ctx * g.n_embd_kv());
        k_.assign((size_t) g.n_layers, std::vector<float>(kv));
        v_.assign((size_t) g.n_layers, std::vector<float>(kv));
        for (int64_t l = 0; l < g.n_layers; ++l) {
            const LayerWeights& L = w.layers[(size_t) l];
            norms_attn_.push_back(R::vector_of(L.attn_norm));
            norms_ffn_.push_back(R::vector_of(L.ffn_norm));
            v_bias_.push_back(L.attn_v_gate_b ? R::vector_of(L.attn_v_gate_b) : std::vector<float>());
            e_bias_.push_back(L.ffn_exp_probs_b ? R::vector_of(L.ffn_exp_probs_b) : std::vector<float>());
        }
        norm_out_ = R::vector_of(w.output_norm);
    }

    int64_t ctx() const { return n_ctx_; }

    /// tokens at positions pos0 .. through every layer; the logits of the last one into `logits` (n_vocab).
    void forward(const std::vector<int32_t>& tokens, int64_t pos0, std::vector<float>& logits) {
        const int64_t T = (int64_t) tokens.size(), N = g_.n_embd;
        if (T == 0 || pos0 + T > n_ctx_) throw std::runtime_error("the context is full");
        std::vector<float> x((size_t) (T * N));
        embed(tokens, x.data());
        for (int64_t l = 0; l < g_.n_layers; ++l) layer(l, x, T, pos0);
        std::vector<float> h((size_t) N);
        R::group_rms_norm(&x[(size_t) ((T - 1) * N)], norm_out_.data(), h.data(), N, g_.n_norm_groups, (float) g_.rms_eps);
        logits.resize((size_t) g_.n_vocab);
        mv(w_.output, 0, g_.n_vocab, h.data(), 1, logits.data());
    }

private:
    void embed(const std::vector<int32_t>& tokens, float* x) const {
        const TensorRef& E = w_.token_embd;
        const size_t rb = ggml_row_size((ggml_type) E.type(), g_.n_embd);
        for (size_t t = 0; t < tokens.size(); ++t) {
            if (tokens[t] < 0 || tokens[t] >= g_.n_vocab) throw std::runtime_error("token id out of range");
            strata::k2h::dequantize_rows(E.type(), E.data() + (size_t) tokens[t] * rb, x + (int64_t) t * g_.n_embd, g_.n_embd);
        }
    }

    /// y[t][j] = W[row0 + j] . x[t] for j < n_out, T tokens, through ggml-cpu (activations to vec_dot_type, then vec_dot).
    void mv(const TensorRef& W, int64_t row0, int64_t n_out, const float* x, int64_t T, float* y) {
        const int64_t n_in = (int64_t) W.info->shape[0];
        const ggml_type wt = (ggml_type) W.type();
        const ggml_type_traits_cpu* tr = ggml_get_type_traits_cpu(wt);
        if (!tr || !tr->vec_dot) throw std::runtime_error(W.name() + ": no ggml-cpu dot product for " + W.info->type_name());
        const ggml_type vt = tr->vec_dot_type;
        const size_t arow = ggml_row_size(vt, n_in), wrow = ggml_row_size(wt, n_in);
        const uint8_t* act = (const uint8_t*) x;
        if (vt != GGML_TYPE_F32) {
            act_.resize((size_t) T * arow);
            const ggml_from_float_t ff = ggml_get_type_traits_cpu(vt)->from_float;
            for (int64_t t = 0; t < T; ++t) ff(x + t * n_in, act_.data() + (size_t) t * arow, n_in);
            act = act_.data();
        }
        const uint8_t* base = W.data() + (size_t) row0 * wrow;
        pool_.run(n_out, [&](int64_t j0, int64_t j1) {
            for (int64_t j = j0; j < j1; ++j)
                for (int64_t t = 0; t < T; ++t)
                    tr->vec_dot((int) n_in, &y[t * n_out + j], 0, base + (size_t) j * wrow, 0, act + (size_t) t * arow, 0, 1);
        });
    }

    /// For each expert some token chose: the rows of those tokens through it together.  `fn(e, rows, m, out)` gets the
    /// m gathered input rows and writes m output rows; out[t * K + k] receives the row of the token's k-th choice.
    void by_expert(const std::vector<int32_t>& ids, int64_t T, int64_t K, const float* in, int64_t n_in, int64_t n_out,
                   std::vector<float>& out, const std::function<void(int32_t, const float*, int64_t, float*)>& fn) {
        std::map<int32_t, std::vector<int64_t>> users;   // expert -> entries t * K + k, in order
        for (int64_t i = 0; i < T * K; ++i) users[ids[(size_t) i]].push_back(i);
        out.resize((size_t) (T * K * n_out));
        std::vector<float> xin, yout;
        for (const auto& [e, entries] : users) {
            const int64_t m = (int64_t) entries.size();
            xin.resize((size_t) (m * n_in));
            for (int64_t r = 0; r < m; ++r)
                std::copy(in + (entries[(size_t) r] / K) * n_in, in + (entries[(size_t) r] / K + 1) * n_in, xin.begin() + r * n_in);
            yout.resize((size_t) (m * n_out));
            fn(e, xin.data(), m, yout.data());
            for (int64_t r = 0; r < m; ++r)
                std::copy(yout.begin() + r * n_out, yout.begin() + (r + 1) * n_out, out.begin() + entries[(size_t) r] * n_out);
        }
    }

    void layer(int64_t l, std::vector<float>& x, int64_t T, int64_t pos0) {
        const LayerWeights& L = w_.layers[(size_t) l];
        const int64_t N = g_.n_embd, HD = g_.head_dim, H = g_.n_head, HKV = g_.n_head_kv, KV = g_.n_embd_kv(), HQ = H * HD;
        const float eps = (float) g_.rms_eps;
        std::vector<float> h((size_t) (T * N));
        for (int64_t t = 0; t < T; ++t)
            R::group_rms_norm(&x[(size_t) (t * N)], norms_attn_[(size_t) l].data(), &h[(size_t) (t * N)], N, g_.n_norm_groups, eps);
        std::vector<float> q((size_t) (T * HQ)), k((size_t) (T * KV)), v((size_t) (T * KV));
        mv(L.attn_q, 0, HQ, h.data(), T, q.data());
        mv(L.attn_k, 0, KV, h.data(), T, k.data());
        if (L.attn_v_exps) {
            // MoVA: route, each chosen value expert's silu(W x) times its weight, added in the router's order
            const int64_t NV = g_.n_value_expert, KK = g_.n_value_expert_used;
            std::vector<float> logits((size_t) (T * NV));
            mv(L.attn_v_gate, 0, NV, h.data(), T, logits.data());
            const R::Routed r = R::route(logits.data(), v_bias_[(size_t) l].empty() ? nullptr : v_bias_[(size_t) l].data(), T, NV, KK,
                                         g_.gating, g_.expert_weights_norm, g_.expert_weights_scale,
                                         [](const char*, const std::vector<float>&) {});
            std::vector<float> parts;
            by_expert(r.ids, T, KK, h.data(), N, KV, parts, [&](int32_t e, const float* xin, int64_t m, float* yout) {
                mv(L.attn_v_exps, (int64_t) e * KV, KV, xin, m, yout);
            });
            for (int64_t t = 0; t < T; ++t)
                for (int64_t i = 0; i < KV; ++i) {
                    float acc = R::silu(parts[(size_t) (t * KK * KV + i)]) * r.weights[(size_t) (t * KK)];
                    for (int64_t kk = 1; kk < KK; ++kk)
                        acc += R::silu(parts[(size_t) ((t * KK + kk) * KV + i)]) * r.weights[(size_t) (t * KK + kk)];
                    v[(size_t) (t * KV + i)] = acc;
                }
        } else {
            mv(L.attn_v, 0, KV, h.data(), T, v.data());
        }
        if (L.attn_q_norm || L.attn_k_norm) throw std::runtime_error("per-head Q/K norms are not in strata-k2h yet");
        for (int64_t t = 0; t < T; ++t) {
            R::rope_neox(&q[(size_t) (t * HQ)], H, HD, g_.n_rot, pos0 + t, (float) g_.rope_freq_base);
            R::rope_neox(&k[(size_t) (t * KV)], HKV, HD, g_.n_rot, pos0 + t, (float) g_.rope_freq_base);
        }
        std::vector<float>& kc = k_[(size_t) l];
        std::vector<float>& vc = v_[(size_t) l];
        std::copy(k.begin(), k.end(), kc.begin() + pos0 * KV);
        std::copy(v.begin(), v.end(), vc.begin() + pos0 * KV);

        // attention: (token, head) pairs across the pool
        std::vector<float> att((size_t) (T * HQ));
        const float kq_scale = 1.0f / sqrtf((float) HD);
        const int64_t group = H / HKV;
        pool_.run(T * H, [&](int64_t a0, int64_t a1) {
            std::vector<float> sc;
            for (int64_t a = a0; a < a1; ++a) {
                const int64_t t = a / H, hh = a % H, n_pos = pos0 + t + 1, kvh = hh / group;
                sc.resize((size_t) n_pos);
                const float* qv = &q[(size_t) (t * HQ + hh * HD)];
                float mx = -INFINITY;
                for (int64_t p = 0; p < n_pos; ++p) {
                    const float* kv = &kc[(size_t) (p * KV + kvh * HD)];
                    float dot = 0;
                    for (int64_t i = 0; i < HD; ++i) dot += qv[i] * kv[i];
                    sc[(size_t) p] = dot * kq_scale;
                    mx = std::max(mx, sc[(size_t) p]);
                }
                double sum = 0;
                for (int64_t p = 0; p < n_pos; ++p) sum += (sc[(size_t) p] = expf(sc[(size_t) p] - mx));
                const float inv = (float) (1.0 / sum);
                float* o = &att[(size_t) (t * HQ + hh * HD)];
                std::fill(o, o + HD, 0.0f);
                for (int64_t p = 0; p < n_pos; ++p) {
                    const float w = sc[(size_t) p] * inv;
                    const float* vv = &vc[(size_t) (p * KV + kvh * HD)];
                    for (int64_t i = 0; i < HD; ++i) o[i] += w * vv[i];
                }
            }
        });
        if (L.attn_gate) {
            std::vector<float> gate((size_t) (T * HQ));
            mv(L.attn_gate, 0, HQ, h.data(), T, gate.data());
            for (size_t i = 0; i < gate.size(); ++i) att[i] *= R::softplus2(gate[i]);
        }
        std::vector<float> o((size_t) (T * N));
        mv(L.attn_output, 0, N, att.data(), T, o.data());
        for (size_t i = 0; i < x.size(); ++i) x[i] += o[i];

        // FFN
        for (int64_t t = 0; t < T; ++t)
            R::group_rms_norm(&x[(size_t) (t * N)], norms_ffn_[(size_t) l].data(), &h[(size_t) (t * N)], N, g_.n_norm_groups, eps);
        if (g_.is_moe_layer(l)) {
            const int64_t NE = g_.n_expert, KK = g_.n_expert_used, F = g_.n_ff_exp;
            std::vector<float> logits((size_t) (T * NE));
            mv(L.ffn_gate_inp, 0, NE, h.data(), T, logits.data());
            const R::Routed r = R::route(logits.data(), e_bias_[(size_t) l].empty() ? nullptr : e_bias_[(size_t) l].data(), T, NE, KK,
                                         g_.gating, g_.expert_weights_norm, g_.expert_weights_scale,
                                         [](const char*, const std::vector<float>&) {}, true);
            std::vector<float> down;
            by_expert(r.ids, T, KK, h.data(), N, N, down, [&](int32_t e, const float* xin, int64_t m, float* yout) {
                swiglu(L.ffn_gate_exps, L.ffn_up_exps, L.ffn_down_exps, e, F, xin, m, yout);
            });
            for (int64_t t = 0; t < T; ++t)
                for (int64_t i = 0; i < N; ++i) {
                    float acc = down[(size_t) (t * KK * N + i)] * r.weights[(size_t) (t * KK)];
                    for (int64_t kk = 1; kk < KK; ++kk) acc += down[(size_t) ((t * KK + kk) * N + i)] * r.weights[(size_t) (t * KK + kk)];
                    o[(size_t) (t * N + i)] = acc;
                }
            if (L.ffn_up_shexp) {
                std::vector<float> sh((size_t) (T * N));
                swiglu(L.ffn_gate_shexp, L.ffn_up_shexp, L.ffn_down_shexp, 0, g_.n_ff_shexp, h.data(), T, sh.data());
                for (size_t i = 0; i < o.size(); ++i) o[i] += sh[i];
            }
        } else {
            swiglu(L.ffn_gate, L.ffn_up, L.ffn_down, 0, g_.n_ff_dense, h.data(), T, o.data());
        }
        for (size_t i = 0; i < x.size(); ++i) x[i] += o[i];
    }

    /// down(silu(gate(x)) * up(x)) through expert e's rows (e = 0 for a 2-D FFN), T rows of x into y [T][n_embd].
    void swiglu(const TensorRef& gate, const TensorRef& up, const TensorRef& down, int64_t e, int64_t F, const float* x, int64_t T, float* y) {
        const int64_t N = g_.n_embd;
        std::vector<float> gt((size_t) (T * F)), u((size_t) (T * F));
        mv(up, e * F, F, x, T, u.data());
        mv(gate, e * F, F, x, T, gt.data());
        for (size_t i = 0; i < gt.size(); ++i) gt[i] = R::silu(gt[i]) * u[i];
        mv(down, e * N, N, gt.data(), T, y);
    }

    const Geometry& g_;
    const Weights& w_;
    int64_t n_ctx_;
    Pool pool_;
    std::vector<uint8_t> act_;
    std::vector<std::vector<float>> k_, v_;
    std::vector<std::vector<float>> norms_attn_, norms_ffn_, v_bias_, e_bias_;
    std::vector<float> norm_out_;
};

// ------------------------------------------------------------------------------------------------ sampling
struct Sampling {
    float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 0;
    uint64_t seed = 0;
    bool seeded = false;
};

int32_t argmax(const std::vector<float>& lg) {
    int32_t best = 0;
    for (int32_t j = 1; j < (int32_t) lg.size(); ++j)
        if (lg[(size_t) j] > lg[(size_t) best]) best = j;   // the lowest id on equal logits
    return best;
}

int32_t sample(const std::vector<float>& lg, const Sampling& s, std::mt19937_64& rng) {
    if (s.temperature <= 0.0f) return argmax(lg);
    std::vector<int32_t> idx((size_t) lg.size());
    for (int32_t j = 0; j < (int32_t) idx.size(); ++j) idx[(size_t) j] = j;
    const size_t keep = s.top_k > 0 ? std::min<size_t>((size_t) s.top_k, idx.size()) : idx.size();
    std::partial_sort(idx.begin(), idx.begin() + (std::ptrdiff_t) keep, idx.end(),
                      [&](int32_t a, int32_t b) { return lg[(size_t) a] > lg[(size_t) b]; });
    idx.resize(keep);
    std::vector<double> p(idx.size());
    const double mx = lg[(size_t) idx[0]];
    double sum = 0;
    for (size_t i = 0; i < idx.size(); ++i) sum += (p[i] = std::exp(((double) lg[(size_t) idx[i]] - mx) / s.temperature));
    for (double& v : p) v /= sum;
    size_t n = p.size();
    if (s.min_p > 0)
        while (n > 1 && p[n - 1] < s.min_p * p[0]) --n;
    if (s.top_p < 1.0f) {
        double c = 0;
        for (size_t i = 0; i < n; ++i)
            if ((c += p[i]) >= s.top_p) {
                n = i + 1;
                break;
            }
    }
    std::discrete_distribution<size_t> d(p.begin(), p.begin() + (std::ptrdiff_t) n);
    return idx[d(rng)];
}

std::vector<int32_t> parse_ids(const char* s, bool& ok) {
    std::vector<int32_t> v;
    ok = true;
    while (*s == ' ') ++s;
    while (*s) {
        char* e = nullptr;
        const long x = std::strtol(s, &e, 10);
        if (e == s) {
            ok = false;
            break;
        }
        v.push_back((int32_t) x);
        s = e;
        if (*s == ',') ++s;
        else if (*s == '\0' || *s == ' ' || *s == '\n') break;
        else {
            ok = false;
            break;
        }
    }
    return v;
}

// ------------------------------------------------------------------------------------------------ the serve protocol
class Serve {
public:
    Serve(Engine& e, std::vector<int32_t> eos) : E(e), eos_(std::move(eos)) {}

    int run() {
        std::printf("READY %lld stop\n", (long long) E.ctx());
        std::fflush(stdout);
        std::thread reader([this] {
            std::string l;
            while (std::getline(std::cin, l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                if (l == "STOP") {
                    stop_.store(true);
                    continue;
                }
                std::lock_guard<std::mutex> lk(mu_);
                lines_.push_back(l);
                cv_.notify_one();
            }
            std::lock_guard<std::mutex> lk(mu_);
            eof_ = true;
            cv_.notify_one();
        });
        reader.detach();
        for (;;) {
            std::string line;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return eof_ || !lines_.empty(); });
                if (lines_.empty()) return 0;
                line = lines_.front();
                lines_.pop_front();
            }
            if (line.empty()) continue;
            if (line.rfind("GEN ", 0) == 0) request(line);
            else std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...>\n");
            std::fflush(stdout);
        }
    }

private:
    bool is_eos(int32_t t) const { return std::find(eos_.begin(), eos_.end(), t) != eos_.end(); }

    void request(const std::string& line) {
        stop_.store(false);
        const char* p = line.c_str() + 4;
        char* end = nullptr;
        const long long max_new = std::strtoll(p, &end, 10);
        if (end == p) {
            std::printf("ERR bad request: max_new\n");
            return;
        }
        p = end;
        Sampling smp;
        for (;;) {
            while (*p == ' ') ++p;
            const char* t0 = p;
            while (*p && *p != ' ') ++p;
            const std::string tk(t0, (size_t) (p - t0));
            const size_t eq = tk.find('=');
            if (tk.empty() || eq == std::string::npos) {
                p = t0;
                break;
            }
            const std::string key = tk.substr(0, eq);
            const char* v = tk.c_str() + eq + 1;
            if (key == "temperature") smp.temperature = std::strtof(v, nullptr);
            else if (key == "top_p") smp.top_p = std::strtof(v, nullptr);
            else if (key == "top_k") smp.top_k = std::atoi(v);
            else if (key == "min_p") smp.min_p = std::strtof(v, nullptr);
            else if (key == "seed") {
                smp.seed = std::strtoull(v, nullptr, 10);
                smp.seeded = true;
            }
            // other keys (penalties, img_keys, ...) are accepted and not applied
        }
        bool ok = false;
        const std::vector<int32_t> ids = parse_ids(p, ok);
        if (!ok || ids.empty() || max_new < 0) {
            std::printf("ERR bad request: max_new or the ids\n");
            return;
        }
        if ((long long) ids.size() + max_new > (long long) E.ctx()) {
            std::printf("ERR prompt (%zu tokens) + max_new (%lld) exceeds the context (%lld)\n", ids.size(), max_new, (long long) E.ctx());
            return;
        }
        // the cached prefix: what the KV cache holds of these ids (at least the last id is run again for its logits)
        size_t reuse = 0;
        while (reuse < cached_.size() && reuse + 1 < ids.size() && cached_[reuse] == ids[reuse]) ++reuse;
        std::mt19937_64 rng(smp.seeded ? smp.seed : (uint64_t) Clock::now().time_since_epoch().count());
        const auto t0 = Clock::now();
        std::vector<float> lg;
        const size_t chunk = 256;
        size_t done = reuse;
        cached_.resize(reuse);
        try {
            while (done < ids.size()) {
                const size_t n = std::min(chunk, ids.size() - done);
                E.forward(std::vector<int32_t>(ids.begin() + (std::ptrdiff_t) done, ids.begin() + (std::ptrdiff_t) (done + n)),
                          (int64_t) done, lg);
                cached_.insert(cached_.end(), ids.begin() + (std::ptrdiff_t) done, ids.begin() + (std::ptrdiff_t) (done + n));
                done += n;
                const double ms = ms_since(t0);
                std::printf("PP %zu %zu %.0f %.1f\n", done - reuse, ids.size() - reuse, ms, ms > 0 ? 1000.0 * (double) (done - reuse) / ms : 0.0);
                std::fflush(stdout);
                if (stop_.load()) break;
            }
        } catch (const std::exception& e) {
            cached_.clear();
            std::printf("ERR %s\n", e.what());
            return;
        }
        const double prompt_ms = ms_since(t0);
        const auto d0 = Clock::now();
        long long produced = 0;
        const char* finish = "length";
        if (stop_.load()) finish = "cancel";
        else {
            for (long long i = 0; i < max_new; ++i) {
                const int32_t nx = sample(lg, smp, rng);
                std::printf("T %d\n", nx);
                std::fflush(stdout);
                ++produced;
                if (is_eos(nx)) {
                    finish = "stop";
                    break;
                }
                if (stop_.load()) {
                    finish = "cancel";
                    break;
                }
                if (i + 1 == max_new || (int64_t) cached_.size() + 1 > E.ctx()) break;
                try {
                    E.forward({nx}, (int64_t) cached_.size(), lg);
                } catch (const std::exception& e) {
                    cached_.clear();
                    std::printf("ERR %s\n", e.what());
                    return;
                }
                cached_.push_back(nx);
            }
        }
        const double decode_ms = ms_since(d0);
        std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu 0 0 0 0 0.0 %lld 0\n", produced, ids.size(), prompt_ms, decode_ms, finish, reuse,
                    (long long) (ids.size() - reuse));
        std::fprintf(stderr, "strata-k2h: prompt %zu tokens (%zu reused) in %.0f ms, %lld generated in %.0f ms (%.2f tok/s), %s\n", ids.size(),
                     reuse, prompt_ms, produced, decode_ms, decode_ms > 0 ? 1000.0 * (double) produced / decode_ms : 0.0, finish);
    }

    Engine& E;
    std::vector<int32_t> eos_;
    std::vector<int32_t> cached_;   // the ids whose K/V the cache holds, positions 0 ..
    std::atomic<bool> stop_{false};
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::string> lines_;
    bool eof_ = false;
};

void usage() {
    std::fprintf(stderr,
                 "usage: strata-k2h --model M.gguf (--serve | --tokens IDS [--n N]) [--ctx C] [--threads T] [--eos IDS]\n"
                 "  K2 Horizon (k2-horizon) on the CPU; docs/K2_HORIZON.md\n");
}
}  // namespace

int main(int argc, char** argv) {
    std::string model, tokens, eos_list = "1,250019";
    bool serve = false;
    long long n_gen = 16, ctx = 2048;
    int threads = (int) std::max(1u, std::thread::hardware_concurrency());
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage();
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--model" || a == "--native") model = next();
        else if (a == "--tokens") tokens = next();
        else if (a == "--n") n_gen = std::atoll(next().c_str());
        else if (a == "--ctx") ctx = std::atoll(next().c_str());
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--eos") eos_list = next();
        else if (a == "--serve") serve = true;
        else if (a == "--help" || a == "-h") {
            usage();
            return 0;
        } else {
            std::fprintf(stderr, "strata-k2h: unknown argument %s\n", a.c_str());
            usage();
            return 2;
        }
    }
    if (model.empty() || (!serve && tokens.empty())) {
        usage();
        return 2;
    }
    try {
        const auto t0 = Clock::now();
        const strata::GgufModel m = strata::GgufModel::open(model);
        Geometry g;
        Weights w;
        std::string err;
        if (!strata::k2h::geometry_from_gguf(m.meta(), g, err) || !strata::k2h::map_weights(m, g, w, err)) {
            std::fprintf(stderr, "strata-k2h: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata-k2h: %s\n", strata::k2h::describe(g).c_str());
        if (ctx <= 0) ctx = 2048;
        Engine E(g, w, ctx, threads);
        std::fprintf(stderr, "strata-k2h: ready in %.1f s: context %lld, %d threads, KV cache %.2f GiB (F32), weights read in place (mmap)\n",
                     ms_since(t0) / 1000.0, ctx, threads, 2.0 * (double) g.n_layers * (double) ctx * (double) g.n_embd_kv() * 4 / (1u << 30));
        bool ok = false;
        const std::vector<int32_t> eos = parse_ids(eos_list.c_str(), ok);
        if (serve) {
            Serve sv(E, eos);
            return sv.run();
        }
        const std::vector<int32_t> ids = parse_ids(tokens.c_str(), ok);
        if (!ok || ids.empty()) {
            std::fprintf(stderr, "strata-k2h: bad --tokens\n");
            return 2;
        }
        std::vector<float> lg;
        const auto p0 = Clock::now();
        E.forward(ids, 0, lg);
        const double pms = ms_since(p0);
        const std::vector<int32_t> top = R::argsort_desc(lg.data(), (int64_t) lg.size());
        std::printf("prompt: %zu tokens in %.0f ms; top 5:", ids.size(), pms);
        for (int i = 0; i < 5; ++i) std::printf(" %d (%.4f)", top[(size_t) i], lg[(size_t) top[(size_t) i]]);
        std::printf("\n");
        std::vector<int32_t> out;
        const auto d0 = Clock::now();
        for (long long i = 0; i < n_gen; ++i) {
            const int32_t nx = argmax(lg);
            out.push_back(nx);
            if (i + 1 < n_gen) E.forward({nx}, (int64_t) (ids.size() + out.size() - 1), lg);
        }
        const double dms = ms_since(d0);
        std::printf("generated %lld tokens in %.0f ms (%.2f tok/s):", n_gen, dms, dms > 0 ? 1000.0 * (double) n_gen / dms : 0.0);
        for (size_t i = 0; i < out.size(); ++i) std::printf("%s%d", i ? "," : " ", out[i]);
        std::printf("\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-k2h: %s\n", e.what());
        return 1;
    }
}
