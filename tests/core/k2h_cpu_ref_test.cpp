// tests/core/k2h_cpu_ref_test.cpp - the pieces of K2 Horizon's CPU reference forward (include/strata/k2h/cpu_ref.hpp) on
// hand-made inputs (no model, no GPU): the grouped RMS norm, NEOX RoPE, the base-2 softplus gate, SiLU, ggml's Q8_K and
// Q8_0 activation rounding, F32 -> F16, and the matrix product over a synthetic GGUF weight.
//
// The model-level check is strata-k2h-ref against llama.cpp's own tensors (docs/K2_HORIZON.md).
#include "gguf_fixture.hpp"

#include "strata/k2h/cpu_ref.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace R = strata::k2h::ref;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-100s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol * std::max(1.0, std::fabs(b)); }
}  // namespace

int main() {
    std::printf("k2h_cpu_ref_test\n");
    {
        // grouped RMS norm: each half of the vector scaled on its own, then the weight
        const float x[8] = {1, 1, 1, 1, 2, 2, 2, 2};
        const float w[8] = {1, 1, 1, 1, 1, 1, 1, 3};
        float y[8];
        R::group_rms_norm(x, w, y, 8, 2, 0.0f);
        check(near(y[0], 1, 1e-6) && near(y[4], 1, 1e-6) && near(y[7], 3, 1e-6), "group norm: each group to RMS 1, then the weight");
        R::group_rms_norm(x, nullptr, y, 8, 1, 0.0f);
        const double rms = std::sqrt((4 * 1.0 + 4 * 4.0) / 8);
        check(near(y[0], 1 / rms, 1e-6) && near(y[4], 2 / rms, 1e-6), "group norm with one group is the plain RMS norm");
    }
    {
        // NEOX RoPE: position 0 is the identity; at position p, pair (0, half) turns by p radians (theta_0 = p)
        std::vector<float> v = {1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<float> v0 = v;
        R::rope_neox(v.data(), 1, 8, 8, 0, 10000.0f);
        check(v == v0, "RoPE at position 0 changes nothing");
        std::vector<float> u = {1, 0, 0, 0, 0, 0, 0, 0};
        R::rope_neox(u.data(), 1, 8, 8, 3, 10000.0f);
        check(near(u[0], std::cos(3.0), 1e-6) && near(u[4], std::sin(3.0), 1e-6) && u[1] == 0 && u[5] == 0,
              "RoPE pairs dim i with dim i + n_rot/2 (NEOX), angle pos * base^(-2i/n_rot)");
        std::vector<float> h2 = {0, 1, 0, 0, 0, 0, 0, 0};
        R::rope_neox(h2.data(), 1, 8, 8, 2, 10000.0f);
        const double th = 2.0 * std::pow(10000.0, -2.0 / 8.0);
        check(near(h2[1], std::cos(th), 1e-6) && near(h2[5], std::sin(th), 1e-6), "the second pair turns by pos * base^(-2/n_rot)");
        std::vector<float> two(16, 0.f);
        two[8] = 1;   // head 1, dim 0
        R::rope_neox(two.data(), 2, 8, 8, 3, 10000.0f);
        check(near(two[8], std::cos(3.0), 1e-6) && near(two[12], std::sin(3.0), 1e-6), "every head is rotated the same way");
    }
    {
        // the attention gate log2(1 + 2^x)
        check(near(R::softplus2(0.0f), 1.0, 1e-6), "softplus2(0) = log2(2) = 1");
        check(near(R::softplus2(3.0f), std::log2(9.0), 1e-6), "softplus2(3) = log2(9)");
        check(near(R::softplus2(-10.0f), std::log2(1 + std::pow(2.0, -10.0)), 1e-5), "softplus2(-10) = log2(1 + 2^-10)");
        check(R::softplus2(40.0f) == 40.0f * 0.6931471805599453f * 1.4426950408889634f, "softplus2 of a large x is x (ggml's > 20 cut)");
        check(near(R::silu(1.0f), 1.0 / (1.0 + std::exp(-1.0)), 1e-7), "silu(1)");
    }
    {
        // Q8_K: 256 values on a grid of max/-127 steps, the largest |x| exact, round half to even
        std::vector<float> x(256);
        for (int i = 0; i < 256; ++i) x[(size_t) i] = std::sin(0.1f * (float) i) * 3.0f;
        x[17] = -5.0f;
        std::vector<float> r = x;
        R::round_q8_K(r.data(), 256);
        const float d = 1 / (-127.f / -5.0f);
        bool on_grid = true;
        double worst = 0;
        for (int i = 0; i < 256; ++i) {
            const double q = r[(size_t) i] / d;
            on_grid = on_grid && std::fabs(q - std::round(q)) < 1e-3 && std::fabs(q) <= 127.5;
            worst = std::max(worst, std::fabs((double) r[(size_t) i] - x[(size_t) i]));
        }
        check(on_grid && r[17] == -5.0f && worst <= d / 2 + 1e-6, "Q8_K rounding: on the max/127 grid, max exact, error <= half a step");
        check(R::nearest_int(2.5f) == 2 && R::nearest_int(3.5f) == 4 && R::nearest_int(-2.5f) == -2, "nearest_int rounds half to even");
        std::vector<float> z(256, 0.f);
        R::round_q8_K(z.data(), 256);
        check(z[0] == 0 && z[255] == 0, "Q8_K of an all-zero block stays zero");
        std::vector<float> q0 = {1.0f, -0.5f, 0.25f};
        q0.resize(32, 0.f);
        R::round_q8_0(q0.data(), 32);
        check(std::fabs(q0[0] - 1.0f) < 1e-3 && std::fabs(q0[1] + 0.5f) < 1e-2 && q0[31] == 0.0f,
              "Q8_0 rounding keeps values to a 1/127 step of the block max");
    }
    {
        // F32 -> F16: exact values, rounding to nearest even, the edges
        check(R::fp32_to_fp16(1.0f) == 0x3c00 && R::fp32_to_fp16(-2.0f) == 0xc000 && R::fp32_to_fp16(65504.0f) == 0x7bff,
              "fp32_to_fp16: 1, -2, 65504");
        check(R::fp32_to_fp16(1.0f + 1.0f / 2048) == 0x3c00 && R::fp32_to_fp16(1.0f + 3.0f / 2048) == 0x3c02,
              "fp32_to_fp16: ties to even (1 + 2^-11 -> 1, 1 + 3 * 2^-11 -> 1 + 2^-9)");
        check(R::fp32_to_fp16(1e6f) == 0x7c00 && R::fp32_to_fp16(5.9604645e-8f) == 0x0001 && R::fp32_to_fp16(1e-9f) == 0,
              "fp32_to_fp16: overflow to inf, the smallest subnormal, underflow to 0");
        bool round_trip = true;
        for (uint32_t h = 0; h < 0x7c00; h += 7) round_trip = round_trip && R::fp32_to_fp16(strata::fp16_to_fp32((uint16_t) h)) == h;
        check(round_trip, "fp32_to_fp16(fp16_to_fp32(h)) == h over the finite halves");
    }
    {
        // the router: the bias chooses, the unbiased probabilities weigh; renormalised, then scaled
        const float logits[4] = {2.0f, 0.0f, 1.0f, -1.0f};
        const float bias[4] = {0.0f, 5.0f, 0.0f, 0.0f};   // expert 1 has the lowest probability but is chosen first
        std::vector<std::string> seen;
        const R::Routed r = R::route(logits, bias, 1, 4, 2, strata::k2h::Gating::Sigmoid, true, 2.5,
                                     [&](const char* what, const std::vector<float>&) { seen.push_back(what); });
        const float p1 = R::sigmoid(0.0f), p0 = R::sigmoid(2.0f);
        check(r.ids.size() == 2 && r.ids[0] == 1 && r.ids[1] == 0, "the selection bias decides which experts are chosen");
        check(near(r.weights[0], 2.5 * p1 / (p1 + p0), 1e-6) && near(r.weights[1], 2.5 * p0 / (p1 + p0), 1e-6),
              "their weights are the unbiased probabilities, renormalised to 1 and times 2.5");
        check(seen == std::vector<std::string>({"probs_biased", "argsort", "topk", "probs", "weights_norm", "weights"}),
              "the router reports llama.cpp's intermediate tensors");
        const R::Routed u = R::route(logits, nullptr, 1, 4, 2, strata::k2h::Gating::Sigmoid, false, 0.0,
                                     [](const char*, const std::vector<float>&) {});
        check(u.ids[0] == 0 && u.ids[1] == 2 && u.weights[0] == p0, "no bias, no norm, no scale: the top probabilities as they are");
        const float tiny[2] = {1e-30f, 1e-30f};
        const R::Routed c = R::route(tiny, nullptr, 1, 2, 2, strata::k2h::Gating::Softmax, true, 0.0,
                                     [](const char*, const std::vector<float>&) {});
        check(near(c.weights[0], 0.5, 1e-6), "softmax gating: two equal logits share the weight");
        // equal values: the order is std::sort's on the identity with v[a] > v[b], as ggml's CPU argsort
        std::vector<float> eq(64, 1.0f);
        eq[10] = 2.0f;
        std::vector<int32_t> ref((size_t) 64);
        for (int i = 0; i < 64; ++i) ref[(size_t) i] = i;
        std::sort(ref.begin(), ref.end(), [&](int32_t a, int32_t b) { return eq[(size_t) a] > eq[(size_t) b]; });
        check(R::argsort_desc(eq.data(), 64) == ref && ref[0] == 10, "argsort_desc keeps ggml's CPU order on ties");
    }
    {
        // the matrix product over a GGUF weight: W [n_in 4, n_out 3] F32, two tokens
        const fs::path dir = fs::temp_directory_path() /
                             ("strata-k2h-cpu-ref-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(dir);
        const fs::path f = dir / "w.gguf";
        fixture::write(f, {fixture::str("general.architecture", "none")}, {{"w", {4, 3}, 0, 1}});
        {
            // replace the fixture's byte pattern with known floats
            const strata::GgufFile g0(f.string());
            const uint64_t at = g0.data_start() + g0.tensors()[0].offset;
            const float W[12] = {1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1};
            std::FILE* fp = std::fopen(f.string().c_str(), "r+b");
            std::fseek(fp, (long) at, SEEK_SET);
            std::fwrite(W, 4, 12, fp);
            std::fclose(fp);
        }
        const strata::GgufFile g(f.string());
        const strata::k2h::TensorRef W{&g, &g.tensors()[0]};
        const float x[8] = {1, 2, 3, 4, -1, 0.5f, 2, 0};
        float y[6] = {};
        R::matmul(W, x, 2, y, 1, R::ActRound::None);
        check(y[0] == 1 && y[1] == 2 && y[2] == 10 && y[3] == -1 && y[4] == 0.5f && y[5] == 1.5f,
              "matmul: y[t][j] = W row j . x[t] (GGUF [n_in, n_out], row j contiguous)");
        float y2[6] = {};
        R::matmul(W, x, 2, y2, 1, R::ActRound::LlamaCpu);
        check(std::memcmp(y, y2, sizeof y) == 0, "an F32 weight meets F32 activations (no rounding), as in ggml");
        std::error_code ignored;
        fs::remove_all(dir, ignored);
    }
    std::printf(g_fail ? "FAILED (%d)\n" : "ALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
