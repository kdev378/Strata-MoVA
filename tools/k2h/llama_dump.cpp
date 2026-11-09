// tools/k2h/llama_dump.cpp - the reference side of the K2 Horizon port: llama.cpp's own intermediate tensors, written out
// whole, so strata-k2h-ref can be compared with llama.cpp layer by layer (docs/K2_HORIZON.md).
//
// Not part of Strata's build: it links llama.cpp (the reference is b11454, the commit that added K2 Horizon).  Build it
// against a llama.cpp build tree:
//
//   g++ -O2 -std=c++17 tools/k2h/llama_dump.cpp -I$LLAMA/include -I$LLAMA/ggml/include \
//       -L$LLAMA/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA/build/bin -o llama-k2h-dump
//
//   llama-k2h-dump MODEL.gguf OUT_DIR (--prompt TEXT | --tokens 1,2,3) [--layers N] [--threads N] [--ctx N] [--logits]
//
// Writes OUT_DIR/tokens.txt (the token ids), OUT_DIR/index.txt (one line per tensor: name type ne0 ne1 ne2 ne3) and
// OUT_DIR/<name>.f32 (the tensor as F32, ne0 fastest, contiguous).  The tensors are the graph's names (src/models/
// k2-horizon.cpp, src/llama-graph.cpp) for layers 0 .. N-1: attn_norm, Qcur, Kcur, Vcur, kqv_out, ffn_inp, ffn_norm,
// ffn_out, l_out and, on MoE/MoVA layers, the routers' intermediates.  With --layers N the graph stops after l_out-(N-1)
// (llama_decode then reports the abort; the files are complete).  --logits runs the whole model and adds result_norm
// and result_output.
//
// The KV cache is F32 and flash attention is off, so that the reference holds no F16 rounding the Strata side does not.
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

namespace {

struct Dump {
    std::string dir;
    int layers = 3;
    bool logits = false;
    std::regex keep{"^(attn_norm|Qcur|Kcur|Vcur|kqv_out|ffn_inp|ffn_norm|ffn_out|l_out|ffn_moe_.*|v_moe_.*|Vcur_routed|"
                    "ffn_shexp.*|attn_gate.*)-(\\d+)$|^(result_norm|result_output)$"};
    std::ofstream index;
    bool stop = false;
    int written = 0;
};

bool wanted(Dump& d, const ggml_tensor* t) {
    std::cmatch m;
    if (!std::regex_match(t->name, m, d.keep)) return false;
    if (m[3].matched) return d.logits;
    return d.logits || std::atoi(m[2].str().c_str()) < d.layers;
}

float value_at(const uint8_t* p, ggml_type type) {
    switch (type) {
    case GGML_TYPE_F32: return *(const float*) p;
    case GGML_TYPE_F16: return ggml_fp16_to_fp32(*(const ggml_fp16_t*) p);
    case GGML_TYPE_BF16: return ggml_bf16_to_fp32(*(const ggml_bf16_t*) p);
    case GGML_TYPE_I32: return (float) *(const int32_t*) p;
    default: return NAN;
    }
}

bool on_tensor(ggml_tensor* t, bool ask, void* user) {
    Dump& d = *(Dump*) user;
    if (ask) return wanted(d, t);
    if (d.stop) return false;
    if (!wanted(d, t)) return true;
    std::vector<uint8_t> host;
    const uint8_t* data = (const uint8_t*) t->data;
    if (!ggml_backend_buffer_is_host(t->buffer)) {
        host.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, host.data(), 0, host.size());
        data = host.data();
    }
    std::vector<float> out;
    out.reserve((size_t) ggml_nelements(t));
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2)
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1)
                for (int64_t i0 = 0; i0 < t->ne[0]; ++i0)
                    out.push_back(value_at(data + i3 * t->nb[3] + i2 * t->nb[2] + i1 * t->nb[1] + i0 * t->nb[0], t->type));
    std::ofstream f(d.dir + "/" + t->name + ".f32", std::ios::binary | std::ios::trunc);
    f.write((const char*) out.data(), (std::streamsize) (out.size() * sizeof(float)));
    d.index << t->name << ' ' << ggml_type_name(t->type) << ' ' << t->ne[0] << ' ' << t->ne[1] << ' ' << t->ne[2] << ' '
            << t->ne[3] << '\n';
    d.index.flush();
    ++d.written;
    if (!d.logits && std::string(t->name) == "l_out-" + std::to_string(d.layers - 1)) d.stop = true;
    return true;
}

std::vector<llama_token> parse_ids(const char* s) {
    std::vector<llama_token> v;
    for (const char* p = s; *p;) {
        char* end = nullptr;
        v.push_back((llama_token) std::strtol(p, &end, 10));
        p = *end ? end + 1 : end;
    }
    return v;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s MODEL.gguf OUT_DIR (--prompt TEXT | --tokens IDS) [--layers N] [--threads N] [--ctx N] [--logits]\n",
                     argv[0]);
        return 2;
    }
    Dump d;
    d.dir = argv[2];
    std::string prompt;
    std::vector<llama_token> tokens;
    int threads = 4, n_ctx = 2048;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--prompt" && i + 1 < argc) prompt = argv[++i];
        else if (a == "--tokens" && i + 1 < argc) tokens = parse_ids(argv[++i]);
        else if (a == "--layers" && i + 1 < argc) d.layers = std::atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--ctx" && i + 1 < argc) n_ctx = std::atoi(argv[++i]);
        else if (a == "--logits") d.logits = true;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_extra_bufts = false;   // no CPU repack: the weights stay in the mmap (a 22 GB file on a 15 GB machine) and
                                  // the matmuls read the file's own blocks
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (tokens.empty()) {
        std::vector<llama_token> buf(prompt.size() + 16);
        const int n = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), buf.data(), (int32_t) buf.size(), true, true);
        if (n < 0) { std::fprintf(stderr, "tokenize failed\n"); return 1; }
        tokens.assign(buf.begin(), buf.begin() + n);
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) n_ctx;
    cp.n_batch = cp.n_ubatch = (uint32_t) std::max<size_t>(tokens.size(), 1);
    cp.n_threads = cp.n_threads_batch = threads;
    cp.type_k = GGML_TYPE_F32;
    cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval = on_tensor;
    cp.cb_eval_user_data = &d;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "cannot make a context\n"); return 1; }
    {
        std::ofstream tf(d.dir + "/tokens.txt");
        for (size_t i = 0; i < tokens.size(); ++i) tf << (i ? "," : "") << tokens[i];
        tf << '\n';
    }
    d.index.open(d.dir + "/index.txt", std::ios::trunc);
    const int rc = llama_decode(ctx, llama_batch_get_one(tokens.data(), (int32_t) tokens.size()));
    std::fprintf(stderr, "llama-k2h-dump: %zu tokens, %d tensors written to %s (llama_decode %d%s)\n", tokens.size(), d.written,
                 d.dir.c_str(), rc, d.stop ? ", stopped after the last wanted layer on purpose" : "");
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return d.written > 0 ? 0 : 1;
}
