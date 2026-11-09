// include/strata/k2h/layout.hpp - K2 Horizon's tensors: the layout table, its check against a model file, and the
// name -> weight mapping the engine reads them through.
//
// The table is llama.cpp b11454's llama_model_k2_horizon::load_arch_tensors, entry for entry (names from
// gguf-py/gguf/constants.py, shapes in GGUF order, dim 0 contiguous):
//
//   global     token_embd.weight [n_embd, n_vocab]; output_norm.weight [n_embd];
//              output.weight [n_embd, n_vocab], optional: absent = tied to token_embd (TENSOR_DUPLICATED)
//   every layer
//              attn_norm.weight [n_embd]                  ffn_norm.weight [n_embd]
//              attn_q.weight [n_embd, head_dim * n_head]  attn_k.weight [n_embd, n_embd_kv]
//              attn_output.weight [head_dim * n_head, n_embd]
//              attn_q_norm.weight / attn_k_norm.weight, optional: one weight per head, [head_dim, heads] or the same
//                  numbers flat (TENSOR_ALLOW_RESHAPE); K2-Horizon-MoVA-36B-A4B has none (query_key_norm false)
//              attn_gate.weight [n_embd, head_dim * n_head], optional: the softplus output gate
//   MoVA layer (MoE and value experts)
//              attn_v_gate.weight [n_embd, n_value_expert]  attn_v_gate.bias [n_value_expert], optional
//              attn_v_exps.weight [n_embd, n_embd_kv, n_value_expert]
//   other layers
//              attn_v.weight [n_embd, n_embd_kv]
//   MoE layer  ffn_gate_inp.weight [n_embd, n_expert]       exp_probs_b.bias [n_expert], optional (selection only)
//              ffn_gate_exps / ffn_up_exps.weight [n_embd, n_ff_exp, n_expert]
//              ffn_down_exps.weight [n_ff_exp, n_embd, n_expert]
//              with shared experts: ffn_gate_shexp / ffn_up_shexp.weight [n_embd, n_ff_shexp],
//                                   ffn_down_shexp.weight [n_ff_shexp, n_embd]
//   dense layer
//              ffn_gate / ffn_up.weight [n_embd, n_ff_dense]  ffn_down.weight [n_ff_dense, n_embd]
//
// The table is complete: a blk.* or global tensor it does not name is an error (llama.cpp refuses those too: "done_getting_tensors").
// Header-only and GPU-free.
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/k2h/geometry.hpp"

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace strata::k2h {

/// One weight of the model file: where it is (shard, directory entry) and its bytes (mapped, read-only).
struct TensorRef {
    const strata::GgufFile* file = nullptr;
    const strata::TensorInfo* info = nullptr;
    explicit operator bool() const { return info != nullptr; }
    const uint8_t* data() const { return file->tensor_data(*info); }
    uint64_t bytes() const { return strata::tensor_payload_bytes(*info); }
    uint32_t type() const { return info->type; }
    const std::string& name() const { return info->name; }
};

struct LayerWeights {
    TensorRef attn_norm, attn_q, attn_k, attn_v, attn_output, attn_gate, attn_q_norm, attn_k_norm;
    TensorRef attn_v_gate, attn_v_gate_b, attn_v_exps;   // MoVA
    TensorRef ffn_norm;
    TensorRef ffn_gate, ffn_up, ffn_down;                 // dense
    TensorRef ffn_gate_inp, ffn_exp_probs_b, ffn_gate_exps, ffn_up_exps, ffn_down_exps;   // MoE
    TensorRef ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp;
};

struct Weights {
    TensorRef token_embd, output_norm, output;   ///< output is token_embd's ref when the file ties them
    bool output_tied = false;
    std::vector<LayerWeights> layers;
};

/// One entry of the table.  `shapes` holds the shapes the file may store it in (more than one only for the per-head
/// norms, which llama.cpp reshapes); `slot` is where it lands in the mapping.
struct Want {
    std::string name;
    std::vector<std::vector<int64_t>> shapes;
    bool required = true;
    TensorRef* slot = nullptr;
};

/// The table for `g`, its entries pointing into `w` (resized to the layer count).
inline std::vector<Want> tensor_table(const Geometry& g, Weights& w) {
    w = Weights{};
    w.layers.resize((size_t) g.n_layers);
    const int64_t N = g.n_embd, V = g.n_vocab, HQ = g.head_dim * g.n_head, KV = g.n_embd_kv();
    std::vector<Want> t;
    auto add = [&](std::string name, std::vector<int64_t> shape, TensorRef* slot, bool required = true) {
        t.push_back({std::move(name), {std::move(shape)}, required, slot});
    };
    add("token_embd.weight", {N, V}, &w.token_embd);
    add("output_norm.weight", {N}, &w.output_norm);
    add("output.weight", {N, V}, &w.output, false);
    for (int64_t l = 0; l < g.n_layers; ++l) {
        LayerWeights& L = w.layers[(size_t) l];
        const std::string b = "blk." + std::to_string(l) + ".";
        add(b + "attn_norm.weight", {N}, &L.attn_norm);
        add(b + "attn_q.weight", {N, HQ}, &L.attn_q);
        add(b + "attn_k.weight", {N, KV}, &L.attn_k);
        t.push_back({b + "attn_q_norm.weight", {{g.head_dim, g.n_head}, {HQ}}, false, &L.attn_q_norm});
        t.push_back({b + "attn_k_norm.weight", {{g.head_dim, g.n_head_kv}, {KV}}, false, &L.attn_k_norm});
        if (g.is_mova_layer(l)) {
            add(b + "attn_v_gate.weight", {N, g.n_value_expert}, &L.attn_v_gate);
            add(b + "attn_v_gate.bias", {g.n_value_expert}, &L.attn_v_gate_b, false);
            add(b + "attn_v_exps.weight", {N, KV, g.n_value_expert}, &L.attn_v_exps);
        } else {
            add(b + "attn_v.weight", {N, KV}, &L.attn_v);
        }
        add(b + "attn_output.weight", {HQ, N}, &L.attn_output);
        add(b + "attn_gate.weight", {N, HQ}, &L.attn_gate, false);
        add(b + "ffn_norm.weight", {N}, &L.ffn_norm);
        if (g.is_moe_layer(l)) {
            add(b + "ffn_gate_inp.weight", {N, g.n_expert}, &L.ffn_gate_inp);
            add(b + "exp_probs_b.bias", {g.n_expert}, &L.ffn_exp_probs_b, false);
            add(b + "ffn_up_exps.weight", {N, g.n_ff_exp, g.n_expert}, &L.ffn_up_exps);
            add(b + "ffn_gate_exps.weight", {N, g.n_ff_exp, g.n_expert}, &L.ffn_gate_exps);
            add(b + "ffn_down_exps.weight", {g.n_ff_exp, N, g.n_expert}, &L.ffn_down_exps);
            if (g.n_expert_shared > 0) {
                add(b + "ffn_up_shexp.weight", {N, g.n_ff_shexp}, &L.ffn_up_shexp);
                add(b + "ffn_gate_shexp.weight", {N, g.n_ff_shexp}, &L.ffn_gate_shexp);
                add(b + "ffn_down_shexp.weight", {g.n_ff_shexp, N}, &L.ffn_down_shexp);
            }
        } else {
            add(b + "ffn_up.weight", {N, g.n_ff_dense}, &L.ffn_up);
            add(b + "ffn_gate.weight", {N, g.n_ff_dense}, &L.ffn_gate);
            add(b + "ffn_down.weight", {g.n_ff_dense, N}, &L.ffn_down);
        }
    }
    return t;
}

inline std::string shape_str(const std::vector<uint64_t>& s) {
    std::string r;
    for (uint64_t d : s) r += (r.empty() ? "" : ", ") + std::to_string(d);
    return "[" + r + "]";
}
inline std::string shape_str(const std::vector<int64_t>& s) {
    std::string r;
    for (int64_t d : s) r += (r.empty() ? "" : ", ") + std::to_string(d);
    return "[" + r + "]";
}

/// What map_weights found, for the log.
struct MapReport {
    size_t found = 0;           ///< table entries present in the file
    size_t optional_absent = 0; ///< optional entries the file leaves out
    uint64_t bytes = 0;         ///< their payload
    std::map<std::string, size_t> types;   ///< tensors per ggml type
};

/// Maps every tensor of `m` to `w` through the table of `g`.  False with `err` (the first problem) when a required
/// tensor is missing, a shape differs, a type has no known block size or its bytes are not inside its shard, or the file
/// holds a tensor the table does not name.
inline bool map_weights(const strata::GgufModel& m, const Geometry& g, Weights& w, std::string& err, MapReport* report = nullptr) {
    MapReport r;
    std::set<std::string> named;
    for (Want& want : tensor_table(g, w)) {
        named.insert(want.name);
        size_t s = 0;
        const strata::TensorInfo* t = m.find(want.name, &s);
        if (!t) {
            if (want.required) {
                err = "missing " + want.name;
                return false;
            }
            ++r.optional_absent;
            continue;
        }
        bool fits = false;
        for (const auto& shape : want.shapes) {
            bool same = t->shape.size() == shape.size();
            for (size_t i = 0; same && i < shape.size(); ++i) same = t->shape[i] == (uint64_t) shape[i];
            fits = fits || same;
        }
        if (!fits) {
            err = want.name + " is " + shape_str(t->shape) + ", the K2 Horizon layout wants " + shape_str(want.shapes.front());
            return false;
        }
        if (!m.in_bounds(*t, s)) {
            err = want.name + " (" + t->type_name() + ") has no known size inside " + m.shard(s).path();
            return false;
        }
        *want.slot = TensorRef{&m.shard(s), t};
        ++r.found;
        r.bytes += strata::tensor_payload_bytes(*t);
        ++r.types[t->type_name()];
    }
    for (size_t s = 0; s < m.size(); ++s)
        for (const strata::TensorInfo& t : m.shard(s).tensors())
            if (!named.count(t.name)) {
                err = t.name + " is not in the K2 Horizon layout";
                return false;
            }
    if (!w.output) {   // llama.cpp: no output.weight -> the token embedding, duplicated
        w.output = w.token_embd;
        w.output_tied = true;
    }
    if (report) *report = r;
    return true;
}

}  // namespace strata::k2h
