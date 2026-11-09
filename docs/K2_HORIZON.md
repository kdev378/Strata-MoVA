# K2 Horizon (in development)

**Status: Strata recognises K2 Horizon files and maps their weights. No build runs the model yet.** Every build
refuses the file with "K2 Horizon ... is recognised, but no build of Strata runs it yet". Setup does not offer it.

This page is the working notes for the port: what the model is, where each number comes from, and what has been
checked. It is for developers. Nothing here is a speed or quality claim.

## The model

- IFM's K2-Horizon-MoVA-36B-A4B. Reference file: `IFM/K2-Horizon-MoVA-36B-A4B-GGUF`,
  `K2-Horizon-MoVA-36B-A4B-Q4_K_M.gguf` (22,368,011,616 bytes, SHA-256
  `513dd78590ac29135a7cea5a99865d57d65291b1f857a8904fb9b1878d4f4cbd`).
- GGUF architecture `k2-horizon`. Community quants are not used for this work.
- The references, in this order when they disagree: llama.cpp b11454 (`src/models/k2-horizon.cpp`,
  `conversion/k2_horizon.py`), then `modeling_k2_horizon.py` in `IFM/K2-Horizon-MoVA-36B-A4B`.

## The numbers, and where they come from

| What | Value | Source |
|---|---|---|
| Layers | 48: 0-2 dense (FFN 6144, plain `attn_v`), 3-47 MoE + MoVA | GGUF `leading_dense_block_count` = 3 (config `mlp_only_layers`) |
| Hidden, heads | 2560; 32 query heads, 8 KV heads, head width 128 | GGUF metadata |
| Norm | RMS norm over 2 groups of 1280, then one full-width weight; on `attn_norm`, `ffn_norm` and `output_norm`; eps 1e-6 | llama.cpp `k2_horizon_group_rms_norm` |
| Q/K norm | none in this file (`query_key_norm: false`); llama.cpp would apply one weight per head | config.json; llama.cpp `load_arch_tensors` |
| RoPE | NEOX style, all 128 dims, base 1e7 | llama.cpp `llama_model_rope_type`; GGUF metadata |
| Attention gate | `attn_gate` from the normed input; gate = log2(1 + 2^x) (softplus with beta = ln 2); multiplies the attention output before `attn_output` | llama.cpp graph (`gated`) |
| MoE FFN | 100 experts, top-8, FFN 768, plus 1 shared expert (FFN 768) added | GGUF metadata |
| Routers | sigmoid; `exp_probs_b.bias` / `attn_v_gate.bias` choose the experts but are not in the weights; top-k weights renormalised (sum clamped at 6.1e-5), then x 2.5 | llama.cpp `build_moe_ffn`, `build_routed_value`; GGUF `expert_weights_norm`, `expert_weights_scale` |
| MoVA | per token, 4 of 64 value experts; each is 2560 -> 1024 (8 KV heads x 128); V = sum of weight x SiLU(expert(x)); goes into the KV cache as a normal V | llama.cpp `build_routed_value` |

The FFN router and the MoVA router share the gating function, the renormalisation and the scale (one set of GGUF keys).

## The file, as stored (Q4_K_M)

798 tensors, 20.82 GiB of weights:

| Part | GiB | Types |
|---|---|---|
| Routed FFN experts | 14.90 | Q4_K; `ffn_down_exps` Q6_K in 21 of 45 layers |
| MoVA value experts (`attn_v_exps`) | 3.96 | Q4_K |
| Attention (q, k, gate, output) | 0.86 | Q4_K |
| `output.weight` / `token_embd.weight` | 0.49 / 0.34 | Q6_K / Q4_K (not tied) |
| Shared experts, dense FFN, routers, norms | 0.30 | Q4_K, Q6_K, F32 |

The MoVA router `attn_v_gate.weight` is Q4_K (the FFN router `ffn_gate_inp` is F32). Its top-4 choice depends on
quantized logits, so near-ties are the first place to look when Strata and llama.cpp disagree.

The selection biases are large and coarse: `exp_probs_b.bias` is 9.2 to 27.5 and `attn_v_gate.bias` 6.2 to 22.6, in
steps of 1/16 to 1/8, and within one layer they span only 0.125 to 1.0 (many experts share exactly the same bias). The
choice is made on sigmoid(logit) + bias in F32. Near 27 an F32 step is about 2e-6, so two experts whose
probabilities differ by less than that tie exactly. Matching llama.cpp therefore needs the same tie order as its top-k
(ggml_argsort_top_k), not just the same sums.

## Checking a file

`strata-k2h-check MODEL.gguf` reads the header, the geometry and maps every tensor through the K2 layout table
(`include/strata/k2h/layout.hpp`): present, the right shape, inside the file, and no tensor the table does not name.
`--read` then dequantizes every tensor on the CPU and checks that every value is finite.

Unit tests (no model, no GPU): `k2h_geometry_test`, `k2h_layout_test`.

Measured on the reference file above (SHA-256 checked), 2026-11-09, a 4-core cloud VM with 15 GB of RAM, no GPU:

```
layout: ok - 798 of 798 tensors mapped (96 optional entries absent), 20.821 GiB of weights; F32 232, Q4_K 519, Q6_K 47
mapping: 45 MoE layers, 45 MoVA layers, output its own, attention gate present, Q/K norm absent,
         value router bias present, expert selection bias present
read: 798 tensors, 37.445 G weights, 20.821 GiB, 0 non-finite values, 61.5 s
```

The 96 absent optional entries are `attn_q_norm` and `attn_k_norm` on all 48 layers.
