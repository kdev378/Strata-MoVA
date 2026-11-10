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

## The CPU reference forward, against llama.cpp

`include/strata/k2h/cpu_ref.hpp` is a plain F32 forward over the dequantized weights, in llama.cpp's order, for checking
numbers (no speed). So far it runs the dense layers 0-2 and their attention; the MoE and MoVA layers come next.

The reference side is `tools/k2h/llama_dump.cpp`, a small program linked against llama.cpp b11454 (build line in the
file). It writes llama.cpp's own intermediate tensors (`attn_norm`, `Qcur`, `Kcur`, `Vcur`, `kqv_out`, `ffn_inp`,
`ffn_norm`, `ffn_out`, `l_out` per layer) for a prompt, with an F32 KV cache, flash attention off and the CPU weight
repack off (the weights stay in the file's mmap). `strata-k2h-ref MODEL --ref DIR` runs the same token ids and compares
every tensor.

llama.cpp's CPU matmul rounds the activations to Q8_K before multiplying by a Q4_K or Q6_K weight. The reference does
the same rounding (`ActRound::LlamaCpu`, the default). Without it the two drift apart by several percent within two
layers; with it they agree to float rounding.

Measured 2026-11-09 on the reference file, prompt "The capital of France is" (tokens 0, 864, 7169, 331, 8465, 395),
llama.cpp b11454 CPU build on a 4-core AVX-512 cloud VM. "rms rel" is the RMS of the difference over the RMS of
llama.cpp's values:

| Run | Layer 0 `l_out` | Layer 1 `l_out` | Layer 2 `l_out` | Worst tensor (max diff / max value) | Non-finite |
|---|---|---|---|---|---|
| layers 0-2 in one run | 4.0e-6 | 7.2e-5 | 1.5e-4 | 4.6e-3 (`Vcur-2`) | 0 |
| each layer from llama.cpp's input (`--isolate`) | 4.0e-6 | 6.4e-6 | 4.5e-5 | 3.1e-5 (`ffn_out-2`) | 0 |
| 2 tokens, then one at a time against the KV cache (`--split 2`) | same as the first row, to the digit | | | | 0 |

- Layer by layer the arithmetic matches llama.cpp to float rounding (`--isolate`): the grouped norm, Q/K/V, NEOX RoPE,
  GQA attention, the base-2 softplus gate, `attn_output` and the dense FFN.
- Across layers the difference grows: `l_out` from 4.0e-6 after layer 0 to 1.5e-4 after layer 2. The likely cause, not
  proven: a value that sits on a
  Q8_K rounding boundary rounds to the next integer on one side and not the other, and the first small differences come
  from SiLU, where ggml's CPU uses its own vectorised exp (`ggml_v_expf`, AVX-512 here) and the reference uses `expf`.
  This is backend-specific: llama.cpp on CUDA computes these differently again.
- Decoding one token at a time against the KV cache gives the same numbers as the batch, so positions above 0 work.

### MoVA (layer 3, the first MoVA layer)

The value router and the value experts as llama.cpp's `build_routed_value`: `attn_v_gate` logits (a Q4_K matmul, so
the activations are rounded to Q8_K first), sigmoid, `attn_v_gate.bias` added for the choice only, the top 4 in ggml's
own sort order, the 4 unbiased probabilities renormalised and times 2.5, each chosen expert's output (2560 -> 1024)
through SiLU and times its weight, the 4 added in the router's order. The KV cache then holds this V like any other.

ggml's CPU top-k is an unstable `std::sort` of the indices 0 .. n-1 with `v[a] > v[b]`; the reference does exactly the
same (`argsort_desc`), so on equal values it picks what llama.cpp's CPU build picks. llama.cpp's CUDA argsort is a
different algorithm, so on equal values the two llama.cpp builds themselves can pick differently.

Measured as above (same prompt, same machine), against a 4-layer llama.cpp dump:

| Run | `v_moe_logits-3` | Chosen value experts | `Vcur_routed-3` | `ffn_inp-3` | Non-finite |
|---|---|---|---|---|---|
| layer 3 from llama.cpp's input (`--isolate`) | 1.1e-7 | same 4, same order, 6 of 6 tokens (the whole 64-way order identical) | 1.8e-7 | 8.2e-8 | 0 |
| layers 0-3 in one run | 4.3e-3 | same 4, same order, 6 of 6 tokens (ranks 5 and below differ) | 6.0e-3 | 1.6e-4 | 0 |

(rms rel as above.) The MoE FFN of layer 3 comes next; the reference stops after `ffn_inp-3` until then.

### The MoE FFN, and the whole model

The FFN router and the experts as llama.cpp's `build_moe_ffn`: `ffn_gate_inp` logits (F32, so no activation
rounding), sigmoid, `exp_probs_b` for the choice only, the top 8, renormalised and times 2.5; each chosen expert's
`down(silu(gate(x)) * up(x))` times its weight, the 8 added in the router's order; then the shared expert's own SwiGLU
FFN added (no gate). The head is the grouped norm with `output_norm`, then `output.weight`.

Measured as above (same prompt, same machine), against a whole-model llama.cpp dump (1,559 tensors; llama.cpp keeps
only the last token's row from layer 47's `ffn_inp` on, so those are compared on that row):

| Run | Layer 3 `l_out` | Layer 47 `l_out` | Logits (`result_output`) | Next token | Router choices (90: 45 MoVA + 45 FFN) |
|---|---|---|---|---|---|
| each layer from llama.cpp's input (`--isolate`) | 7.1e-8 | 1.2e-7 | 2.2e-7, top 5 in the same order | same (11511) | all 90 the same set in the same order, on every token |
| all 48 layers in one run | 1.9e-4 | 5.5e-2 | 5.8e-2, the same top 5 (ranks 3 and 4 swapped) | same (11511) | 38 of 90 the same set on every token; 52 differ on 1 or 2 of the 6 tokens |

(rms rel as above; 0 non-finite values in every tensor.) Each layer computes what llama.cpp computes. Run as one, the
small rounding differences of the first layers (1e-4) are enough to change some router choices from layer 4 on: the
routers have near-ties (the coarse selection biases above), so a different expert wins by a hair. Every later layer
then works on a slightly different input. llama.cpp's own CPU and CUDA builds differ in their rounding in the same way,
so the same effect is to be expected between them.

## Reusing Strata's expert code (step 6 of the port)

Read from the code, not yet built or run for K2 (no GPU here); unconfirmed static analysis.

| Part | For K2 | Why |
|---|---|---|
| CPU native experts (`kernels/cpu/native_expert`: `native_fmt`, `native_gu_rows`, `native_down_rows`) | reusable | GGUF-form blob `[gate | up | down]`, any type ggml-cpu has a dot for (Q4_K gate/up, Q4_K or Q6_K down), activations in the type's `vec_dot_type` (Q8_K, as llama.cpp's CPU), `silu(gate) * up` then down: K2's SwiGLU. n_ff 768 is three Q8_K blocks; the activation buffers (`kNativeActBytes` for 2560, `kNativeHBytes` 1024 bytes >= 876 for 768) are large enough. |
| CPU expert pool (`kernels/cpu/pool`: `run_split_multi_native`) | reusable | The same entry point Qwen3.6's hybrid mode uses for the experts not in VRAM. |
| GPU grouped experts (`native_expert_grouped`) | gate/up yes, down partly | Q4_K gate/up is in `STRATA_GU_FMTS`. Q6_K down is in `STRATA_D_FMTS` (with `STRATA_Q6K_EXPERTS`, on by default), Q4_K down is not: 24 of K2's 45 layers have Q4_K `ffn_down_exps`. Adding type 12 to the down list is likely a one-line change, but it needs a CUDA and HIP build and a parity test. |
| The weighted combine (`native_moe_combine`) | reusable | k up to 15, the shared expert added once unweighted: K2's 8 experts plus an ungated shared expert. |
| GPU routers (`native_router_topk`, Qwen3.6's `k_route_group`) | not reusable | softmax only, without a selection bias, and only (512, 10) and (256, 8). K2 needs sigmoid + bias for (100, 8) and (64, 4). |
| Flash-Next's VRAM expert tier (`core/expert_cache`) | not reusable | Part of the Flash-Next session: a frequency profile file for 48 x 512 experts and the verify-window kernels. |
| Qwen3.6's placement (`place_experts`, hybrid mode in `q36_engine.cu`) | the design, not the code | It lives inside Qwen3.6's engine (its Model and Engine structs, MTP, the packed Q8_0 copies). The idea carries over: every expert in host memory, as many per-layer VRAM slots as fit, the rest on the CPU pool. |

The smallest K2 path from here: the routed experts repacked at load into native blobs in host memory (14.9 GiB) and
computed by the CPU pool, with zero VRAM slots to begin with (the minimum expert cache); the routers on the CPU with
`argsort_desc` (100 and 64 values per token), which also keeps llama.cpp's CPU choice on ties.

### Greedy generation, 32 tokens

`strata-k2h-ref MODEL --ref DIR --generate 32` (the prompt as one batch, then one token at a time against the KV
cache) against `llama-k2h-dump MODEL DIR --prompt ... --generate 32`, both taking the highest logit (the lowest id on
equal logits). Same prompt, same machine, 2026-11-10:

| | Tokens the same as the CPU reference's |
|---|---|
| llama.cpp b11454 CPU, F32 KV cache, flash attention off | 32 of 32 |
| llama.cpp b11454 CPU, its default context (F16 KV cache, flash attention auto) | 32 of 32 |

The text: " Paris. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Germany is Berlin. The
capital of the United Kingdom is London." One prompt only, and an easy one to predict: the router choices that differ
between the two (above) did not change a greedy token here. More prompts come with the generation step. The CPU
reference took 2 min 55 s for the prompt and 32 tokens (4 cores, the file larger than the RAM, no SIMD): it is a
checking tool, not a way to serve the model.

## strata-k2h, the CPU engine

`strata-k2h` (`src/k2h/k2h_engine.cpp`) runs K2 Horizon on the CPU. It is the minimum viable engine: it runs the model,
it is not fast, and it does not use the GPU yet.

- Every matrix product goes through ggml-cpu's own kernels: the activations rounded to the weight type's
  `vec_dot_type` with its `from_float`, then its `vec_dot` per weight row. These are the functions llama.cpp's CPU
  backend calls.
- The weights are read in place from the GGUF's mmap. A routed expert's rows are a slice of its layer's tensor: no
  repack, no copy. The tokens of a batch that chose the same expert go through it together.
- The routers run on the CPU in ggml's CPU top-k order (the CPU reference's `route`). MoVA runs on the CPU.
- F32 KV cache for `--ctx` positions (2048: 0.75 GiB). A request whose ids start with the previous request's reuses
  that prefix.
- `--tokens IDS --n N`: greedy check mode. `--serve`: serve/server.py's engine protocol (`READY <ctx> stop`,
  `GEN <max_new> [key=value ...] <ids>` -> `PP` / `T <id>` / `DONE`, `STOP`). Ends on `--eos` (default 1 and
  250019, `<|ifm|endoftext|>` and `<|ifm|im_end|>`). Sampling: temperature, top_k, top_p, min_p, seed; greedy
  without a temperature.

Measured 2026-11-10 on the reference file, 4-core cloud VM with 15 GB of RAM (the 22 GB file does not fit in it):

| | Result |
|---|---|
| Greedy, 32 tokens after "The capital of France is", against llama.cpp b11454 CPU | 32 of 32 the same |
| First step's top 5 | the same 5 as llama.cpp (ranks 3 and 4 swapped, as in the CPU reference) |
| Decode, cold page cache (first run) | 1.71 tokens/s |
| Decode, warm page cache (serve, later requests) | 4.7 to 5.4 tokens/s |

## Serving it (localhost API)

From a source checkout, setup installs it when it is named (it is not in setup's menu, the model lists, `strata launch` or
the app, and nothing else changes for the other models):

```sh
python setup.py --family k2h --model Q4_K_M          # add --gguf-dir DIR to use a file you already have
```

It compiles `strata-k2h` for this CPU (`build-k2h/`; no GPU and no llama.cpp needed), downloads IFM's file at the pinned
revision and checks its size and SHA-256, makes the tokenizer folder, and writes `strata-k2h-q4_k_m.json` and the start
script `run-k2h-q4_k_m.sh` (`.bat` on Windows). `--context` sets the context (2048 by default; the KV cache takes 0.75 GiB
of RAM per 2048 tokens). Checked on the VM above: setup from `--gguf-dir`, then the start script, then one request
("Say hello in French, one word." -> "Bonjour", stop).

`strata launch` and the app do not offer it: they read setup's model tables, and a CPU-only model in development should
not appear in their lists. That waits for the GPU path.

The manual way, without setup:

1. Build `strata-k2h` (CMake target, CPU build; it needs `STRATA_NATIVE_EXPERTS`, on by default).
2. Make the tokenizer folder from the GGUF: `python tools/k2h/make_pack.py K2-Horizon-MoVA-36B-A4B-Q4_K_M.gguf k2pack`.
3. A config, for example `k2.json`:

   ```json
   {"exe": "build/strata-k2h",
    "args": ["--model", "K2-Horizon-MoVA-36B-A4B-Q4_K_M.gguf", "--ctx", "2048", "--threads", "8"],
    "tokenizer": "k2pack",
    "arch": "k2-horizon",
    "model_name": "k2-horizon-mova-36b-a4b"}
   ```

4. `python -m serve.server --engine strata --config k2.json --port 8080`.

What the server does differently for this pack, each switched on by a key only K2's `tokenizer.json` has (other packs
render, tokenize and stop exactly as before):

- `add_bos`: the template gets `bos_token` (`<|ifm|begin_of_text|>`); K2's template starts with it.
- `special_ids.eot`: `<|ifm|im_end|>` ends a reply (the eos, `<|ifm|endoftext|>`, is not written at the end of a turn).
- `output_text`: K2's thinking tags (`<ifm|think>`, `</ifm|think>` and the `_fast` / `_faster` ones) decode as `<think>` /
  `</think>`, so the server's reasoning parser splits the reasoning from the answer. Prompts keep K2's own tokens.
- The chat template's `{% generation %}` tag (transformers' AssistantTracker) is understood by every template now: its
  body is rendered as it is.

Checked 2026-11-10 on the same VM (4 cores, 15 GB RAM, the reference Q4_K_M, temperature 0):

| Request | Reply | Finish | Decode |
|---|---|---|---|
| OpenAI, "What is the capital of France? Answer in one sentence.", thinking off | "The capital of France is Paris." | stop | 2.7 tok/s |
| OpenAI, "What is 17 * 23? Reply with the number only.", thinking on (the template's default) | reasoning: computes and checks 17 x 23 = 391; content: "391" | stop | 2.2 tok/s |
| OpenAI, streamed, "Write a short paragraph about why the sky is blue.", thinking off | a five-sentence paragraph on Rayleigh scattering, no repetition | stop | 2.3 tok/s |
| OpenAI, "日本の首都はどこですか？一文で答えてください。", thinking off | "日本の首都は東京です。" | stop | 2.3 tok/s |
| Anthropic `/v1/messages`, "Name three primary colors.", thinking disabled | "Red, blue, and yellow." | end_turn | 2.3 tok/s |

Decode through the server measured 2.2 to 2.7 tok/s, against 4.7 to 5.4 tok/s in the engine's own repeated check of
one prompt; the cause is not measured (this machine's RAM holds about two thirds of the file, so which experts are
already in the page cache matters).

### Greedy agreement on ten prompts

`strata-k2h --tokens IDS --n 32` against `llama-k2h-dump ... --generate 32` (llama.cpp b11454 CPU, F32 KV cache), ten
prompts: five raw (English, a story opening, Python, Chinese, a history paragraph) and five chat-template ones (thinking
off, except the last). The 32 tokens run past the end of the turn on purpose (no stop), to compare as far as possible.
Same VM, 2026-11-10.

| Prompt | Same leading tokens | Where they part |
|---|---|---|
| raw: "The capital of France is" | 32 / 32 | |
| raw: "def fibonacci(n):" | 32 / 32 | |
| raw: "The industrial revolution began ..." | 32 / 32 | |
| chat: "Explain what a hash table is in two sentences." | 32 / 32 | |
| chat, thinking on: "What is 17 * 23? ..." | 32 / 32 | |
| chat: "What is the capital of France? ..." | 11 / 32 | after the end of the turn: the answer and its `<|ifm|im_end|>` are the same |
| chat: "日本の首都はどこですか？..." | 24 / 32 | after the end of the turn, likewise |
| raw: "Once upon a time" | 17 / 32 | " She had a special gift" / " She had a deep love" |
| chat: "Write a haiku about autumn." | 10 / 32 | "Whispers of the wind in trees" / "Whispers on the cooling breeze" |
| raw: "人工智能是" | 6 / 32 | "...的？" / "...的\n\n" |

Every text, both sides, reads as fluent language (the CPU logs keep them). Where they part, the next-token logits of
the same input (the prompt and the shared tokens; `strata-k2h --dump-logits` against llama.cpp's `result_output`):

| Where | Largest \|logit difference\| | RMS difference | llama.cpp's top-1 margin |
|---|---|---|---|
| story, token 17 | 0.58 | 0.08 | 0.17 |
| 人工智能是, token 6 | 1.35 | 0.15 | 0.04 |
| haiku, token 10 | 0.96 | 0.17 | 0.10 (llama.cpp's own batch run of this input picks " on", its one-token-at-a-time run picked " of") |
| 日本の首都, token 24 (after the turn) | 0.56 | 0.11 | 0.02 |
| capital, token 11 (after the turn, after `<|ifm|endoftext|>`) | 5.54 | 1.09 | 2.12 |

So inside the answers the two differ by up to 1.35 in a logit (0.08 to 0.17 RMS), and they part only where llama.cpp's
own first and second choices are closer than that (0.02 to 0.17). The haiku case shows that this is the size of
llama.cpp's own batch-versus-decode difference. Matching llama.cpp to the token on every prompt would need the same
kernels in the same order, which no second implementation (and no other llama.cpp backend) has; the acceptance check
is therefore read as: 32 of 32 on five prompts, and where they part, a near-tie of llama.cpp's own.

## The GPU's grouped expert kernels: Q4_K down

`native_expert_grouped` (`src/kernels/cuda/iq_kernels.cu`) takes Q4_K for gate/up but had no Q4_K down: none of the
other models has one. K2 Horizon's Q4_K_M has a Q4_K `ffn_down_exps` in 24 of its 45 MoE layers (Q6_K in the other 21,
which the kernels take). Type 12 is now in `STRATA_D_FMTS` (both variants, with and without `STRATA_Q6K_EXPERTS`), with
the same `Fmt<12>` row dot the gate/up already use (ggml's `vec_dot_q4_K_q8_1`); n_ff 768 is three whole 256-blocks.
`native_grouped_parity` checks K2's pairs at its 2560 x 768 (Q4_K / Q4_K and Q4_K / Q6_K).

Checked here (no GPU): `iq_kernels.cu` compiles with nvcc 13.5 for compute_86 (the RTX 3070's), with and without
`STRATA_Q6K_EXPERTS`, and the kernels it makes differ from the file before by exactly three:
`native_down_kernel<12>` and `native_down_multi_kernel<12, false|true, false>` (436 -> 439 kernels; every other kernel's
name the same). Not checked: a full CUDA / HIP build, and `native_grouped_parity` on a GPU, which compares the kernel's
results with the CPU's; run it on the card before the GPU path uses these kernels.
