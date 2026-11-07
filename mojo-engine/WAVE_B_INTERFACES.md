# Wave B shared interfaces (coordinator contract, 2026-11-07)

Every Wave B lane builds against THIS page plus the Rust twins
named in its brief. Where a lane finds this contract wrong, it
stops and reports; it does not silently change the contract.
The coordinator reconciles at integration.

## Ground truth (Glimmer, from the Rust census, verified)

- 52 layers. Global attention at indices 3, 7, 11, ..., 51
  (index % 4 == 3; 13 layers). All other 39 layers are sliding
  window 2048 (`crates/attention/src/cache.rs`
  `glimmer_layer_pattern`, `crates/muse-glimmer/src/config.rs`).
- hidden 6656, FFN 19968, 32 Q heads, 2 KV heads, head dim 128.
  Q width 4096, KV width 256 per token per layer.
- RoPE base 500,000, interleaved (adjacent) pairing, sliding
  layers ONLY; globals are NoPE. In the Rust CPU path RoPE is
  applied to Q and K rows inside the attention step
  (`crates/cpu-compute/src/cpu.rs` rope_row + attention fn;
  `crates/attention/src/attention.rs` holds the math core).
- Score scale 1/sqrt(128) (`AttentionParams` default).
- RMS eps 1e-5 for the named norms; POST_NORM_EPS 1e-8 for the
  two post norms (`crates/muse-glimmer/src/model.rs`).
- Final logits: multiply by logit scale 0.1961161345243454
  FIRST, then softcap 20.0 * tanh(z/20) (Wave A final_logit op).
- Golden: prompt "The capital of France is" tokenizes (with
  BOS) to [200000, 954, 7963, 323, 11698, 373]; greedy
  generation must produce first 8 ids [13796, 26, 1573, 373,
  290, 2167, 26899, 1212] (" Paris. It is the most populous").
  First-token top-1 probability recorded ~0.4035 on the Rust
  CPU stack (llama.cpp 0.3962); a probability sanity band, not
  a gate. EOS from model metadata: tokenizer.ggml.eos_token_id
  = 200001 for this file; the golden run does not hit EOS in
  the first 8.

## Layer forward order (crates/muse-glimmer/src/model.rs
## forward_rows; the ONLY correct order)

embed(tokens) -> rms_norm_plain (weightless, eps 1e-5); then per
layer: residual = x; h = rms_norm(residual, attn_norm);
q = proj(h, attn_q); k = proj(h, attn_k); v = proj(h, attn_v);
q = rms_norm per-head (attn_q_norm, rows*32); k = rms_norm
per-head (attn_k_norm, rows*2); attn = attention(q, k, v, layer
kind, cache) [RoPE inside for sliding]; gate = proj(h,
attn_gate); attn = attn * sigmoid(gate); out = proj(attn,
attn_output); out = rms_norm(out, post_attention_norm, eps
1e-8); x = residual + out; residual = x; h = rms_norm(residual,
ffn_norm); ffn = swiglu_ffn(h) (gate/up projs, SwiGLU, down
proj); ffn = rms_norm(ffn, post_ffw_norm, eps 1e-8);
x = residual + ffn. Finish: final norm (output_norm weight,
eps 1e-5; confirm exact finish in model.rs finish_last) ->
lm_head (output.weight, Q5_K) -> final_logit transform.

Tensor names follow the GGUF file (blk.{i}.attn_norm.weight,
attn_q, attn_k, attn_v, attn_q_norm, attn_k_norm, attn_gate,
attn_output, post_attention_norm, ffn_norm, ffn_gate, ffn_up,
ffn_down, post_ffw_norm; token_embd.weight; output_norm.weight;
output.weight). Verify each against the loader census in code,
never from memory.

## B1 KV cache contract (src/kv/)

Mirror `crates/attention/src/cache.rs` semantics:
- LayerKind: Global | Windowed(window). Pattern fn for Glimmer.
- One layer store: flat f32 keys and values, one row per
  position (global) or ring slot (windowed), row =
  [num_kv_heads][head_dim] = 256 floats. Windowed slot =
  position % window; slot_position tracked; appended count is
  the next expected position.
- append(layer, k_row, v_row, position): refuses by name when
  position != appended for that layer (the Rust contract),
  when layer id is out of range, and on shape mismatch.
- Reads: key(layer, position, kv_head) / value(...) slices for
  positions still visible; visible range = [appended -
  stored, appended); a read of an evicted or future position
  refuses by name. stored_len(layer), next_position(layer).
- NO checkpoint/rewind in Wave B (spec decoding is a later
  wave); say so in the module header.
- Sizes from parameters (layer count, kinds, kv heads, head
  dim, context/window), never literals; the Glimmer defaults
  come from config at the call site.
- CPU-true storage (List[Float32]) with the layout above; the
  device seam is the row access functions. B2 reads through
  this contract only.

## B2 attention contract (src/attention/)

- CPU reference fn: attention_cpu(q for ONE position
  [32][128], cache view for the layer, position, kind) ->
  [32][128] output, computing: for each Q head h, kv_head =
  h // 16; scores over visible positions, scale 1/sqrt(128),
  causal by construction (only appended positions visible),
  window by the cache's visible range; softmax (Wave A
  softmax_cpu); weighted sum of values.
- RoPE: the CPU attention path applies rope to the Q row and
  to K rows exactly as the Rust CPU compute does (read
  cpu.rs: which K rows get rotated, at which positions, and
  whether rotated K is what the cache stores or rotation
  happens at read time; MIRROR IT, do not invent).
- GPU path: a kernel computing the same for one position
  against a device-resident copy of the layer's K/V rows;
  proven against the CPU twin on the card.
- Prefill (multi-position) attention may be built as the loop
  of the single-position fn over positions with the cache
  appended incrementally (llama-style causal), stated plainly
  if so; a batched prefill kernel is NOT required in Wave B.

## B3 model contract (src/model/)

- ModelWeights: built from the Wave A loader; per-layer tensor
  byte ranges + dtypes kept as loader views, dequant/GEMV via
  the Wave A kernels family CPU twins for the CPU path
  (k1_gemv_cpu for Q4_K, q6k_gemv_cpu, q5k_gemv_cpu,
  f32_gemv_cpu) so the whole forward runs on CPU for parity
  rungs; GPU residency is NOT required in Wave B (first light
  is a correctness wave; say so in the note).
- forward(tokens, cache) -> final hidden rows; logits_last(
  hidden_row) -> 202048 logits after final_logit transform.
- Per-layer capture hooks mirroring the Rust forward_capturing
  idea (layer input clone at named layers) for divergence
  isolation, behind the diag seam.

## B4 generation contract (src/generate/)

- generate(config, model, cache, prompt_ids, max_tokens) ->
  List[Int] generated ids: prefill = forward over prompt ids
  (positions 0..n-1 appended to cache layer by layer inside
  forward), then decode loop: argmax of logits_last -> next
  id; stop when id == eos (from metadata via the caller) or
  max_tokens reached; the sampled token is fed back as a
  one-token forward at the next position.
- Greedy argmax with first-max tie-break (lowest index), the
  Rust sampler's greedy rule; confirm in
  crates/itak-engine/src/sampler.rs.
- Diag per token through the seam: token id + position.
- The loop depends on B3's forward/logits signatures and B1's
  cache type; until integration it builds against a mock
  forward with the same shapes (the brief's VM proof).

## Suite wiring

New TestSuite files are added to tests/run_tests.sh's list by
the COORDINATOR at integration, not by lanes (one editor of
that file). Lanes ship their test files and state their
expected tallies. Every module: diag wired, named refusals,
`mojo format` clean, no compiler warnings introduced.
