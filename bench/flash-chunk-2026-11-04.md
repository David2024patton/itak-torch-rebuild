# Flash attention for chunk plans (2026-11-04)

Lane: lane/flash-attn, based on merged main 864dffd.
Implements candidate C1 of the prefill chain profile
(~/workspace/lanes/profile/PROFILE-prefill-chain-2026-11-03.md):
the resident session's chunk-plan attention now runs the
fused flash form instead of the staged two-pass
scores + reduce.

Nothing in this note was measured on a GPU. This VM enumerates
no wgpu adapter, so every GPU test below compiles and skips
here by the crate's established adapter-absence policy. All
performance figures are EXPECTATIONS derived in the profile
note, labeled as such; the server commands at the end are
what turns them into measurements.

## Design

The surprise of the lane: the flash kernel needed no change.
K6's FLASH_SHADER was already shape-generic, and the profile
note's sketch (an env toggle at plan build) was only ever an
A/B vehicle. The kernel takes one workgroup per (query row,
Q head) at any row count, computes the causal position per
row (`start_pos + row`), applies the sliding mask in the
form `kp + window <= p` (no underflow), slots keys modulo
the ring capacity exactly as the scores kernel does, and
carries online-softmax state across 256-key tiles in
ascending order, which is the two-pass operation sequence
per output element. The sliding window at chunk shapes and
the chunked ring (window + cap - 1) therefore carry over
unchanged; the tiling cannot fail to express the mask,
because it never expressed anything else.

What changed is the session wiring:

- `ChunkPlan::build` prepares K6 flash for every layer at
  every row count. The `rows == 1` selection is gone;
  `LayerPlan` holds one `flash: K6FlashCall` (no longer
  optional, no scores/reduce calls beside it).
- The scores scratch is deleted from plan scratch. No
  session buffer grows with the stored position count
  anymore, so plan currency is KV-buffer currency alone:
  the scores-capacity growth rebuilds in `forward_chunk`
  are gone with the buffer they served.
- The scores budget is deleted (`SCORES_BUDGET_ELEMS` and
  the shrink inside `chunk_rows`). Under the two-pass form,
  chunk rows decayed as context grew (32 rows at 16,384
  stored, 4 rows at the full 131,072 context), multiplying
  per-chunk submits, syncs, and host records. Chunks now
  hold their cap at any context length.
- Chunk sizing is config-driven, per house law. The cap
  resolves once at session construction from
  `WGPU_COMPUTE_CHUNK_ROWS` (default 64, the value every
  recorded bench ran at), validated against the structural
  ceiling `MAX_CHUNK_ROWS` (64: the largest chunk the ring
  algebra is built around); unparsable or out-of-range
  values fall back to the default, and the resolution is
  logged through the diag channel at construction. The
  windowed ring capacity (window + cap - 1) and the
  `rewind_to` bound derive from the resolved cap, so a
  lowered cap shrinks the ring and tightens the rewind
  bound together, keeping the aliasing algebra sound.
  The pure sizing law (`resolve_chunk_rows`,
  `chunk_rows_for`, `ring_capacity`) is unit tested on the
  host: 4 tests, run for real on this VM.
- Dispatch accounting: the per-layer base drops from 17
  to 16 unconditionally (the flash form is one K6 dispatch
  where the two-pass form was two, on chunk plans; decode
  plans already counted it). Decode behavior is otherwise
  untouched: a one-row plan prepares the same flash call
  with the same geometry as before, so the decode path is
  bit-identical by construction.
- The two-pass form is NOT deleted: the trait path
  (`compute.rs`, one row per step) still runs the K6
  one-shot scores + reduce, and the K6 module docs now say
  which form serves which driver.

Refusals stay named: kernel refusals still surface under
the `attention` step through the existing `we(...)` wrap,
the head-dim bound (128) still refuses in
`ResidentSession::new` and in `K6Pipeline::check_geom`, and
the rewind refusal now names the session's resolved cap.

## Expected deltas (from the profile note, DERIVED there)

Per 64-row chunk on the real model (731 dispatches derived
by the profile from the plan builder's own accounting):

- Dispatches: 52 fewer per chunk, one per layer
  (731 to 679 derived). On the resident fixture the session
  stats pin moves from 57 to 51 dispatches per chunk,
  asserted in `resident_session_matches_trait_and_cpu`.
- Scores traffic: the scores buffer round trip is gone:
  8.39 MB written and 8.39 MB read per layer at 1,024
  stored, about 873 MB per chunk, plus the scores scratch
  allocation itself (up to 64 MiB under the old budget).
- Softmax work: the reduce's per-lane exp stream (about
  268M exp evaluations per layer at 1,024 stored, one per
  (row, head, key, lane)) collapses to one exp per
  (row, head, key), about 2.1M per layer.
- Chunk shape at depth: rows stay 64 at the full 131,072
  context instead of decaying to 4, so per-chunk fixed
  costs (submit, blocking sync, head, host record) are
  paid up to 16x less often per token at deep context.

What this does NOT promise: the profile showed the chain
is bound by instruction mix and serialization, not DRAM
bytes, and the per-stage split of the roughly 3.03 ms/token
chain was never measured. Flash at chunk shapes is proven
only at decode geometry (within noise of two-pass at 69
stored); the win is an S-scaling claim. The recorded
baselines to beat are the MMQ build's: marginal 161 tok/s,
gross 135.8 tok/s at 1,024 tokens. The profile's kill
criteria stand: a session marginal gain under 10% at
S=1024, or any exactness gate moving, kills the switch.

## Parity bar and tests

- `k6_flash_matches_two_pass` (tests/fused_kernels.rs)
  extended with four chunk geometries at the file's
  established bar (max abs diff under 1e-6 against the
  two-pass form over the same resident KV): a 64-row global
  chunk over 1,024 stored (four tiles, per-row causality);
  a 64-row sliding chunk at window 512, ring 575, slots
  wrapping, both masks biting; the full-context sliding
  geometry (window 2048, ring 2111, nine tiles, deep
  absolute positions); a 16-row partial chunk crossing one
  tile boundary.
- `resident_session_matches_trait_and_cpu` now exercises
  flash inside chunk plans against BOTH references at the
  cross-backend bar (argmax agreement at every step,
  max-abs logit diff under 0.5): the CPU model and the
  trait path (which still runs two-pass K6 per step), over
  a 4-row chunk, five decode steps, and a 70-token two-chunk
  prompt through a window-4 ring.
- Host unit tests for the sizing law (above).

## Server proof (ubuntu-server, GPU free, one heavy job)

Stage this branch's tree, verify the staged hash equals the
branch tip, then:

```sh
export PATH="$HOME/.cargo/bin:$PATH"
# 1. Kernel-level parity must EXECUTE (not skip):
cargo test -p wgpu-compute --test fused_kernels k6_flash_matches_two_pass -- --nocapture
# 2. Session parity against CPU + trait, dispatch pin 51/chunk:
cargo test -p wgpu-compute --test resident_session -- --nocapture
# 3. The spec path's resident block pin and shader compile gate:
cargo test -p wgpu-compute --test resident_block -- --nocapture
cargo test -p wgpu-compute --test shader_compile
cargo test -p wgpu-compute -p itak-spec
```

Then the prefill bench on the real file, same discipline as
the earlier notes (medians of 3, same window, this branch
against merged main as the two-pass control):

```sh
cargo build --release -p itak-loader --example glimmer_generate
GEN=./target/release/examples/glimmer_generate
MODEL=/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
# The repeated-sentence prompts from bench/mmq-wide-bench.sh
# (about 254 and about 1,024 tokens), greedy, resident path:
$GEN "$MODEL" "<254-token prompt>" 32 --gpu
$GEN "$MODEL" "<1024-token prompt>" 32 --gpu
# A 4,096-token prompt exercises the removed chunk decay:
$GEN "$MODEL" "<4096-token prompt>" 32 --gpu
```

Read `phase=prefill ms=` from each run and compute gross
and marginal rates between the two standard lengths.
Acceptance, all in the same window:

1. Gross and marginal at 1,024 tokens improve over the
   two-pass control build; the recorded MMQ-era baseline
   (marginal 161 tok/s, gross 135.8) is the floor the
   control should reproduce, and flash must beat the
   control, not just the record.
2. Exactness gates from the profile's M5: first-token
   probability 0.403462, the 8/8 Stage 1 ids, and generated
   continuations token-identical to the two-pass control at
   254 and 1,024 tokens.
3. SessionStats dispatches per 64-row chunk drop by 52
   against the control (the derived 731 to 679), and the
   per-chunk lines show 64 rows held at the 4,096-token
   prompt's depth.

## VM gate record (this lane, before push)

- cargo build --release -p wgpu-compute -p itak-spec: ok
- cargo test -p wgpu-compute -p itak-spec: 163 passed,
  0 failed (GPU parity tests skip: no adapter on this VM)
- cargo fmt --all -- --check: clean
- cargo clippy -p wgpu-compute --all-targets: no warnings
- cargo deny check: advisories ok, bans ok, licenses ok,
  sources ok
