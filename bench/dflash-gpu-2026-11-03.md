# DFlash on GPU: the draft's resident forward (2026-11-03)

Lane `lane/dflash-gpu` (on `lane/spec-gpu` tip d10987e). This note
is the design record and the server proof script for removing the
blocker `bench/spec-gpu-port-2026-11-03.md` named: the DFlash
draft had no GPU path, a draft block cost roughly 31-35 s on CPU
against GPU verify rounds in tens of milliseconds, and T-9.6's
speed bar (88.6 tok/s effective) could not pass until the draft
ran on GPU. The draft now has a resident GPU forward behind the
same Speculator seam. Nothing in this note claims a server
verdict: the local gates are green (below), and every
adapter-bound proof below still has to run on the server.

## What was built

- `crates/dflash`: one new error variant,
  `DflashError::StepFailed { step, detail }`, the draft-vocabulary
  mirror of `GlimmerError::StepFailed`, so backend refusals reach
  the speculative loop as `SpecError::DraftRefused` and never as
  raw backend errors.
- `crates/wgpu-compute/src/kernels/k15_dflash.rs` (K15): the two
  primitives no existing kernel provides.
  - NEOX rotate: K5's rotate is the interleaved (GPT-J) pairing;
    the draft's arch rotates half-split (NEOX,
    `torch_math::rope`). Positions come from a device array,
    because inject rows carry caller-supplied positions while
    block rows carry a contiguous span. Inverse frequencies are
    host-computed in f64, stored f32, the K5 convention.
  - Draft attention: K6 is causal, ring-addressed, and
    Glimmer-shaped. The draft's attention is (on the real file)
    non-causal, reads a sparse position-addressed committed set
    plus the block's own K/V in one visible set, and masks by
    sliding window over absolute positions. One workgroup per
    (query row, query head), the GQA group folded in, online
    softmax with f32 accumulation throughout (the CPU two-pass
    softmax's math, reassociated).
- `crates/wgpu-compute/src/dflash_session.rs`:
  `DflashRawWeights` mirrors `dflash::Weights::load` over raw
  `TensorView`s (same variant gates, same shape checks, same
  `fc`-width derivation, same error names), and `DflashSession`
  mirrors `DflashModel`'s two passes in the resident pattern:
  weights uploaded once through the resident session's own
  `upload_proj`/`upload_norm`, one command chain per pass, plans
  cached per row count, logits readback appended to the chain's
  encoder. Composition: K1/K2/K3 GEMV projections at every batch
  (no activation quantization anywhere in the draft path, so
  parity is by construction rather than by tolerance luck), K4
  norms and residual closes (per-head norms fold heads into
  rows; the optional post norms use K4's weighted-residual
  form, the CPU composition's exact math), K7 SwiGLU, K8
  embedding gather, K15 for rotate and attention.
  - Resident KV is a flat position-addressed space (slot =
    absolute position), NOT a ring: the CPU cache is a
    position-keyed map whose reads filter by mask, and the flat
    layout is that map's exact device shape. The sliding window
    stays a read-side mask, as in the CPU implementation.
    Injection lands rows with encoder copies, coalesced over
    contiguous position runs, segmented at 64 rows per segment
    inside one submission per segment.
  - Validation order and diag event names match the CPU model
    (`inject.begin`, `forward_block.end`, and friends, on the
    draft's `Diag` seam). KV capacity is session config
    (`DflashSessionConfig::kv_capacity`, default 8192 positions,
    about 320 MB at the real geometry), never a compiled
    constant; an injection reaching past it refuses by name.
- `crates/itak-spec`: `DflashSpeculator` gains a backend enum.
  All loop bookkeeping (capture staging and its refusals,
  commit consumption, proposal row slicing, the greedy/sampled
  arms) is shared code above the arm; the arm only answers
  inject, forward_block, and committed_len. The GPU arm holds a
  `DflashSession<'static>` (the runner leaks the backend, the
  discipline it already uses for the target's resident
  session). Selection is config with a default:
  `SpecConfig.draft_backend: DraftBackend` (library default
  Cpu). `spec.model()` became `spec.config()`, which both arms
  answer; `SpecSession` needed no other change.
- The runner (`spec_generate_gpu`) takes `--draft=gpu|cpu`
  (its own default is gpu; unknown values refuse by name) and
  `--kv-capacity=<positions>`. Both residents share one leaked
  backend in phase B. Draft weights load per arm inside phase
  B, so a GPU run never pays the CPU dequantization.

## Local gates (this VM, no adapter)

Executed for real here:

- `cargo test --workspace`: 447 passed, 0 failed.
- The raw-load refusal battery (missing tensor, variant gate,
  wrong shape, fc width not a multiple of the target layer
  count) executes on CPU and passes; the K15 geometry guards
  and the session's bookkeeping helpers (contiguous runs,
  reported width, config default) execute and pass; the
  itak-spec suite (37 tests, including the speculator staging
  contract and the DraftBackend default/parse test) passes
  unchanged on the shared bookkeeping.
- `cargo clippy --workspace --all-targets`: clean.
  `cargo fmt --all -- --check`: clean.

Compiled and skipped here (the workspace's standard printed
skip, "no suitable adapter (enumerated: [])"), to execute on
the server:

- `wgpu-compute/tests/dflash_session.rs`: session parity
  (fuse, inject, block logits vs the CPU model over contiguous
  and sparse commits), construction refusals, K15 rotate vs
  `torch_math::rope`.
- `wgpu-compute/tests/shader_compile.rs`: the K15 pipeline
  build test.
- `itak-spec` speculator test `gpu_arm_shares_the_speculator_bookkeeping`.

## Server proof (ubuntu-server, GPU free, merged tree)

```bash
export PATH=$HOME/.cargo/bin:$PATH
cd <merged tree>

# 1. Fixture parity, executed for real on the XTX. The parity
#    fixture feeds the CPU model the dequantization of the very
#    bytes the session uploads, so a failure here is device
#    arithmetic, not fixture drift. Bar: per-row argmax
#    agreement and max-abs logit drift under 0.5, the resident
#    tests' cross-backend bar.
cargo test -p wgpu-compute --test dflash_session
cargo test -p wgpu-compute --test shader_compile
cargo test -p itak-spec

# 2. The spec-gpu lane's owed resident pin, still outstanding
#    from its note:
cargo test -p wgpu-compute --test resident_block

# 3. Real-file parity between the draft arms, at the loop
#    level: one plain stream, then each arm in spec mode
#    against it. Both must print IDENTITY: PASS. This is the
#    check that the GPU draft and the CPU draft drive the same
#    committed stream on the real files.
cargo build --release -p itak-spec --example spec_generate_gpu
T=/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
D=/mnt/models/Muse-Glimmer-30B/dflash-Muse-Glimmer-30B-Q4_K_M.gguf
./target/release/examples/spec_generate_gpu $T $D 64 6 "The capital of France is" plain /tmp/plain_stream.txt
./target/release/examples/spec_generate_gpu $T $D 64 6 "The capital of France is" spec /tmp/plain_stream.txt --draft=cpu
./target/release/examples/spec_generate_gpu $T $D 64 6 "The capital of France is" spec /tmp/plain_stream.txt --draft=gpu

# 4. The T-9.6 verdict run with the GPU draft (identity first,
#    then the speed bar; exit code follows correctness only):
./target/release/examples/spec_generate_gpu $T $D 64 6 --draft=gpu
```

## Honest expectations

- The draft side stops being the wall. Per block, the GPU
  draft streams about 2.5 GB of weights (about 1.5 GB of layer
  projections, 0.12 GB of `fc`, and about 0.92 GB of the
  borrowed Q5_K head, which is the single largest stream in
  the pass); at the K1/K2/K3 rates this workspace measured
  (about 815 GB/s), that is roughly 3 ms per block, plus about
  0.2 ms per commit for injection. Against 31-35 s, the draft
  rounds become noise next to the target's verify forwards.
- Whether T-9.6 flips to a full PASS is NOT this lane's to
  promise. The remaining terms are the verify side's cost per
  round and the real acceptance rate at depth 6 (the llama.cpp
  sweep measured 0.541 on its harness; this stack's acceptance
  on the proof prompt is whatever step 4 prints). The
  arithmetic says the bar is reachable; the run decides.
- Memory: phase B now holds the target resident (about
  17 GB) plus the draft resident (about 2.5 GB of weights,
  320 MB of KV at the default capacity, small scratch) on one
  24 GB card. It fits, with a few GB of headroom; if a future
  run shares the card, `--kv-capacity` is the knob.
- First-failure triage if step 1 fails on the server: the
  rotate test isolates NEOX; the fuse comparison isolates the
  fc + norm prefix; the block comparison then implicates the
  per-layer chain or the K15 attention (the newest math). All
  three signals are separate tests for exactly this reason.

## Named gaps and deliberate differences

- Borrowed-table resolution (missing table, wrong shape)
  happens at session construction, not at first forward as on
  the CPU pass, because the session uploads the tables once.
  Same error names, same order (embedding before head).
- Only the quantized projection families (Q4_K, Q5_K, Q6_K)
  map onto the GEMV kernels; an F32 projection or head refuses
  at session build with `StepFailed`, the same refusal the
  resident target path gives. The real draft and target files
  are entirely in-family. The embedding gather additionally
  accepts F32/F16/BF16/Q8_0 through K8.
- The K15 attention refuses `value_length` above 1024 by name
  at prepare time (per-lane accumulator stride cap). The real
  file's value length is 128.
- RoPE angles are computed in f32 on device (host f64
  inverse frequencies, the K5 convention) where the CPU
  reference uses f64 angles. Inside the cross-backend bar at
  proof-scale positions; at very long contexts the drift
  grows, and a future precision pass would start here.
- Long generations: commits grow monotonically, so a run
  whose committed length passes the KV capacity refuses at
  that commit by name. The 64-token proof never nears the
  default 8192; production-length runs set `--kv-capacity`
  (full 131072-position residency at the real geometry costs
  about 5.2 GB, a deployment choice, not a code change).
