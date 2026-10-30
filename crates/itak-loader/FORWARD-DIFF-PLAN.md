# Forward diff: localizing Run B's first-token divergence

Companion to INTEGRATION-RUN-B.md. Run B proved the pipeline runs
and that the numerics diverge from llama.cpp at the first generated
token (our top-1: 290 " the", p 4.4e-5; reference: 13796 " Paris",
p 0.396), with our top-5 holding 1.6e-4 of the mass against the
reference's 0.62. This file is the plan that turns that end-to-end
fact into a named layer and a named defect, plus the tooling built
for it.

## What was built

- `muse_glimmer::trace` (crates/muse-glimmer/src/trace.rs): a
  `TraceSink` trait, a `NullTrace` default, statistics
  (`stats_of`: mean, population variance, max abs, first N values,
  computed in f64), and a `TraceLog` collector that reduces each
  capture to one line of a stable text format (version 1, specified
  in the module docs).
- `GlimmerModel::forward_traced`: the ordinary forward, same code
  path, additionally capturing the residual stream at these stage
  boundaries: `embed`, `layer.<i>.mid` (after the attention half,
  post-attention norm applied), `layer.<i>.out` (after the full
  layer), `final_norm`, `logits`. Each capture also emits a
  `trace.capture` event on the model's diagnostic channel. Unit
  tests prove the stage order, the shapes, the statistics, the
  line format byte-for-byte, and that traced and untraced forwards
  return identical logits.
- `examples/glimmer_layer_trace.rs` (this crate): the runnable
  tool. It tokenizes the golden `factual_capital` prompt exactly as
  Run B did, prints the prompt ids and whether they match Run B's
  `[200000, 954, 7963, 323, 11698, 373]`, enforces the coverage
  gate, runs one traced prefill, and prints (or with `--out`,
  files) the 107 trace lines (1 embed + 52 x (mid, out) +
  final_norm + logits), followed by the prefill top-5 in Run B's
  format as a self-check: if the traced run's top-5 does not
  reproduce Run B's position-1 table exactly, the trace is not
  measuring the same forward and nothing downstream of it counts.

Run it on the server, beside the model file, the same way Run B
ran (release build; one prefill is minutes-class on cpu-compute):

```sh
cargo run --release -p itak-loader --example glimmer_layer_trace -- \
    /mnt/models/<glimmer-30b>.gguf --out our-trace.txt
```

## What reference data the diff needs, exactly

A llama.cpp dump of the same stage boundaries, for the same GGUF,
the same prompt ids, CPU only, reduced to the same statistics:

1. The residual stream after the embedding lookup, all 6 positions
   (maps to our `embed`).
2. Per layer 0..51: the residual stream after the attention half
   (the tensor the reference graph feeds its FFN norm) and after
   the full layer (maps to `layer.<i>.mid` and `layer.<i>.out`).
   Last position is the must-have; all positions are wanted (a
   position-dependent error, e.g. RoPE geometry, is invisible in a
   last-row-only diff).
3. The final norm's output for the last position and the finished
   logits (maps to `final_norm` and `logits`).

How to obtain it: llama-server exposes none of this; the practical
route is a small runner (or a patched llama-cli) built against the
same llama.cpp tree Run B used (build-vulkan2), registering the
`cb_eval` callback in `llama_context_params` to copy out the named
graph tensors as the graph executes. The tensor names to capture
are whatever that tree's muse-glimmer graph calls its per-layer
residual, post-attention, and layer-output tensors; mapping our
stage names to its tensor names is a by-eye job against the graph
construction source for this architecture, and that same source
read settles hypothesis 1 (norm convention) and hypothesis 4/5
(placement, RoPE geometry) on paper, possibly before any dump is
needed.

## What exists today: nothing to diff against yet

Searched this repo on 2026-10-29: there is no llama.cpp activation
dump for Glimmer anywhere in it. `golden/` holds token-level
fixtures only (llama-server completions for TinyLlama-1.1B and
Qwen2.5-1.5B, consumed by the arch-llama crate), no per-layer
tensors for any model, and no `.npy`/`.bin` activation artifacts.
The raw Run B record on the server (`~/itak-run-b/`) holds logits-
level comparisons (top-5 tables), not graph intermediates.

Consequence, stated plainly: **the first divergent layer cannot be
named from data that exists today.** Our side of the diff (the
tool above) is built and tested; the reference side is one
llama.cpp runner away. Any number quoted before that dump exists
would be invented, so none is.

## How the diff reads, once both sides exist

Compare stage by stage in capture order, last-row statistics
first, whole-tensor statistics second:

- A stage is "matching" when mean and variance agree to roughly
  1e-2 relative and the first values track: f32 dequantization and
  summation-order noise live well under that, and a structural
  error (wrong norm form, wrong placement, missing scale) moves
  the statistics by far more.
- The first stage whose inputs matched but whose output does not
  is the defect's home. `embed` splits embedding handling from
  everything else; `layer.<i>.mid` vs `layer.<i>.out` splits each
  layer's attention half from its FFN half; `final_norm` and
  `logits` isolate the finishing math (already largely cleared by
  Run B's item 6).

## Ordered hypotheses

Run B's diagnosis order, with the stage that decides each:

0. **Embedding handling** (decided by `embed` alone). If the
   reference graph scales the embedding output (a Gemma-family
   habit: multiply by sqrt(hidden)) and ours does not, every layer
   downstream is wrong by a constant input scale and no layer
   diff will localize below `embed`. Cheapest test in the plan:
   one stage.
1. **Norm convention** (Run B item 1; decided by `layer.0.mid`).
   Our forward applies plain `x * w` RMSNorm everywhere; the
   file's norm weights are centered oddly for that reading
   (output_norm mean +0.017, block norms centered at +0.67..+1.34).
   If the reference uses the `(1 + w)` form, our layer 0 output is
   already systematically mis-scaled, and the error compounds 52
   times. Also settleable from the graph source alone.
2. **Post-norm placement and attention-gate placement** (Run B
   item 3; decided by the mid/out split). Ours applies the post
   norms after each residual add and reads the gate from the
   normed input. If the reference places either differently, the
   divergence appears between `.mid` and `.out` (or inside the
   attention half) at layer 0, not as a smooth scale drift.
3. **QK-norm placement/convention** (Run B item 4; attention half
   only). A pure attention temperature shift: it degrades but
   cannot alone un-concentrate the mass. If stages match through
   `layer.0.mid`'s inputs yet the attention output is off, this
   and item 4 are the suspects; separating them needs a Q/K
   capture point, a small extension of the trace sink, deliberately
   not built until the layer is named.
4. **RoPE geometry** (Run B item 5). The file carries no
   `rope.dimension_count`, so ours rotates the full head dim; the
   reference could hardcode partial rotation for this arch. Its
   signature is position-dependent: whole-tensor statistics drift
   while the first position stays clean, which is exactly why the
   trace keeps both views.
5. **Finishing math** (Run B item 6): excluded as the argmax
   cause (monotone map cannot reorder), kept in the diff only as
   the `logits` stage confirmation.
6. **Decode-path KV handling** (Run B's second symptom, the
   decode-time freeze): NOT addressed by this tool, by design.
   The freeze lives in single-token decode steps after a correct
   or incorrect prefill; a prefill trace cannot see it. The same
   sink and line format extend directly: capture the same stages
   on each decode forward and diff step-over-step self-consistency
   (Run B showed the distribution frozen to 1e-6 across steps
   feeding different tokens; a traced decode makes the frozen
   stage visible). That is the follow-up once the prefill error is
   named, or in parallel if the lanes split that way.

## Gates run for this deliverable

- `cargo test -p muse-glimmer`: 29 passed, 0 failed (6 new trace
  tests: statistics, empty/constant inputs, byte-exact line
  format, collector order and last-row extraction, degenerate
  shapes, traced-stage order and shapes).
- `cargo test -p itak-loader`: 9 passed plus 1 doc test; the
  example builds under the workspace lint gates.
- `cargo clippy -p muse-glimmer -p itak-loader --all-targets`:
  clean under the workspace deny-warnings lints.
- `cargo test --workspace`: every suite green, 0 failed.
