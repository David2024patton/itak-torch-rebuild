# Prefill host gap: the 1,457 ms, diagnosed and closed (2026-11-05)

Lane: lane/prefill-host-gap, based on main 060546b. One
crate changed: `crates/wgpu-compute/src/session.rs`, plus
its resident-session test pins. No kernel changed, no
decode behavior changed, no dispatch added or removed.

**Verdict in one line:** the profile's host gap is two
serializations, not one. The bulk is the session's weight
uploads, staged on the queue at construction and never
flushed, so the first forward's first chunk waits on
~16.76 GB of device copies that no kernel timing ever
sees. The rest is the per-chunk blocking logits readback:
every chunk drains the device on a map whose contents
`forward` discards for every chunk but the last. Both are
fixed here: construction flushes its uploads, and a
multi-chunk forward reads back only its final chunk.

## The gap, restated

From bench/prefill-profile-2026-11-05.md: the 1,022-token
prefill wall is 8,116 ms (timing off), the GPU kernel
total is 6,659 ms (the timed runs), the gap is 1,457 ms
(18% of wall). The profile averages the gap as "about
91 ms per chunk" and separately reports about 1.7 s of
one-time first-slice latency inside slice 0's 2,064 ms
wall. Those two shapes only reconcile one way, and the
code says which.

## Mechanism 1 (the bulk): the uploads are paid inside the first forward

- `GpuContext::upload_bytes` (crates/wgpu-compute/src/
  device.rs) is `create_buffer` plus `queue.write_buffer`.
  A wgpu queue write is staged: its device copy executes
  in queue order ahead of the next submission.
  `ResidentSession::new` uploads every tensor this way
  (the embedding, the final norm, the head, and every
  layer through `ResidentLayer::upload`) and returns
  without ever submitting or polling.
- So when the first `forward` runs, the whole weight set
  (731 tensors, 16.76 GB on the real file) is still
  queued ahead of chunk 0's chain, and chunk 0's
  `map_staging` (device.rs: `map_async` plus
  `device.poll(wait_indefinitely)`) is the first wait
  that pays for all of it. The harness measures
  `phase=session_build` up to `new()`'s return, so its
  prefill clock starts with the debt unpaid; so does
  `glimmer_generate`'s. Both shipped walls carry it.
- Copies are not dispatches: the diag recorder brackets
  dispatches only, so the copy time is in every wall and
  in none of the 6,659 ms of kernel time. That is the
  profile's "first-forward lazy work": not lazy, just
  unattributed.
- Size check: slice 0's timed wall is 2,064 ms. Chunk
  0's kernel time is about 356 ms (the MMQ pair is flat
  per chunk at about 346 ms; flash at chunk 0 is 3.2 ms;
  the rest is single digits) and the harness's first
  drain is about 36 ms, leaving about 1,672 ms for the
  flush plus chunk 0's host preparation (Mechanism 2's
  first-chunk share: KV allocation and the cap-rows plan
  build, bounded at tens of ms by the code census
  below). 16.76 GB in about 1.6 s is about 10 GB/s
  effective, the right magnitude for 731 staged
  `write_buffer` segments through the queue's staging
  belt.

## Mechanism 2: every chunk drains the device on a readback nobody reads

- `forward_chunk` ends every chunk with `map_staging`,
  a full device drain. When it returns, the queue is
  empty; only then does the host prepare the next chunk,
  with the device idle for all of that preparation.
- `forward` overwrites each chunk's logits with the
  next chunk's; `forward_argmax` overwrites each
  chunk's winner the same way. For every chunk but the
  last, the finished logits (808 KB at the real
  vocabulary) or the eight-byte winner are computed,
  copied to staging, mapped back, and discarded. Only
  the final chunk's tail is ever returned. The shipped
  driver pays this on both of its prefill shapes:
  `glimmer_generate` drives the prompt through
  `session.forward(ids)` or `session.forward_argmax(ids)`
  depending on its tail flag (crates/itak-loader/
  examples/glimmer_generate.rs).
- The host work each drain exposes, per steady-state
  chunk (release build, counted from the code): 52
  `ensure_layer` capacity checks; `plan.set_positions`,
  which is 52 times (`K9Call::set_start_position`, one
  4-byte `queue.write_buffer`, crates/wgpu-compute/src/
  kernels/k9_attn_prep.rs, plus `K6FlashCall::
  set_geometry`, one params write, kernels/
  k6_attention.rs); the token-id `write_buffer`; about
  653 prepared-call encodes (each a `set_pipeline`, a
  `set_bind_group`, and a `dispatch_workgroups`; K13's
  `encode_into` is the exemplar); encoder finish and
  submit. At wgpu release-mode recording costs that is
  an estimated 3-10 ms per chunk, 45-150 ms over the 15
  boundaries, plus the tail chunk's `ChunkPlan::build`
  (rows 62: scratch allocation plus about 650 bind
  group creations) executed after chunk 14's drain with
  the device idle. Small next to Mechanism 1, fully
  exposed by the drains, and the reason any host jitter
  (scheduling, a driver flushing its output) lands
  whole in the prefill wall.

## Reconciliation of the published numbers

wall_off 8,116 = uploads U + host prep H + GPU work.
Slice 0 bounds U + (chunk-0 host) at about 1,672 ms, so
U + H is 1,457-1,672 ms under every consistent reading;
the spread is the accounting residual between the timed
and untimed runs (per-dispatch timestamp brackets over
10,448 dispatches, and the harness's first drain
plausibly costing more than its 36 ms average), which
the published aggregates cannot separate further. They
do not need to be separated: U is at least about 1.3 s
under every assignment, H is 45-150 ms, and the fix
removes the serialization that exposes both. The server
protocol below measures U directly, as the growth of
`phase=session_build`.

## Candidates eliminated

- Readback payload cost: converting 202,048 f32 values
  is about 1 ms per chunk. The wait is the cost, not
  the copy.
- Pipeline or bind-group reconstruction per chunk:
  none. Pipelines compile at backend construction
  (kernels/k13_q4k_mmq.rs: "Compilation happens at
  backend construction, like every kernel in this
  crate"); bind groups are built once per `ChunkPlan`
  and replayed by `encode_into`.
- Timestamp drains: the harness's ~36 ms times 16
  exists only in its timing mode; the shipped path
  never drains, and the profile's own A/B shows timing
  costs nothing on it (8,137 vs 8,169 ms). Excluded.
- Per-chunk allocation: three small Vecs per chunk
  (the KV capacity snapshot, the views, the token
  bytes). Capture staging allocates per chunk only on
  capturing forwards, which are not the prefill path.

## The fix

All in `crates/wgpu-compute/src/session.rs`:

1. **Construction flushes the uploads.**
   `ResidentSession::new`, after the uploads and before
   returning, submits an empty command buffer (which
   flushes the queue's pending writes) and polls the
   device; a flush failure refuses construction under
   the `session` step. "Session built" now means the
   weights are resident, and U is paid under
   `phase=session_build`. The copy work itself is
   unchanged; only its attribution moves.
2. **Chunk submissions pipeline.** `forward` and
   `forward_argmax` count their chunks up front (the
   cap is fixed for the session's life) and mark every
   chunk but the last deferred via the pure predicate
   `defers_readback(index, count)` (unit-tested).
   `forward_chunk` takes the flag and honors it only
   for the logits and argmax tails without captures: a
   deferred chunk encodes and submits identically
   (same chain, same head, same tail copy, same diag
   stamps, same dispatch count) and skips only its
   map, yielding `ChunkOutcome::Deferred`. Soundness
   is the single queue's total order: chunk N+1's
   queue writes (positions, token ids) and dispatches
   execute only after chunk N's work completes, so
   nothing observes a half-updated plan; a plan dropped
   by a mid-forward rebuild (the tail chunk's, or a KV
   growth's) stays alive in the queue's usage tracking
   until its submission finishes. Decode is untouched
   by construction: a single-chunk forward's only
   chunk is its last, so it never defers; refusal
   behavior is unchanged (every refusal still happens
   at prepare time, before the chunk's submission).
3. **One deliberate contract change:**
   `SessionStats.readbacks` and `argmax_readbacks` now
   count blocking maps (one per forward call) instead
   of tail copies (one per chunk). `submissions`,
   `chunks`, `dispatches`, and `tokens` are unchanged.
   The pins in tests/resident_session.rs are updated
   to say so, and a new device test
   (`resident_session_multichunk_forward_defers_
   readbacks`) pins the fingerprint: a 150-token
   forward (chunks 64/64/22) against the same tokens
   stepped chunk by chunk returns equal logits with
   equal dispatches, at readbacks 1 against 3.

## What remains serialized, and why

- Chunk 0's preparation before the first submission
  (KV allocation for 52 layers; the cap-rows
  `ChunkPlan::build`): nothing exists yet to overlap
  it with. Bounded at tens of ms in release. Moving
  the cap-rows plan build into construction is the
  named follow-up; a plan's currency is defined
  against the session's KV buffers, which allocate
  lazily, so that is a design step, not a relocation.
- The final chunk's readback wait: inherent; it
  carries the return value.
- `forward_capturing` drains per chunk as before: its
  capture staging is allocated per chunk and mapped
  inside the chunk, and the captures are the call's
  product. A diagnostics path, not the prefill path.
- `forward_block_capturing` still maps per chunk:
  every row's logits are returned data, and the block
  staging is one buffer per plan, so deferral there
  needs per-chunk (double-buffered) block staging.
  Verify blocks are single-chunk in practice (block
  rows are at most the chunk cap), so nothing is owed
  today; named here in case that ever changes.
- `DflashSession`'s construction has the same
  unflushed-upload shape through the same
  `upload_bytes` (its first block forward pays its own
  much smaller draft flush, about 1.6 GB). The same
  one-submit fix applies; left for the spec lane to
  take deliberately rather than bundled here.
- Timing mode is unchanged in kind: stamps past the
  recorder's 2,048-dispatch window drop until a drain,
  exactly as before (maps never drained the recorder;
  drains live in the harness and the smoke test).

## Expected wall, from the arithmetic

Off-mode GPU work is 6,444-6,659 ms depending on the
timed total's residual. The fix takes U plus the
exposed host prep (1,457-1,672 ms together) out of the
prefill wall and hides the steady per-chunk preparation
behind device execution. What stays in the wall: chunk
0's preparation (tens of ms) and the final chunk's
readback wait. Expected `phase=prefill` for the
1,022-token prompt: about 6.5-6.7 s, about 153-157
tok/s, against today's 8,116 ms and 125.9 tok/s: the
profile's GPU-bound rate, reached with no kernel
changes. End-to-end one-shot time is about unchanged
(U is still paid, during session build); a long-lived
engine pays U once at load and every prefill gains the
full amount.

## Server confirmation protocol

Card-state rules as usual: cleared card, the phase
lock, the staged tree hash-verified against the lane
tip, CARGO_BUILD_JOBS=4, MALLOC_ARENA_MAX=2
MALLOC_TRIM_THRESHOLD_=0, production Glimmer
untouched, no reboot.

1. Correctness first: `cargo test --release -p
   wgpu-compute --test resident_session` on the card
   (the deferral test and the cross-path comparisons
   must pass for real, not skip), then the golden
   subset leg the overnight lanes use. Prefill is
   deterministic; any logit or token movement is a
   stop, not a number.
2. Re-run bench/prefill-profile-run.sh unchanged.
   Predictions, in order: (a) `phase=session_build`
   grows by about 1.3-1.7 s against the profile run
   (the flush, now attributed, which also measures U
   exactly); (b) the harness's slice 0 wall falls from
   2,064 ms to chunk scale (a few hundred ms); (c)
   `phase=prefill` timing-off falls from 8,116 ms
   toward 6.5-6.7 s; (d) `glimmer_generate`'s
   `phase=prefill` agrees in both tail modes, and its
   prefill argmax and generated ids are unchanged
   from the pre-fix run (the identity stream).
3. Stats fingerprint: the harness drives per-slice
   forwards, so its `phase=session_stats` shows
   readbacks == forwards; a monolithic
   `glimmer_generate` run shows chunks == 16 for the
   1,022-token prompt with readbacks == forwards (one
   blocking map for the whole prefill).

## Gates on this VM

All on the committed tree, PATH with ~/.cargo/bin:

- `cargo test -p wgpu-compute`: every suite green, 0
  failed. The lib suite (90 passed) includes the new
  `defers_readback_only_before_the_final_chunk` unit
  test, which executes here. The resident suite (5
  passed) includes the new
  `resident_session_multichunk_forward_defers_readbacks`
  device test and the `resident_block` suite (2 passed)
  is green; this VM has no GPU adapter, so the device
  tests pass by their designed skip, and their on-card
  execution is step 1 of the server protocol above.
- `cargo test -p itak-spec`: 47 passed, 0 failed.
- `cargo fmt --check`: clean.
- `cargo clippy -p wgpu-compute --all-targets -- -D
  warnings`: clean.
- `cargo check --workspace --all-targets`: clean.
