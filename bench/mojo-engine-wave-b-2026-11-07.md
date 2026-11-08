# Mojo engine Wave B: KV cache, attention, model forward, generation loop

Date: 2026-11-07, closed 2026-11-08. Branch
`mojo-max-engine`. Engine version 1.3 as of the wave's
closing commit: the golden verdict is in and it PASSED (see
THE GOLDEN VERDICT below for rung 2 counted, rung 3 8/8, and
the end-to-end run). Remote main never moved off b416987.

Suite at this writing (VM, `tests/run_tests.sh` via pixi):
107 tests green (Wave A 78 + B1 9 + B2 7 + B4 9 + B3 4),
plus the 3-test seam-on diag rerun, config probes, and the
engine hello: PASS, 0 failed, 0 skipped, reproduced after
every integration.

## Lane B1: KV cache (src/kv/cache.mojo, 595 lines; tests 9/9)

Mirror of crates/attention/src/cache.rs semantics: LayerKind
(global | windowed), the Glimmer pattern (52 layers, global
at index % 4 == 3, windowed 2048 elsewhere), flat f32 rows
[num_kv_heads][head_dim], windowed ring slot = position %
window with slot_position tracking, appends refused by name
out of order / out of range / wrong shape, reads by absolute
position refused by name when evicted or unwritten. Proofs:
global round-trip exact; small-window wraparound exact at
every boundary; the REAL window 2048 shape at positions
2047/2048/2049 exact; a global sibling unaffected by a
windowed wrap; 11 named refusals asserted. Size accounting
CORRECTED the wave brief's reading: the recorded full-context
figure 3,653,238,784 B is the WINDOW-CAPPED sum (13 x 131072
+ 39 x 2048 positions), and the all-global upper bound is
13,958,643,712 B; the accounting exposes both and the tests
assert both. No checkpoint/rewind (later wave), stated in the
module header.

## Lane B2: attention (src/attention/attention.mojo, 528 lines; CPU tests 7/7, GPU proof PASS)

RoPE findings from the Rust CPU path, now the engine's law:
rotation happens at WRITE time in CpuCompute::attention
(cpu.rs rope_row); Q and K rows rotate at the position's
absolute position with the SWA base (500,000); the cache
holds ROTATED K; globals never rotate; the attention math
core takes pre-rotated rows (scale 1/sqrt(head_dim), GQA
kv_head = h // 16, causal + window masking over absolute
positions). The lane shipped both seams against flat K/V
stores: attention_cpu (raw K, rotates at read) and
attention_cpu_rotated_cache (rotated K, rotates only Q
internally), proven equivalent. CPU proofs against in-test
Float64 references: tiny GQA max_err 1.18e-07, causal
3.32e-07, window boundary 4.80e-08, global NoPE 8.87e-08,
sliding RoPE 1.02e-07, prefill loop worst 6.93e-08, 5 named
refusals. GPU proof on the card (64 positions, head_dim 128,
RoPE on): cosine 0.9999999999999861, rel 5.13e-07, VRAM
bookends measured exactly (drained 33,271,808 B, restored
8,932,147,200 B), tenants healthy after. Flash status,
stated plainly: the GPU kernel is flash-STYLE decoding (one
block per Q head, single streaming pass, online softmax, no
materialized score row); shared-memory K/V tile staging and
split-KV parallelism are OWED (occupancy work), recorded in
the module header.

## Lane B4: generation loop + greedy sampler (src/generate/, tests 9/9 in both seam modes)

The brief's function-parameter seam does not compile in Mojo
1.2 (thin calling-convention mismatch, probed both
spellings); the seam as built is a ForwardModel trait
(step/logits) with generate() generic over it. Sampler:
first-max (lowest index) tie-break exactly as
crates/itak-engine/src/sampler.rs; empty logits refuse by
name instead of the Rust stochastic path's silent 0. EOS is
EXCLUDED from the returned ids, uncounted, and never fed
back (crates/itak-engine/src/engine.rs generate_inner; the
speculative session's inclusive truncation is its own rule
and was not followed). Mock proofs assert the exact generated
ids AND the exact (token, position) call sequence across the
prefill/decode boundary, EOS stop, max-tokens stop, and the
named refusals.

## Lane B3: model forward (src/model/model.mojo, 690 lines; rung 1 tests 4/4)

Built by the coordinator directly (the subagent spawn
facility failed twice with a reservation error during this
wave; the lane worktree lane/mojo-b3-model carries the work
and it integrated through the same merge + suite + push
gate). ModelConfig from the file's own metadata (all
muse-glimmer.* keys; the sliding_window_pattern bool array
wins, cross-checked against the derived pattern on a
52-layer file, refusal by name on disagreement). Weights
handle resolves all 731 expected tensors by name and refuses
by name on any miss; weight BYTES ARE CACHED at load (the
loader's per-call staged reads would re-read ~17 GB from
disk per token; the serving wave can revisit). Projections
dispatch by dtype to the Wave A GEMV CPU twins (Q4_K, Q6_K,
Q5_K, F32, F16). The embedding handles its actual storage
dtype by per-row dequant for Q4_K/Q6_K/Q5_K, direct gather
for F32/F16 (the Wave A census says this file's F32 tensors
are the norms, so the quantized path is the live one; the
server proof prints the measured type ids). Step order is
model.rs forward_rows statement for statement, including the
weightless embedding norm, per-head Q/K norms, sigmoid
attn_gate, and both post norms at POST_NORM_EPS 1e-8.
Captures (embed, named-layer in/out) are built in for
divergence isolation.

RUNG 1 (VM, synthetic tiny GGUF through the model's own
loader path, independent in-test Float64 reference): sliding
layer max_rel 7.09e-06, global layer max_rel 5.85e-05,
global 6-token control max_abs 8.04e-07, head argmax sane;
bar 1e-4. One real bug found and fixed on this rung: the
first build rotated Q in the model AND passed it to
attention_cpu_rotated_cache, which rotates the raw Q
internally -- a double rotation that diverged the sliding
fixture at max_rel 4.37 while the global fixture passed.
Fix: Q passes raw; only K rotates, at write time. The fix is
recorded in the module header so the seam contract is not
re-tripped.

RUNG 2 (server, real layer 0 sliding + layer 3 global at
position 0, input = the model's own embedding + plain norm
for token 13796) and RUNG 3 (teacher-forced golden: prompt
[200000, 954, 7963, 323, 11698, 373], predictions must walk
[13796, 26, 1573, 373, 290, 2167, 26899, 1212]) run through
tests/server_proof_model.mojo in a self-restoring cleared
window (phase lock, tenants stopped and restored by trap,
VRAM measured via sysfs at both ends, staging = git archive
of abbd8cf with STAGED_COMMIT stamp and the staged
model.mojo sha256 verified equal to the worktree's
98bb15dc...). The window's log is ~/gpu-wave-b/proof3.log on
the server (two earlier attempts, proof.log and proof2.log,
died at launch on a wrong pixi path -- the server's pixi is
~/mojo-probe/pixi/bin/pixi, NOT ~/mojo-probe/bin/pixi -- and
a wrong VRAM node; both failed windows restored their
tenants cleanly with exact bookends and ran no model code).
The corrected window measured VRAM_DRAINED exactly
33,271,808 B, then: model load 21,150 ms with embd_type=12
(Q4_K, confirming the quantized embedding path is the live
one) and output_type=13 (Q5_K). RUNG 2 EXECUTED: layer 0
(sliding) step 4,055 ms, output sum 65.22733633872122, max
22.1184024810791, first8 [0.08513339, 0.026624471,
-0.42816293, 3.276528, -1.1669861, 2.0494282, -0.08531731,
0.34014347]; layer 3 (global) step 4,115 ms, sum
157.77914571017027, max 19.754636764526367, first8
[1.254974, -0.3695352, -0.49299628, 1.3146128, -0.27929068,
1.1605207, 0.099737644, 0.68446195]. The independent Python
Float64 spot check of those two rows has since been RUN and
rung 2 is COUNTED (see THE GOLDEN VERDICT). RUNG 3 ran to
completion in the same window; its per-position lines, its
exit line, and the window's restore lines are all in
~/gpu-wave-b/proof3.log (verdict in THE GOLDEN VERDICT).

## Integration record

Order: B1 (merge bb4979a + wiring c5c5062, pushed), B2 + B4
(merges + wiring 7a91491, pushed), B3 (merge + wiring
abbd8cf, pushed). Every push SHA-verified MATCH via the
replay push; remote main re-verified at b416987 after each.
The interfaces contract lives at
mojo-engine/WAVE_B_INTERFACES.md (committed with the B1
integration).

## THE GOLDEN VERDICT

PASS, on all three rungs, closed 2026-11-08.

RUNG 2 COUNTED. The independent Python Float64 spot check
(~/gpu-wave-b/spot_check_rung2.py: GGUF parsed and
dequantized from the raw file bytes in float64, each
dequantizer self-checked against a scalar transcription of
the Mojo element functions on 200 random elements per
tensor; layer math recomputed independently at position 0,
where attention is exact and RoPE is the identity) agrees
with the Mojo-recorded rows at float32 noise level:
layer 0 sum rel diff 1.674e-07, max rel diff 9.814e-07,
first8 max abs diff 3.414e-06; layer 3 sum rel diff
1.029e-06, max rel diff 9.188e-07, first8 max abs diff
2.435e-06. SPOT_CHECK PASS.

RUNG 3 PASS 8/8. The teacher-forced walk in
~/gpu-wave-b/proof3.log completed with PROOF3_EXIT 0:
predictions 13796, 26, 1573, 373, 290, 2167, 26899, 1212,
every one equal to its golden id (winning logits 17.34,
18.88, 17.27, 19.25, 17.75, 16.75, 16.04, 15.78; full-step
times 236,040 to 245,846 ms; prediction section total
1,823,416 ms). Top-5 at index 0: 13796 (17.343933), then
262, 290, 1085, 1437, a sane margin.

END-TO-END PASS. The runner (src/runner/generate_main.mojo,
commit 4dc4921, its own feature commit on this branch)
opened the real file through the loader, built the prompt
with the Wave A tokenizer plus BOS (printed PROMPT_IDS
[200000, 954, 7963, 323, 11698, 373], exactly the Rust
harness's construction), wrapped the model in
GlimmerForwardAdapter, and ran B4 generate for 16 tokens
greedy at eos 200001 in a fresh cleared-card window staged
from 4dc4921 (git archive, STAGED_COMMIT stamp, staged
generate_main.mojo sha256 516e780b... equal to the
worktree's). Produced TOKENS [13796, 26, 1573, 373, 290,
2167, 26899, 1212, 6996, 310, 11698, 341, 290, 2167, 26899,
1212]: the first 8 are exactly the golden, and the runner's
own GOLDEN_FIRST8_MATCH printed 1. Decoded text (strict
decode verified, TEXT_STRICT_OK 1): " Paris. It is the most
populous city in France and the most populous" -- fluent,
on-prompt, and the repeated "26899, 1212" tail (", Paris")
is the model's own phrasing, not a loop artifact in our
plumbing. RUNNER_EXIT 0.

FIRST-LIGHT NUMBERS (CPU path, no verdict attached; speed
tuning is a later wave): model open 24,425 ms; prefill
1,411,755 ms for the 6 prompt positions (measured on a
second fresh model stepping the same ids, forward_rows);
generate total 5,216,444 ms for 16 tokens; decode about
253.6 s per step, about 0.00394 tok/s ((generate minus
prefill) over the 15 decode steps, the runner's stated
approximation).

With the golden passed, the closing commit bumps the engine
to 1.3 and marks DESIGN.md step 5 landed (steps 3 and 4
proven; step 6 stays greedy-only).

## Server closeout

B2's window: exact bookends, tenants healthy, port 8080
clear. The rung 2/3 window (proof3.log) closed with
PROOF3_EXIT 0, TENANT_EMBED3 active, WAVE_B_WINDOW3_DONE
present, VRAM_DRAINED3 exactly 33,271,808 B and
VRAM_RESTORED3 2,584,600,576 B. The end-to-end window
(~/gpu-wave-b-e2e/run.log, staged 4dc4921) closed with
RUNNER_EXIT 0, TENANT_EMBED active (re-verified after the
window), VRAM_DRAINED exactly 33,271,808 B, VRAM_RESTORED
2,584,621,056 B, PORT8080 0 before and after,
WAVE_B_E2E_WINDOW_DONE present.

One tenant-set change, recorded plainly because it moves
the restored bookend: the restored VRAM is now about
2.58 GB, not the 8,932,147,200 B of earlier windows. The
cortex-cortex-chat-1 and cortex-cortex-embed-1 containers
no longer exist on the host: both this window's stop and
its restore logged "No such container" for them, and the
rung 3 window's restore listing (TENANT3) already showed
only cortex-cortex-1 and cortex-cortex-surrealdb-1, which
were never stopped and remain up. The absence predates the
end-to-end window (its VRAM_DRAINED still measured the full
33,271,808 B, i.e. the embed service was the only GPU
tenant left to drain) and was not caused by it; the current
tenant set's restored level is the 2.58 GB bookend both
closing windows measured. cortex-embed.service is active.
Glimmer serving stayed down; production llama-server
untouched; no reboot anywhere in this wave.
