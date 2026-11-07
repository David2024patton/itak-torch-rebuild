# Mojo engine Wave B: KV cache, attention, model forward, generation loop

Date: 2026-11-07. Branch `mojo-max-engine`. Engine version
STAYS 1.2: the wave's closing rung (the end-to-end golden on
the server) was still running when this note was written, and
the version bump to 1.3 belongs to the landing that carries
the golden verdict. Remote main never moved off b416987.

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
Float64 spot check of those two rows has NOT been run yet,
so rung 2 is EXECUTED BUT NOT COUNTED. RUNG 3 was still
executing when this update was written (the process was
alive at ~98% CPU in the teacher-forced forward; a full
52-layer step at the rung-2 rate is minutes, and rung 3
needs 13 of them); its per-position lines land in the same
log, followed by the window's restore lines (VRAM_RESTORED3,
TENANT_EMBED3, WAVE_B_WINDOW3_DONE).

## Integration record

Order: B1 (merge bb4979a + wiring c5c5062, pushed), B2 + B4
(merges + wiring 7a91491, pushed), B3 (merge + wiring
abbd8cf, pushed). Every push SHA-verified MATCH via the
replay push; remote main re-verified at b416987 after each.
The interfaces contract lives at
mojo-engine/WAVE_B_INTERFACES.md (committed with the B1
integration).

## THE GOLDEN VERDICT

PENDING: rung 3's verdict (in flight, see above) and the
end-to-end run (integrated tree, real file, B4 generate over
the B3 adapter), which has NOT been run as a separate rung
beyond the proof harness's teacher-forced rung 3. The wave
is therefore NOT declared complete. What remains, in order:
(1) read the rung-3 results from ~/gpu-wave-b/proof3.log,
verify the window's restore lines, and run the rung-2 Python
spot check; (2) wire the end-to-end runner (a main that
opens the real model, wraps it in GlimmerForwardAdapter,
tokenizes "The capital of France is" with the Wave A
tokenizer + BOS, runs B4 generate for 16 tokens at eos
200001, prints the ids) and run it in a cleared window: PASS
is exactly first-8 == [13796, 26, 1573, 373, 290, 2167,
26899, 1212]; (3) record decode tok/s and prefill time from
that run (first-light numbers, no verdict); (4) bump the
engine to 1.3, finish the DESIGN.md step statuses, and close
the wave. If rung 3 or the end-to-end run diverges, the
parity ladder's next step is per-layer capture diffing at
position 0 (captures are built into B3 for exactly this).

## Server closeout

B2's window: exact bookends, tenants healthy, port 8080
clear. The B3/end-to-end window self-restores its tenants by
trap on any exit; its closeout lines (VRAM_RESTORED,
TENANT_EMBED, container status, WAVE_B_WINDOW_DONE) land in
~/gpu-wave-b/proof.log and must be read back before the wave
closes. Glimmer serving stayed down; no reboot anywhere in
this wave.
