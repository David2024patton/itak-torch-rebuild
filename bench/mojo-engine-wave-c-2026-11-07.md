# Mojo engine Wave C: the forward on the GPU

Date: 2026-11-07, closed 2026-11-08 (the final window ran
2026-11-08 03:26 UTC, 22:26 EST). Branch `mojo-max-engine`.
Engine version 1.4 as of the wave's closing commit: the
golden verdict on the GPU path is in and it PASSED (see THE
GOLDEN VERDICT below). Remote main never moved off b416987.

The seam contract the lanes coded against is
`mojo-engine/WAVE_C_INTERFACES.md` (committed first, at
da91f34). The port shape, as shipped: weights and the KV
cache resident on device, per-position step semantics
identical to the CPU model, activations as host rows at the
layer boundary (each op uploads its input row, launches
against resident weights/KV, downloads its output row).

## Memory budget: planned vs observed

Planned (contract, before the first run): weights at most
16,756,683,904 B (the whole file as the bound), device KV
3,653,238,784 B at context 131,072, scratch envelope
536,870,912 B; planning total 20,946,793,600 B against a cap
of 24,696,061,952 B (card total 25,769,803,776 B minus a
1 GiB reserve; `MOJO_ENGINE_GPU_BUDGET_BYTES` overrides).

Observed (Lane C1's server proof, staged 5147c2e): exact
resident weights 16,743,567,360 B (the 13,116,544 B under
the planning bound is file header/metadata), device KV
3,653,238,784 B exactly as planned, reserved total at full
build 20,933,677,056 B (exact weights + KV + scratch
envelope), under the cap with about 3.76 GB to spare. The
budget is enforced by the runtime ledger BEFORE allocation
(`gpu_budget_exceeded` refused first). One ledger behavior
to know: the ledger is per-`GpuRuntime`; it does not sum
across runtimes in one process (see the runner phasing fix
under Integration - that is why the phasing matters).

## Lane C1: device plumbing (src/gpu/, config addition)

`GpuRuntime` (DeviceContext ownership, budget ledger,
upload/download helpers, refusals `gpu_no_device` /
`gpu_budget_exceeded` / `gpu_copy_shape`), `DeviceWeights`
(every slot resident: quantized/F16 as uint8 buffers over
raw bytes, F32 as float32 buffers; named accessors for
embedding/output/final-norm; per-layer access by (layer,
suffix_index)), `DeviceKvCache` (B1 layout byte for byte;
append via one-row staging + scatter kernel; B1 refusal
names under a `gpu_kv_` prefix). Config gained
`MOJO_ENGINE_GPU_BUDGET_BYTES` with the same parse/refuse
discipline as the other values.

Server proof (cleared-card window, log read in the
foreground from ~/gpu-wave-c1/run.log): PROOF GPU_PLUMBING
PASS, 35 PROOF EXACT lines. blk.0.ffn_gate.weight
downloaded from device hashes FNV-1a64
0x88b58419387d763c, equal to the host's, 0 byte mismatches;
an F32 slot (blk.0.attn_norm.weight) downloads value-exact.
Device KV sequences element-exact against B1's CPU cache
fed identical rows (synthetic window-4 + global; the real
2048 window with reads at 0, 1, 777, 2046, 2047; a global
sibling). DeviceWeights built in 7,021 ms.

DEVIATION, kept and recorded: B1's CPU cache accepts
windowed appends past the window and evicts (ring); the
device cache refuses the first wrap append with
`gpu_kv_window_wrapped`, exactly the contract's stated Wave
C limit (the attention kernel consumes visible rows in slot
order from offset 0). The proof demonstrates both behaviors
on identical rows. A device-side visible-order gather is
owed work; contexts past the window stay on the CPU path.

VM findings from the lane: `mojo build` AOT fails with
target 'unknown' for any file that launches GPU kernels on
the GPU-less VM (pre-existing; reproduced on Wave A's
untouched gpu_proof_ops.mojo). Those mains compile on the
card host. Lane C3 later found the VM-side compile route:
`mojo build --target-accelerator gfx1100`.

## Lane C2: GPU layer assembly

`src/model/gpu_layer.mojo` (777 lines): `gpu_layer_step`
statement-for-statement the CPU `layer_step` (norms via
rms_norm_kernel with resident norm buffers; projections by
slot dtype; per-head Q/K norms grid=heads width=head_dim;
sliding layers rotate K before the KV append and Q before
the attention launch, base 500,000 - the kernel contract is
PRE-ROTATED Q/K; attention_kernel on the DeviceKv buffers;
a new `attn_gate_kernel` for attn*sigmoid(gate); swiglu;
post norms at 1e-8; residual adds are HOST adds on the
downloaded boundary rows, stated in the header) and
`gpu_logits` (final norm + Q5_K head over 202,048 rows +
final_logit_kernel, scale/softcap from the CPU model's
config). `src/ops/gpu_embedding.mojo`: device dequant of
one embedding row for the table's actual dtype.

Attempt ladder: (1) first build against guessed contract
surface with per-call norm uploads - dropped mid-lane when
real C1 landed with resident accessors; (2) adapted to real
C1 (`DeviceWeights.build`/`DeviceKvCache.build`, mut rt,
slot views) - kept; (3) server proof against real C1 on a
merge tree with the lane's VM stand-in package deleted from
staging - PASS on every leg, first card attempt.

Server proof numbers (bars: cosine >= 0.999999, max rel
<= 1e-4, argmax exact; log ~/gpu-wave-c2/proof.log):
embedding tokens 13796 and 200000 max abs 0.0 (exact).
Layer 0 (sliding, position 0): cosine 0.9999999999985569,
max rel 2.33e-06, GPU sum 65.22703143954277 vs the counted
65.22733633872122; GPU 16 ms vs CPU 4,002 ms. Layer 3
(global, position 0): cosine 0.9999999999983837, max rel
8.41e-07, GPU sum 157.77902115881443 vs counted
157.77914571017027; GPU 15 ms vs CPU 4,110 ms. The counted
full rows were never published (sum + first8 only), so the
proof verified its in-process CPU row against the counted
record, then the GPU row against that CPU row over the full
row. Logits (on the CPU layer-0 output row, the brief's
pure-function construction): argmax 150015 == 150015 exact,
cosine 0.9999999999987915, max rel 2.85e-06; GPU 35 ms vs
CPU 16,801 ms. GPU_LAYER_PROOF PASS.

Unproven legs, plainly (from the lane): GEMV dispatch
branches for dtypes not present in layers 0/3, the head, or
the embedding were compile-proven only; multi-position
attention (stored > 1) is covered by Wave A's stored=64
attention proof and by this wave's end-to-end run, not by
the layer proof.

## Lane C3: GPU generation

`src/model/gpu_model.mojo`: `GpuGlimmerModel` implementing
the same `ForwardModel` trait over the device trio (CPU
model held for config, captures, fallback); step mirrors
the CPU step (device embedding, plain rms_norm on device,
52 x gpu_layer_step); logits via gpu_logits.
`src/runner/generate_main.mojo`: GPU selection by
`MOJO_ENGINE_GPU` (any value other than "" and "0"), CPU
default byte-identical to Wave B, one added printed line
`MODEL_PATH_KIND gpu|cpu`.
`tests/server_proof_gpu_generate.mojo`: the teacher-forced
device walk (rung 3's exact construction).

The lane coded against contract signatures with bannered
stand-in files for C1/C2 (its report lists all six); at
integration the stand-ins were resolved to the real
surfaces (see Integration). No GPU execution was claimed by
the lane; the card proofs are C1's, C2's, and the final
window's.

## Lane C4: measurement

`tests/wave_c_timing.sh` (451 lines): staged-stamp
verification, teacher-forced + two GPU generate runs (run 1
warmup, run 2 timed, both printed), extraction from the
on-disk logs in the foreground, loud named failures on a
missing or non-1 GOLDEN_FIRST8_MATCH, Wave B CPU comparison
lines with plain arithmetic ratios. Validated on synthetic
logs including the failure branches. shellcheck NOT RUN
(not installed on the VM; standing note). One wart found in
the final window: the harness's extracted
TEACHER_FORCED_VERDICT_LINE matched the proof's header line
(`GPU_GENERATE_PROOF context_len 4096`) instead of the
verdict line, because the proof's verdict line
(`TEACHER_FORCED_GPU 8/8`) matches none of the harness's
preferred shapes before the generic GPU-proof shape. The
harness's hard gates all held (TEACHER_FORCED_EXIT=0 is the
proof's own exit, nonzero on any mismatch), and the
coordinator read the teacher log directly (below). Fixing
the extraction shape list is owed harness hygiene, recorded
here rather than silently patched after the fact.

## Integration record

Order: C4 (merge 3f92928, pushed), C1 (merge 39ebf1c +
test wiring 834f437, pushed), C2 (merge d3773ee + stand-in
drop + test wiring 3e82f2d, pushed), C3 (merge 01710fd +
reconciliation 7ba701f, pushed), runner phasing fix
(521cb5e, pushed). Full suite after every integration:
PASS each time (see the tally below). Remote main
re-verified at b416987 after every push.

Reconciliation fixes in 7ba701f (coordinator): C3's six
bannered stand-ins deleted in favor of the real C1/C2
files (add/add conflicts resolved to the integrated side);
builder spellings `GpuRuntime.create` /
`DeviceWeights.build` / `DeviceKvCache.build`;
`gpu_logits` gained its real fourth argument (cpu_model);
`GpuGlimmerModel.open` threads `EngineConfig.gpu_budget_bytes`
so the env override is honored end to end; and the one
surface change: `ForwardModel.logits` is now `mut self`
(the CPU adapter, the generate mock, and this wave's GPU
model adjusted together) because a device model's logits
enqueue work on its runtime - generate() already held its
model mut, CPU semantics unchanged.

Runner phasing fix in 521cb5e (coordinator, caught by
arithmetic before the window, not by a failed run): the
Wave B runner construction holds the generate model and
the fresh prefill-measurement model alive simultaneously.
On CPU that is host RAM; on device it is two resident
models, about 41 GB against a 24 GB card - it would have
failed mid-run. The GPU path now phases the constructions
(generate phase destroys its model before the prefill
phase builds its own). Printed lines and their order are
unchanged; the device-less runner output on the tiny
fixture stayed byte-identical to Wave B plus the
MODEL_PATH_KIND line, re-verified after the fix.

Suite tally, split honestly: on the VM (no device),
`tests/run_tests.sh` runs 23 TestSuite files - all CPU or
pure-path tests - plus the seam-on diag rerun, config
probes, and the engine hello: PASS, 0 failed, at every
integration and at the closing tree (engine hello now
prints 1.4). Case counts by lane inside that: C1 plumbing
6/6, C2 layer VM 5/5, C2 embedding VM 2/2, config suite now
5 tests; Wave B's close counted 110 green at its tree. GPU
code does not execute on the VM: the runner and
teacher-forced proof were compile-verified there with
`mojo build --target-accelerator gfx1100` (exit 0 both) and
run to their designed refusals. GPU execution happened
only on the card: C1's plumbing proof, C2's layer proof,
and the final window below.

## THE GOLDEN VERDICT

PASS on the GPU path, staged from 521cb5e (git archive,
STAGED_COMMIT stamp, staged generate_main.mojo sha256
2df2346c... equal to the worktree's), cleared-card window,
~/gpu-wave-c-final/ logs read in the foreground by the
coordinator.

Teacher-forced on device (tests/server_proof_gpu_generate
.mojo, context 4096, ~gpu-wave-c-final/out/
teacher_forced_gpu.log): TEACHER_FORCED_GPU 8/8, every
prediction equal to its golden id, exit 0. Winning logits
17.343926, 18.877588, 17.273037, 19.247173, 17.747236,
16.751627, 16.038439, 15.78144 - matching Wave B's CPU
rung-3 winning logits (17.34, 18.88, 17.27, 19.25, 17.75,
16.75, 16.04, 15.78) to about 1e-3. Per-step about 109 ms.

End-to-end (the runner, MOJO_ENGINE_GPU=1, 16 tokens
greedy, eos 200001): BOTH runs printed
GOLDEN_FIRST8_MATCH 1 and TEXT_STRICT_OK 1, and both
produced the identical 16 tokens
[13796, 26, 1573, 373, 290, 2167, 26899, 1212, 6996, 310,
11698, 341, 290, 2167, 26899, 1212] - the first 8 exactly
the golden, and the full 16 identical to the CPU Wave B
run's production, token for token. Decoded text: " Paris.
It is the most populous city in France and the most
populous", identical to the CPU run's text.

Measurements (timed run = run 2, per the harness protocol;
run 1 in parentheses where it differs): model open
LOAD_MS 31,759 (run 1: 20,974) - this is the CPU model
read plus the full device upload, and it varies with host
cache warmth; prefill 684 ms (685) for the 6 prompt
positions; generate total 2,950 ms (2,945); decode
(2,950 - 684) / 15 = 0.1511 s/step, 6.62 tok/s. Against the
Wave B CPU first-light lines, plain arithmetic only:
prefill 1,411,755 ms -> 684 ms (about 2,064x), decode
about 253.6 s/step -> 0.1511 s/step (about 1,679x), model
open 24,425 ms -> 31,759 ms (the GPU open is SLOWER: it
pays the same host read plus a 16.7 GB upload; ratio
0.77x). No fresh CPU control was run in the window (a CPU
16-token run is about 1.5 hours; the comparison control is
the Wave B record, labeled recorded-not-concurrent in the
harness summary, exactly as the protocol states).

With the golden passed on device, the closing commit bumps
the engine to 1.4 and updates DESIGN.md's status and step
5.

## Server closeout

Final window (~gpu-wave-c-final/window.log, read in the
foreground): PORT8080 0 before and after; VRAM_PRESTOP
2,584,600,576 B; VRAM_DRAINED 33,284,096 B (12,288 B above
the 33,271,808 B drained figure of the earlier windows -
the observed number, reported as measured); HARNESS_EXIT
0; TENANT_EMBED active after restore; cortex-cortex-1 and
cortex-cortex-surrealdb-1 Up throughout (never stopped;
the retired cortex-cortex-chat-1 / cortex-cortex-embed-1
were not referenced); VRAM_RESTORED 2,584,600,576 B;
WAVE_C_FINAL_WINDOW_DONE present. Lane windows (C1, C2)
closed with the same shape: only cortex-embed.service
stopped and restored, drained measured at exactly
33,271,808 B in both, restored 2,584,600,576 B, port 8080
clear. Glimmer serving stayed down; production
llama-server untouched; no reboot anywhere in this wave.
All windows ran well clear of the 2026-11-08 02:00-04:00
ET maintenance window (the final window closed at 22:26
EST).

## Still owed after Wave C

- Device-side visible-order KV gather, so windowed layers
  can run past the 2,048-position wrap on device (today:
  named refusal, CPU path serves those contexts).
- Attention occupancy work (shared-memory K/V tiling,
  split-KV), recorded since Wave B.
- Activation residency (this wave crosses activations per
  op at the layer boundary; a fully device-resident chain
  is a later optimization).
- GEMV tuning beyond K1 (Q6_K/Q5_K/F32 are straight ports,
  from Wave A); K1 itself sits at 624.1 GB/s against its
  correctness gate, unchanged.
- A budget-ledger release/unreserve call (C2's proof reset
  the ledger field directly between layer proofs; harmless
  there, owed as API hygiene).
- wave_c_timing.sh verdict-line extraction shapes (wart
  recorded under Lane C4 above).
