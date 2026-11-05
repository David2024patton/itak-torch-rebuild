# K19 register-streaming Q4_K MMV: card verdict, 2026-11-05

Verdict protocol from `bench/mmv-prototype-2026-11-05.md`, run on
the cleared RX 7900 XTX (ubuntu-server, gfx1100, Mesa 26.0.8 RADV).

**Staged tree: main `dff3494639ac7e6ea652e9f7743cc57bae079e38`**
(the merge of lane/mmv-prototype `40c11c3`), exported with
`git archive`, staged at `~/gpu-wave10` on the server. Full
manifest diff against the local tree at that SHA: 335 files,
identical. Sentinels: `crates/wgpu-compute/src/kernels/k19_q4k_mmv.rs`
sha256 `f47bb3c7dd1e976ee97ff4648433baaebdf1a50364ffe971def4531362a3b273`,
`crates/wgpu-compute/src/kernels/mod.rs`
`fad8c52e05ef2200e0cb47d9cfcd88ebbbc2264e0c7fc47cfebb4d06c3b6afa6`,
this verdict's protocol source `bench/mmv-prototype-2026-11-05.md`
`dbd9b0cc864dc54ef56c1fc9d16192d32e548e8485b580d427d8bca691bef6c0`.

**Card state and the taint record.** Before the first
measurement, and at every per-arm guard through the measurement
runs, `/sys/class/drm/card2/device/mem_info_vram_used` read
33,271,808 bytes: driver residue only, no tenant resident.
The 16:47-16:52 EST tenant blip the Cortex chat reported had
already been reverted when the first run acquired the phase
lock (21:51:33 UTC; the window is 21:47-21:52 UTC), and the
guard reading at that moment was the same 33,271,808 bytes, so
no number in this verdict was taken with a tenant on the card.
At 22:03 UTC a later Cortex deploy restarted
`cortex-cortex-chat-1` and `cortex-cortex-embed-1`
(VRAM 6,380,896,256 bytes); the queued run 3 guard stopped
before any measurement, per the standing taint instruction.
Tenants are left as found; nothing was stopped or started by
this lane beyond its own harness processes.

## Step 0: shader autopsy (RADV_DEBUG=shaderstats,asm,cs)

All 15 batch variants of both shapes were compiled under the
dump (33 compute shaders total; the two non-K19 blocks are
small device-internal shaders). Series attribution is proven,
not assumed: the subgroup batch 2 variant appears twice with
the identical driver pipeline hash (2617757115), once from the
`K19Pipeline::new` construction smoke compile and once from
`with_shape`; the subgroup series carries DPP shuffle counts of
exactly 4 per accumulator slice, the tree series carries
s_barrier counts growing 4 per batch step.

Subgroup reduction form (the form the verdict runs use; the
server adapter grants SUBGROUP):

| batch | VGPR | spilled VGPR | scratch | LDS | code size |
|---|---|---|---|---|---|
| 2 | 72 | 0 | 0 B | 1,024 B | 3,292 B |
| 7 | 96 | 0 | 0 B | 1,024 B | 7,984 B |
| 16 | 96 | 0 | 0 B | 1,024 B | 16,660 B |

Every batch 2..16 in this form: VGPR 72 up to a 96 plateau,
**zero spilled VGPRs, zero scratch bytes**, LDS 1,024 B
(RADV's reported allocation; the declared partials array is at
most 256 f32 = 1 KB, at batch 16; no weight byte is ever
staged). Workgroup-tree fallback form, for the record: batch 2
VGPR 72 / code 3,828 B, batch 7 VGPR 84 / 8,712 B, batch 16
VGPR 96 / 17,720 B, likewise zero spills and zero scratch at
every batch.

**Register residence held, at batch 16 included.** The failure
mode the prototype note named as its first risk, the one that
convicted the wide MMQ (1,041-1,213 spilled VGPRs and
78,080-86,528 B of scratch in the 2026-11-04 autopsy), does
not exist in K19's compiled form. SGPR reads 108 in every
variant, the ACO allocation constant already recorded in the
earlier autopsy, not a signal.

## Step 1: executed correctness on the card

`cargo test -p wgpu-compute --test k19_mmv` on the server:
1 passed, executed on a real adapter (nothing skipped).
Batches 2, 3, 5, 7, 9, 16 checked against the CPU f32
reference and against K1/GEMV in both reduction shapes; max
abs diffs 9.8e-4 to 2.0e-3 on fixture scales of 8.4e3 to
1.55e4 (worst relative error about 1.8e-7); the
subgroup-form-vs-GEMV differences were exactly 0.0 in every
reported line. The deep-geometry case (768 x 101) and the
band-edge refusal checks passed in the same run.
`cargo test -p wgpu-compute --test shader_compile k19_mmv`:
1 passed, executed (compiles all 15 batches x 2 shapes
through Naga and the driver).

## Step 2: identity + golden in the ON position: BLOCKED by the box, not by K19

The spec harness's phase B (DFlash draft session plus the
verify target session in one process) was killed by the
kernel OOM killer in every attempt:

- both-mode, `WGPU_COMPUTE_MMV_MAX_ROWS=16`, depth 6:
  SIGKILL 21:54:04 UTC; kernel log: `Out of memory: Killed
  process 1875590 (spec_generate_g) total-vm:44461016kB,
  anon-rss:34678624kB`.
- split-mode (plain stream precomputed, spec phase alone),
  depth 6: SIGKILL 21:59:49 UTC; anon-rss 34,703,672 kB,
  same footprint, same death point: immediately after
  `target coverage: expected=731 found=731 missing=0`,
  during verify target session construction.
- split-mode, depth 1: SIGKILL 22:01:25 UTC, same signature.

The box baseline while this ran: about 90 GB of 123 GB used
(Cortex SurrealDB RSS 31,321,904 kB, Chris's VM QEMU RSS
26,524,368 kB), about 33 GB available, swap 3 GB of 3 GB used.
Phase B completed four times on 2026-11-04 for the K18
verdict; the headroom it needs (about 35 GB anon peak) is not
there today. Corroboration that this is the box and not this
lane: the cleared-card baseline lane's spec phase died at the
identical log line at 21:36 UTC (`~/gpu-wave8/run.log` ends
right after its phase B `target coverage` line, and its
d6.out/d1.out contain no verdict lines), so **no fresh
cleared-card controls exist**; the step 3 fallback remains
the K18 note's pressured-card controls, depth 1 14.22 tok/s,
depth 6 9.19 tok/s.

Nothing adverse about K19 was observed before the kills: in
the ON-env both-mode run, phase A completed with plain
prefill 1.778 s and a golden-correct plain stream at
35.81 tok/s decode (batch 1 traffic, below the band, as
designed), draft coverage 58/58, target coverage 731/731.
Identity and golden for the spec phase were never reached, so
they are recorded as NOT EVALUATED. An earlier run script's
gate line worded this as a correctness rejection; that
wording is wrong and is superseded by this note.

A queued workaround probe (all arms at `--kv-capacity=2048`,
a session-capacity config that changes no math at this
committed length of about 70 positions) never ran: the
tenants returned at 22:03 UTC and the guard stopped the run
first (see the taint record above).

## Steps 3 and 4: not run

The protocol gates speed work behind step 2. No A/B and no
dose-response numbers exist, and none are claimed. One
recording gap, stated plainly: no `mmv config` echo line
appears in any captured harness log from these runs, plain
arms included, so the ON/OFF distinction in the preserved
scripts is by the env value each arm exported
(`~/gpu-wave10/run_all.sh`, `run2.sh`, `run3.sh` and
`logs/`, `logs2/`, `logs3/` on the server).

## Step 5: plain decode, no leak: PASS

Cleared card, split-mode plain arms, 64 tokens, prompt
"The capital of France is":

| arm | decode rate | stream |
|---|---|---|
| MMV unset (shipped default) | 35.45 tok/s | (reference stream) |
| `WGPU_COMPUTE_MMV_MAX_ROWS=1` | 35.41 tok/s | byte-identical to unset |

Both streams begin with the golden first 8 tokens
[13796, 26, 1573, 373, 290, 2167, 26899, 1212]. The unset
rate lands exactly on the recorded plain-decode reference of
35.45 tok/s. `max_rows=1` is indistinguishable from off, as
designed (the band starts at row 2).

## Decision

**The default for `WGPU_COMPUTE_MMV_MAX_ROWS` stays 1
(off).** The prototype note's rule moves the default only if
identity and golden pass in the ON position AND depth 6
beats its control AND depth 1 beats its control AND the
sweep shows no in-band cliff. The first condition could not
be evaluated on this box today (host-RAM OOM in the harness's
phase B, then the tenants' return), so the rule cannot be
satisfied and the default does not move. The reason is
evidentiary, not a K19 failure: every gate that could run
passed, including the compiled-form question the prototype
most needed answered (step 0: register residence holds at
batch 16, zero spills) and the executed numeric gates
(step 1), and step 5 is clean.

Unblockers, in order: (1) re-run steps 2-4 when the card is
in the cleared state again AND phase B can get its roughly
35 GB host peak, either from a lower-RAM moment on the box or
by first proving the `--kv-capacity=2048` probe lets phase B
fit; (2) the fresh cleared-card spec controls are owed by the
same wall (the baseline lane hit it too) and should be taken
in the same session as the retry, so the A/B stays
same-session.
