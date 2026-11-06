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

## Continuation, 2026-11-06: steps 2-4 RUN on the cleared card

Staged tree: main `c439e3af45e30275601129568e3e9cb3a89590cb`
(the full speed-wave merge head: K19/K20/K21/K22, chunked
GEMV, GGUF mmap, diag drain, host-gap fix), exported with
`git archive`, staged at `~/gpu-wave11` on the server.
Manifest: 364 tracked files, sha256-identical to the local
tree. Card state: the Cortex tenants were already stopped
when the session opened (`cortex-cortex-chat-1` and
`cortex-cortex-embed-1` Exited (0), `cortex-embed.service`
inactive); the per-run guard read
`/sys/class/drm/card2/device/mem_info_vram_used` =
33,271,808 bytes before and after every run in this leg.
No number below was taken with a tenant on the card.
David's 2026-11-06 standing rule (the card is always
available for testing; he will not start Cortex work
mid-session) governed the window. Logs preserved in
`~/gpu-wave11/logs11/` on the server.

### Step 2: identity + golden in the ON position: PASS

`WGPU_COMPUTE_MMV_MAX_ROWS=16`, `spec_generate_gpu`, real
target and draft files, greedy, Stage 1 prompt
("The capital of France is"), 64 tokens:

- depth 6: IDENTITY PASS, GOLDEN PASS.
- depth 1: IDENTITY PASS, GOLDEN PASS.

Phase B (draft + verify sessions in one process, the leg
that was OOM-killed at "target coverage: 731" in every
2026-11-05 attempt) completed at both depths; see the
GGUF mmap confirmation appended to
`bench/gguf-mmap-2026-11-05.md` for the RssAnon evidence.

Recording gap, stated again plainly: no `mmv config:
max_rows=` echo line appears in any captured log, the
same gap recorded on 2026-11-05. ON/OFF attribution
rests on the env each driver script exported plus the
verify-ms separation between arms (below), which is
large and consistent. The missing echo is a harness
logging defect, still open.

### Step 3: spec A/B, fresh same-session cleared-card controls

Effective tok/s (generate phase; all-in in parentheses).
Acceptance was identical per depth in every arm (depth 6:
32 rounds, 180 proposed, 31 accepted, rate 0.1722;
depth 1: 38 rounds, 37 proposed, 25 accepted, rate
0.6757), and resident dispatch counts were identical per
depth across arms (14,717 at depth 6; 17,393 at depth 1),
so every difference below is pure timing on identical
trajectories.

| arm | depth 6 | depth 1 |
|---|---|---|
| control (env unset) | 9.09 (9.06) | 16.98 (16.67) |
| `MMV_MAX_ROWS=16` | **10.27 (10.24)** | **19.26 (18.96)** |

Verify time per round (median of the first 12 rounds,
`spec.verify.end` diag lines): depth 6, on about 124 ms
vs control about 159 ms; depth 1, on about 38.5 ms vs
control 45 ms. Verify is the term K19 exists to shrink,
and at the top of its band it shrinks: depth 6 effective
+13.0 percent, depth 1 effective +13.4 percent over the
same-session controls.

For the record, today's cleared-card controls against
the 2026-11-04/05 pressured-card controls: depth 6 9.09
vs 9.19-9.21 (same class), depth 1 16.98 vs 14.22-14.60
(the cleared card lifts the depth 1 control). Phase A
plain decode inside these runs read 35.38-36.29 tok/s,
the recorded cleared-card band.

### Step 4: dose-response sweep at depth 1

| `MMV_MAX_ROWS` | effective tok/s | verify median |
|---|---|---|
| unset (control) | 16.98 | 45 ms |
| 4 | 14.71 | about 50 ms |
| 8 | 15.23 | about 51 ms |
| 12 | 15.20 | about 50 ms |
| 16 | 19.26 | about 38.5 ms |

The dose response is **non-monotonic**: caps 4, 8, and
12 all land BELOW the unset control, with verify
medians above the control's, while cap 16 lands above
the control on both measures. This is the in-band cliff
pattern the research warned about (the strix-llama fork's
split-column finding) and the pattern this sweep exists
to catch. Per-batch attribution (which batch sizes in
the band carry the loss, and which carry the cap-16
gain) is read from the sweep-harness table and the diag
exposure run in the same 2026-11-06 session, recorded
in `bench/mmv-bench-harness-2026-11-05.md` and
`bench/mmv-q6k-2026-11-05.md`.

### Decision against the prototype note's rule

The rule moves `DEFAULT_MMV_MAX_ROWS` only if identity
and golden pass in the on position AND depth 6 beats
its control AND depth 1 beats its control AND the
dose-response sweep shows no cliff inside the band the
default would open.

- Identity + golden in the on position: PASS (step 2).
- Depth 6 beats its control: YES (10.27 vs 9.09).
- Depth 1 beats its control: YES at cap 16 (19.26 vs
  16.98).
- No cliff inside the band: **NOT SHOWN.** Caps 4, 8,
  and 12 sit below the control at depth 1; the middle
  of the band loses on this card and driver (Mesa
  26.0.8) even though the top of the band wins.

**The default stays 1 (off)**, per the rule as written:
the no-cliff condition is not met, and the rule's own
fallback applies (the default stays and the sweep table
is the finding). The finding, precisely: K19's
register-streaming form is correct everywhere it ran
(identity + golden at both depths, all caps) and fast
at batch 16, but a mid-band range of batches runs
slower through it than through GEMV on RADV, enough to
drag caps 4-12 below the shipped control at depth 1.
Any future flip needs a band floor informed by the
per-batch table (for example, routing only the top of
the band), which is a new prototype decision, not a
default change this verdict authorizes. T-9.6's speed
half remains open: even the winning arm (19.26 tok/s at
depth 1) is far under the 88.6 tok/s bar, because
verify is no longer the whole cost once it shrinks;
the draft and the round overhead dominate what is left.
