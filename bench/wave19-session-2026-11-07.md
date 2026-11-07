# Wave19: first calibration profile on the 7900 XTX, profile-governed run, and the unset-defaults confirmation (2026-11-07)

One consolidated card session on ubuntu-server staging main
0ecdc8ff5efc3da0d00b5806c8f2796b65096132 at ~/gpu-wave19
(fresh git archive; logs under ~/gpu-wave19/logs19/; every
verdict run wrapped in bench/failure-bundle.sh, bundles
under ~/gpu-wave19/bench/failures/). Two things were owed:
the auto-tune foundation's first end-to-end card exercise
(calibrate on the real adapter, then a spec run governed
by the produced profile), and the K21F flip note's
defaults confirmation at literally-unset env plus on-card
execution of the two routing arms the flip added.

**Verdicts, stated first.** The calibrator ran end to end
on the real card and wrote a profile keyed to the exact
adapter and driver; **3 of its 9 band cells match the
shipped table exactly**, and the 6 that differ are listed
cell by cell below with the batch-1 measurement artifact
identified. The profile-governed spec run passes IDENTITY
and GOLDEN with the manifest naming the profile key and
resolving the profile's table wherever it differs from
shipped. The unset-defaults confirmation passes: at
literally-unset env the real file passes IDENTITY and
GOLDEN with 37 K21F finish dispatches, and both new
routing arms EXECUTE and pass on the card.

## Staging verification

sha256 of the staged tree, identical to the same files
extracted from commit 0ecdc8f on this VM:

- crates/wgpu-compute/src/profile.rs:
  5195c05e37da58772167cce9976f689c751dcb641a5f4f361041374473b06d16
- crates/wgpu-compute/src/compute.rs:
  2d40280eec794598d2a55709bcc513970c1f1e5fda3cf899e16fe5621eb6094a
- crates/wgpu-compute/src/session.rs:
  8beb90e01756771bff9e905f5389d95ecba76cc835d9fe08c6821b0487479206

Model sha256 re-verified this session:
4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e
(/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf);
draft file dflash-Muse-Glimmer-30B-Q4_K_M.gguf present.
Tenants stopped before card work (VRAM 33,271,808 B,
driver residue) and restored after (closing section).
All builds and runs in the release profile, serialized
through the phase lock, CARGO_BUILD_JOBS=4.

## Leg 1: calibrate_profile on the real card

Bundle 20261107T-leg1: PASS (index line only). The run
completed both arms (control: GEMV ladder, ceiling pinned
closed; MMV: register-streaming forms, ceiling and bands
pinned open) at the default 25 reps over the 99-cell
matrix and wrote ~/gpu-wave19/profile.txt. Adapter key
detected, exact strings:

- adapter_name = AMD Radeon RX 7900 XTX (RADV NAVI31)
- adapter_backend = Vulkan
- adapter_driver = radv
- adapter_driver_info = Mesa 26.0.8-1ubuntu0.3

Full profile file as written:

```
# wgpu-compute calibration profile
# Measured on the adapter named below by the calibrate_profile
# example; construction applies it only to that exact adapter
# and driver. Hand edits are read back through the same
# parsers the WGPU_COMPUTE_MMV_* env overrides use.
format_version=1
adapter_name=AMD Radeon RX 7900 XTX (RADV NAVI31)
adapter_backend=Vulkan
adapter_driver=radv
adapter_driver_info=Mesa 26.0.8-1ubuntu0.3
mmv_max_rows=16
band_q4k_attn=1,6-12
band_q4k_ffn=2-12
band_q4k_kv=1-2,4-6,16
band_q5k_attn=2-16
band_q5k_ffn=1-16
band_q5k_kv=1
band_q6k_attn=2-12
band_q6k_ffn=2-12
band_q6k_kv=4-12
```

Derived vs shipped, cell by cell (ceiling matches: 16 = 16):

| cell | derived | shipped | verdict |
|---|---|---|---|
| q4k attn | 1,6-12 | 6-12 | differs only at batch 1 (see artifact note) |
| q4k ffn | 2-12 | 2,4-12 | DIFFERS: row 3 now a win |
| q4k kv | 1-2,4-6,16 | 2-3 | DIFFERS: rows 3 lost, 4-6 and 16 gained |
| q5k attn | 2-16 | 2-16 | match |
| q5k ffn | 1-16 | 3-16 | DIFFERS: rows 1-2 gained |
| q5k kv | 1 (closed) | 16 | DIFFERS: inverted, win at 16 now closed |
| q6k attn | 2-12 | 2-12 | match |
| q6k ffn | 2-12 | 2-12 | match |
| q6k kv | 4-12 | 12 | DIFFERS: widens to 4-12 |

So 3 of 9 cells match exactly. Two honest qualifications
on the differing cells:

1. **Batch-1 entries are an artifact, identified.** Three
   derived cells (q4k attn, q4k kv, q5k ffn) include batch
   1. The MMV arm pins its bands to 2-16, so at batch 1
   both arms dispatch the identical GEMV form; the diag
   census of the run confirms it (the batch-1 GEMV
   dispatch count is 50 per shape, 25 per arm, with zero
   MMV dispatches at batch 1). A batch-1 "win" under the
   0.98x rule is therefore inter-arm timing drift between
   two sequential arms measuring the same code path, not
   a form win. Discounting batch 1, q4k attn matches the
   shipped cell on every batch that can route. The q5k
   ffn row-2 gain and the q4k ffn row-3 gain are not
   explained by this artifact.
2. **No per-cell medians are claimed.** The calibrator's
   printed output is the derived table only; it publishes
   no per-cell medians or ratios. The session mined the
   diag dispatch stream for per-dispatch medians, but
   those do not reproduce ProjBench's per-call derivation
   on several cells (they disagree in both directions,
   e.g. q5k kv and q4k attn at batches 9 and 12, because
   a ProjBench cell figure aggregates the whole prepared
   call, not the single product dispatch). The mined
   numbers are therefore not reported as evidence for or
   against any cell. What stands is the derivation
   itself, run at verdict reps on the same card and
   driver the shipped table came from: the kv cells and
   the two ffn boundary rows genuinely re-measured
   differently this run, and any future table change
   should be decided with a sweep CSV beside it, the way
   the shipped table was.

## Leg 2: the produced profile governs a real run

spec_generate_gpu, both mode, 64 tokens, prompt "The
capital of France is", depth 1,
WGPU_COMPUTE_PROFILE_PATH=~/gpu-wave19/profile.txt, no
MMV env overrides, WGPU_COMPUTE_DIAG=capture.

- IDENTITY: PASS (64 tokens token-identical to plain GPU
  greedy). GOLDEN: PASS (first 8 match the verified
  continuation). Round accounting rounds=38, proposed=37,
  accepted=25, identical to the wave18 arms.
- The capture manifest (logs19/leg2/diag.manifest.json)
  names the governor: `mmv_profile = AMD Radeon RX 7900
  XTX (RADV NAVI31) | Vulkan | radv | Mesa
  26.0.8-1ubuntu0.3`, `mmv_max_rows = 16`, and the
  resolved band table is the PROFILE's, not the shipped
  one, in every cell where they differ:
  `q4k[attn 6-12 ffn 2-12 kv 2,4-6,16] q5k[attn 2-16
  ffn 2-16 kv none] q6k[attn 2-12 ffn 2-12 kv 4-12]`
  (profile q4k ffn 2-12 over shipped 2,4-12; q4k kv
  2,4-6,16 over shipped 2-3; q5k ffn 2-16 over shipped
  3-16; q5k kv closed over shipped 16; q6k kv 4-12 over
  shipped 12). The describe rendering drops batch 1 from
  cells that carry it, which is why the profile's
  "1,6-12" resolves to a described "6-12".
- Bundle bench/failures/20261107T073310Z-leg2-profile-governed
  exists for one reason only: FAIL(2) = the SPEED line
  (16.87 tok/s under the standing 88.6 bar) and its T-9.6
  summary line. Verified in the bundle log: no other FAIL
  lines; IDENTITY and GOLDEN pass in the same run.

## Leg 3: unset-defaults confirmation (the flip note's owed leg)

Precondition checked first: no profile file exists at
the default path for the server account (no
~/.config/itak directory; XDG_CONFIG_HOME unset), so
nothing had to be moved aside and nothing was.

(a) Real file at literally-unset env. All WGPU_COMPUTE_*
variables unset inside the run's shell; only
WGPU_COMPUTE_DIAG=timing and its OUT were then set.
spec_generate_gpu as in leg 2:

- IDENTITY: PASS. GOLDEN: PASS.
- k21_q5k_mmv_finish fired 37 times at batch=2, finish=2
  on the real head geometry (rows=202048, cols=6656):
  the head route is on at the shipped default position,
  with no env spelling involved. (The per-session diag
  CSV at the shared OUT path was overwritten by the
  second session, the known behavior from wave16; the
  stamp count is from the run's own dispatch stream in
  logs19/leg3/run.out.)
- Bundle bench/failures/20261107T073456Z-leg3-unset-defaults:
  FAIL(2), the same two speed lines only (17.07 tok/s),
  verified in the bundle log.

(b) The two new routing arms, on the card
(tests/k21f_head_routing.rs, release binary, each run
with WGPU_COMPUTE_DIAG=timing, --nocapture):

- Both-envs-unset position (bundle leg3-routing-unset:
  PASS): 4 passed, 0 failed.
  `block_head_routes_to_k21f_at_default_construction`
  EXECUTED and passed (re-run standalone to isolate its
  line: 1 passed, and its session's dispatch stream
  contains the k21_q5k_mmv_finish stamp). The other arms
  skipped with named reasons, verbatim in the log:
  switch-off arm ("WGPU_COMPUTE_K21F_HEAD is unset, which
  resolves the head switch ON at the shipped default;
  this arm needs WGPU_COMPUTE_K21F_HEAD=0"), open-gate arm
  (ceiling env does not admit), closed-gate arm (needs
  WGPU_COMPUTE_MMV_MAX_ROWS=1).
- Explicit-off position, WGPU_COMPUTE_K21F_HEAD=0
  (bundle leg3-routing-headoff: PASS): 4 passed, 0
  failed. `block_head_stays_k3_when_the_head_switch_is_off`
  EXECUTED and passed (its session ran; no skip line for
  it), while the default-construction arm skipped naming
  the set HEAD var, and the open/closed-gate arms skipped
  with the same named reasons as above.

## Closing verdicts and state

- **Calibration:** profile produced on the real card,
  keyed [AMD Radeon RX 7900 XTX (RADV NAVI31) | Vulkan |
  radv | Mesa 26.0.8-1ubuntu0.3], ceiling 16; 3 of 9
  cells match the shipped table exactly (q5k attn, q6k
  attn, q6k ffn), q4k attn matches on every routable
  batch, and 5 cells genuinely differ (q4k ffn row 3,
  q4k kv reshaped, q5k ffn rows 1-2, q5k kv inverted to
  closed, q6k kv widened to 4-12). The batch-1 entries
  in derived cells are a named artifact of measuring two
  sequential arms. No table change is proposed by this
  session; the profile mechanism is what was on trial,
  and it works end to end.
- **Profile governance:** PASS. A run pointed at the
  profile resolves the profile's cells over the shipped
  defaults in every differing cell, names the key in its
  manifest, and stays token-identical.
- **Unset defaults:** PASS. The K21F flip's owed
  confirmation is discharged: at the literal shipped
  default position the route fires (37 finish stamps),
  identity and golden pass on the real file, and both
  flip-era routing arms execute green on the card.
- **Bundles:** leg1-calibrate PASS; leg3-routing-unset
  PASS; leg3-routing-headoff PASS;
  20261107T073310Z-leg2-profile-governed and
  20261107T073456Z-leg3-unset-defaults are speed-line-only
  (FAIL(2) = SPEED + T-9.6 summary in each, verified in
  their logs; the 88.6 tok/s line is the standing T-9.6
  bar, not a leg verdict).
- **Tenants restored:** docker start
  cortex-cortex-chat-1 cortex-cortex-embed-1 (both Up,
  healthy), cortex-embed.service active. VRAM settled at
  8,932,184,064 B (the standing ~8,932,147,200 B plus
  36,864 B of driver-level variance). Production
  llama-server never touched; no reboot.
