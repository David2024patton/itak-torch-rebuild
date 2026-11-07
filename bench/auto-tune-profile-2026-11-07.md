# Auto-tune profile foundation: adapter-keyed calibration profiles (2026-11-07)

Lane: `lane/auto-tune-profile`, based on main
fa3299768c4e880c144bab8cd4f244ff0c5aacc6. VM-side build
and gates only; no card run in this lane (the card
session is owed, below).

## Why

The MMV routing policy (the nine-cell band table plus
the ceiling in `crates/wgpu-compute/src/compute.rs`)
shipped as constants derived from one card's sweep
(the 7900 XTX numbers in
bench/mmv-band-routing-2026-11-06.md) plus env
overrides. That table is measured truth for that card
and an unexamined guess for every other adapter and
driver. David's original Go Torch solved the same
problem with gpu_detect/auto_config: probe the
hardware, tune for it, remember the result. This lane
builds the foundation of the Rust equivalent: a
versioned calibration profile, keyed to the adapter
and driver that produced it, that construction loads
and folds into MMV resolution, plus the calibration
entry point that produces one from the existing sweep
surface.

## What landed

### The profile (crates/wgpu-compute/src/profile.rs)

- `AdapterKey`: adapter name, backend, driver, and
  driver_info, taken from the same `AdapterFacts` the
  diag capture manifest records, and rendered the same
  way (`format!("{:?}", backend)`), so a manifest and
  a profile name the same machine identically. The
  key includes `driver_info` deliberately: it is the
  version-bearing half of the driver identity, so a
  driver upgrade changes the key and retires profiles
  measured under the old driver instead of applying
  them silently.
- `CalibrationProfile`: a `format_version` (currently
  1), the key, an optional MMV ceiling, and the nine
  band cells stored as text in the env overrides' own
  syntax, parsed by the same `parse_mmv_band` at
  resolution time. A profile states opinions only
  where it has them; unstated cells fall through.
- File format: hand-rolled `key=value` lines with a
  `#` comment header. Judgment call: serde is present
  in the workspace lockfile only transitively; it is
  NOT in wgpu-compute's dependency set, so per the
  lane's terms the format is hand-rolled in the
  band-cell parser's style rather than adding a
  dependency for nine strings and a number. The file
  stays diffable and hand-editable, and a hand typo is
  handled by the same named-fallback logging as an env
  typo. Parse rules: unknown keys are ignored (a newer
  file with extra fields still loads under its own
  version), duplicate known keys and non-`key=value`
  lines are malformed, missing identity fields are
  malformed.
- `load_profile` returns `Loaded` / `Absent` /
  `Rejected(reason)`: absent (no file) is the normal
  uncalibrated case; rejected covers unreadable,
  malformed, wrong format version, and adapter-key
  mismatch, each with the path and reason stated.
  Nothing applies a profile it could not verify, and
  nothing panics.

### Resolution (crates/wgpu-compute/src/compute.rs)

Construction loads the profile for the opened
context's adapter before resolving the MMV policy,
and each value resolves in this precedence:

1. The env override (`WGPU_COMPUTE_MMV_MAX_ROWS`, the
   nine `WGPU_COMPUTE_MMV_*_ROWS` cells), when present
   and parsable.
2. The loaded profile's value, when it states one and
   (for cells) it parses.
3. The shipped default (ceiling 16; the shipped band
   table).

Two fail-closed details, matching the module's
existing philosophy:

- A present-but-unparsable ceiling env value still
  lands on the closed fallback (1) and never reaches
  the profile layer: a typo must not widen routing
  through any layer.
- An unparsable env cell falls through to the profile
  layer (the next statement of that cell's value), and
  the construction log line names which layer actually
  governs (profile cell or shipped cell). An
  unparsable profile cell falls through to the shipped
  cell, also logged by name.

Observability: the construction log's `mmv config:`
line now carries `profile <key|none>`, and both
sessions' capture manifests gain an `mmv_profile`
resolved_config pair (the loaded key's one-line
description, or `"none"`), beside the existing
`mmv_bands` pair. A run can always state which table
governed it and where the table came from.

### Path rule

`profile_path()` resolves:

1. `WGPU_COMPUTE_PROFILE_PATH`, when set and
   non-empty, exactly as given.
2. `$XDG_CONFIG_HOME/itak/wgpu-compute-profile.txt`
   when `XDG_CONFIG_HOME` is set and non-empty, else
   `$HOME/.config/itak/wgpu-compute-profile.txt`.
3. Neither set: no path, no profile, shipped defaults.

### Calibration (examples/calibrate_profile.rs)

Runs the shipped derivation's own two arms back to
back in one process through `ProjBench` (so the
variant measured is the variant a forward would
dispatch), on the sweep harness's exact matrix: quant
{Q4_K, Q5_K, Q6_K} x shapes {attn 6656, ffn 19968,
kv 256} from hidden 6656 x rows {1..9, 12, 16}, 25
timed repetitions (3 warmup) per cell, diag timestamps
when the device grants them and wall clock otherwise,
the harness's synthetic fixtures and agreement rule
copied verbatim so the medians are the same
measurement the shipped table came from.

- Control arm: ceiling pinned to 1, K18 and chunked
  pinned off: the GEMV ladder.
- MMV arm: ceiling 16 and all nine band cells pinned
  to `2-16`, so each cell measures the
  register-streaming form itself wherever the kernels
  admit it, not the shipped table's opinion.
- A row wins when its MMV median is under 0.98x its
  control median AND both arms agreed with the direct
  GEMV form (`profile::mmv_wins`, the win rule from
  bench/mmv-band-routing-2026-11-06.md: a tie is not a
  win). Sweep-adjacent wins group into ranges
  (`profile::derive_band_ranges`, the shipped table's
  own convention: its Q4_K FFN "2,4-12" is a row-2 win
  plus the 4-through-12 run); a cell with no wins is
  written as the closed cell "1", never omitted where
  the calibration measured it. The profile's ceiling
  is written as the kernels' full band (16).
- The arms pin every measured knob in the env, so a
  profile already on disk cannot contaminate the
  measurement (env wins per cell by construction).
- Refusals, all named and writing nothing: no suitable
  adapter (the enumerated list is printed), reps below
  the 20-repetition verdict floor (judgment call: the
  sweep only warns on a smoke CSV, but a profile
  outlives its run and steers future routing, so a
  smoke does not mint one), no resolvable output
  location, and write failure.

## Gates (this VM)

- cargo test -p wgpu-compute: 277 passed / 0 failed
  (267 at base + 10 new: nine in profile.rs covering
  text and file round-trips, absent vs rejected,
  key-mismatch, malformed-with-path, version-mismatch,
  path resolution order, the win rule, and band
  derivation; one in compute.rs covering the three
  precedence layers for bands and ceiling, including
  env-beats-profile, profile-opens-a-shipped-closed
  batch, unparsable-profile-cell fall-through, and the
  closed ceiling fallback). The one ignored test is
  the pre-existing timestamps measurement.
- cargo test -p dflash: 24 passed. cargo test
  -p itak-spec: 49 passed.
- clippy -p wgpu-compute --all-targets -D warnings:
  clean. cargo fmt --check: clean. Workspace check:
  clean.
- The example's two refusal paths were executed on
  this adapterless VM: no-adapter refusal (exit 1,
  enumerated list, nothing written) and the
  below-floor reps refusal (exit 1, nothing written).
  The measurement arms cannot execute here; this VM
  enumerates no GPU adapter at all, by design of the
  test environment.

## What is NOT built yet

- No automatic first-run calibration: nothing
  triggers `calibrate_profile` from session or backend
  construction. Calibration is a deliberate run, not a
  surprise 99-cell sweep inside someone's first
  forward.
- The profile carries only the MMV ceiling and band
  cells. The flash-staged switch, the K21F head
  switch, K18/chunked ceilings, and the wide-MMQ
  threshold are not profile knobs yet.
- No per-cell provenance inside the profile beyond
  the key (no measured medians stored): the profile
  is the derived table, not the raw sweep. The raw
  numbers live in the sweep CSVs when a derivation
  needs auditing.
- No card run yet: the profile path has never been
  exercised against a real adapter end to end.

## Follow-up owed: the card session

On ubuntu-server (7900 XTX), under the standing card
rules (tenants stopped and restored, phase lock,
failure bundles):

1. Run `calibrate_profile` (default reps) writing to a
   scratch `--out`; verify it completes both arms and
   prints a derived table.
2. Compare the derived table against the shipped
   table cell by cell. It should match the shipped
   table's provenance (the 2026-11-06 sweep on this
   same card and driver) within measurement noise;
   every differing cell gets its two arms' medians
   examined before anyone believes either table.
3. Re-run with the profile at the standard location
   (or `WGPU_COMPUTE_PROFILE_PATH` pointed at the
   scratch file) and run a spec_generate_gpu defaults
   run: the capture manifest must state the profile
   key under `mmv_profile`, the resolved bands must be
   the profile's, and identity + golden must pass.
4. Run the routing test file's arms on the card with a
   profile present, so the profile layer is exercised
   under the device-gated tests as well.
