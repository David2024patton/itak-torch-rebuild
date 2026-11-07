# Hygiene bundle (2026-11-07)

One lane, four small recorded follow-ups, each its own
clearly separated change in one code commit. Base:
main 0ecdc8f. Lane: lane/hygiene-bundle, worktree
~/workspace/lanes/hygiene-bundle. VM only; no server
GPU jobs. Item 2 is a verified skip (verdict below).

## Item 1: diag manifest flash_staged pair

Gap, recorded in bench/defaults-confirmation-
2026-11-07.md leg 2: the staged-flash resolution was
observable nowhere in the diag manifest or any
shipped log line, because the construction log line
rides the `dbg_log!` seam, which has no enabling
caller in any shipped harness.

Change: both sessions' `capture_manifest`
resolved_config now stamps `flash_staged` with the
resolved bool (`backend.flash_staged().to_string()`),
immediately after the `k21f_head` pair, beside the
existing switch pairs (mmv_profile, k22_mmq,
mmq_wide_min_rows, k21f_head):

- crates/wgpu-compute/src/session.rs (resident)
- crates/wgpu-compute/src/dflash_session.rs (draft)

The flash resolver's semantics are untouched
(unset on, "1" on, "0" off, anything else off
fail-closed) and no routing changed; only the
manifest gained the field. Any capture-mode run on
the merged tree now records the flash resolution it
actually ran with, which is the evidence leg 2 of
the defaults-confirmation session had to go without.

Pin: the manifest-shape sample in
crates/wgpu-compute/src/timestamps.rs
(`manifest_renders_and_its_fields_diff`) now carries
a `flash_staged` resolved_config pair mirroring the
sessions' shape, with a lookup assertion
(`resolved_config.flash_staged` = "true") and a
comment stating why. No stronger host-side pin
exists for the sessions' own pair lists:
`capture_manifest` is private and needs a live
backend, so like the routing fixtures its end-to-end
proof is the next capture-mode card run's manifest.
The resolver itself stays pinned by the existing
compute.rs switch-resolution tests, unchanged.

## Item 2: stale K21F kernel doc: SKIPPED, already accurate

Checked crates/wgpu-compute/src/kernels/
k21_q5k_mmv_finish.rs's module header on this base.
The stale claim the item names (that no session code
prepares the pipeline yet) is not present; the guard
and flip lanes already rewrote the routing
paragraph. What it says now, in substance: routing
is not in the module; the session's block-head site
prepares the pipeline through `try_mmv_q5k_finish`
(beside `try_mmv_q5k`), falling back to K3 on any
refusal; the route is governed by two gates, the
backend's `WGPU_COMPUTE_K21F_HEAD` switch ("on when
unset or `1`, off on `0` or any unrecognized value;
default off under the guard written after the
2026-11-07 defaults-confirmation identity failure,
see bench/k21f-default-guard-2026-11-07.md, and back
to on by default since the wave18 retest met the
guard's reopen criterion, see
bench/k21f-retest-2026-11-07.md and
bench/k21f-default-flip-2026-11-07.md") and the
shared `WGPU_COMPUTE_MMV_MAX_ROWS` band admitting
the block batch; the block head's band is 2..=16
and the standard head (batch 1) stays K3. That is
the current truth (default ON since the flip, main
fa32997; batch floor 2), stated with its history.
No edit made.

## Item 3: gated-note prediction correction

Appended a correction section to
bench/gated-swiglu-routing-2026-11-06.md (its
history is untouched). Source measurement:
bench/baseline-session-2026-11-06.md leg (f).

The miss: the server protocol predicted depth 1
would add +52 dispatches per verify forward (52
swiglu_strided) because the down site routes at
rows 2; measured is 26.0 swiglu_strided per forward
and a +26 net delta, at depth 1 and in the depth-6
rows-3 and rows-5 forwards. The depth-6 rows-7
prediction (+104: 52 gate + 52 swiglu) was exact.

The rule, from the session code: the decomposition
fires per layer, admitted by that layer's own
down-tensor family. `prepare_swiglu_proj` matches
on the down weights: the Q4_K arm consults the
routed ladder for MmvFamily::Q4K, the Q6_K arm for
MmvFamily::Q6K; the down shape (in 19,968, out
6,656) classifies as attn, so Q6_K admits batches
2..=12 and Q4_K admits 6..=12. The note's census
line took the down family from the DFlash file's
census (bench/dflash-e3-2026-10-30.md), where every
layer's down is Q6_K; the target file's downs are
mixed, 26 layers Q6_K and 26 layers Q4_K, pinned
arithmetically from the defaults-confirmation
prefill summaries (bench/results/
defaults-confirmation-20261107/leg2-default.csv and
leg2-off.csv: 182 k13 per 64-row chunk = 156 base +
26 Q4_K downs; 52 k14 per chunk = 26 Q6_K downs +
26 Q6_K V's). Corrected per-forward arithmetic:
batch 2..=5: 26 swiglu_strided, 0 gate_strided,
+26 net; batch 6..=7: 52 + 52, +104 net. No timing
claim in the note changes (the depth-6 estimate was
computed at rows 7; no figure was promised at
depth 1).

## Item 4: DflashMetadata bool-array probe through opt_*

The loader half of the metadata fix (main 88cb83a)
routed GgufMetadata's scalars through gguf's `opt_*`
reads so an absent optional key stays silent, but
DflashMetadata::get_bool_array still probed through
the strict `get_array`, whose refusal is built by
`require` and logs a `missing_metadata` error event
for a merely absent key.

Changes:

- crates/gguf/src/metadata.rs: new `opt_array`, the
  array sibling of `opt_string`/`opt_u32`/`opt_f32`.
  Absent: Ok(None), no refusal constructed. Present
  array: Ok(Some(items)). Present non-array:
  Err through the shared `mismatch` seam, so the
  refusal and its diag event fire exactly as the
  strict getter's do.
- crates/itak-loader/src/adapter.rs:
  `DflashMetadata::get_bool_array` now routes
  through `opt_array`. Absence answers None
  silently; a wrong-typed value refuses inside the
  opt read (logged there, noted by the adapter's
  `note_unreadable`), and the dflash config converts
  None into its own `MissingMetadata` refusal naming
  the key; a non-bool array element keeps its
  existing diag line and None.

Tests:

- gguf (metadata.rs test module): `opt_array`
  joins the absent-key silence pin
  (`opt_getters_are_silent_when_the_key_is_absent`,
  asserted against the captured diag errors), plus
  two new tests: `opt_array_round_trips_a_present_
  array` and `opt_array_still_refuses_a_wrong_type_
  loudly` (unique probe keys, mismatch event
  asserted from the captured errors).
- itak-loader (tests/wiring.rs): a complete tiny
  DFlash GGUF fixture (every key
  `DflashConfig::from_metadata` requires, toy
  sizes, written by the file's existing tiny GGUF
  writer) and three tests:
  `dflash_config_loads_end_to_end_from_gguf_bytes`
  (fixture sanity; sliding pattern round-trips),
  `absent_dflash_bool_array_key_probes_none_quietly`
  (adapter probe of an absent key answers None;
  absence through `load_dflash_config` surfaces as
  the config's MissingMetadata refusal naming
  `dflash.attention.sliding_window_pattern`), and
  `wrong_typed_dflash_pattern_key_refuses_by_name`
  (a String in the pattern key refuses through
  `load_dflash_config`, and the refusal names the
  key in its variant and its message).

## Gates (this lane, VM)

- `cargo test -p wgpu-compute`: 277 passed, 0
  failed (GPU legs skip on this VM by design).
- `cargo test -p gguf`: 49 passed, 0 failed.
- `cargo test -p itak-loader`: 18 passed, 0 failed
  (wiring suite 13 -> 16 with the DFlash tests).
- `cargo test -p dflash`: 24 passed, 0 failed.
- `cargo test -p itak-spec`: 49 passed, 0 failed.
- `cargo clippy -p gguf -p itak-loader
  -p wgpu-compute --all-targets -- -D warnings`:
  clean.
- `cargo fmt --all -- --check`: clean.
- `cargo check --workspace --all-targets`: clean.

One environment note: the lane worktree's own
target directory hit the VM's disk quota partway
through the first build (the accumulated per-lane
target caches), so the gate runs above executed
with CARGO_TARGET_DIR pointed at the merged
auto-tune lane's warm cache. Build cache only; no
source tree other than this lane's was touched.
