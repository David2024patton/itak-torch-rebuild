# Diag env wiring: the Category H HIGH (2026-11-07)

Lane `lane/diag-env-wiring`, from main
012396ad8ce7bf4273f58b4c1183617dbb984ba3. Code and tests in the
first commit on this lane; this note and the RUNBOOK table are
the second commit. All runs executed on this VM against this
lane's tree; no server jobs.

Source of the work: the cold critic panel of 2026-11-07
(`CRITIC-ENGINE-MERGE.md`, Category H; evidence executed by the
operability seat). The build law says every module ships
debugging from birth: a structured diagnostic channel, a way to
turn it on, and a test proving it fires. Linting was clean
everywhere; the debugging half was decoration for six crates:

- attention, cpu-compute, wgpu-compute, torch-math and itak-kv
  exposed only a programmatic `set_debug` / `set_debug_enabled`
  with no env var and no caller anywhere in the workspace, so in
  the shipped engine binary those channels were permanently off.
- tokenizer's seam was the `debug-log` cargo feature; no
  dependent enabled it, so the shipped tokenizer compiled its
  diag macro to nothing.
- Debug self-tests existed only in gguf, itak-loader and
  itak-engine: 10 of 13 seams had none.

## The fix, per crate

Env twins in the gguf shape. gguf's `debug_enabled()` checks the
explicit flag first, then the env var on every call (present with
any value other than `0` or empty enables). Each flag-based crate
now does exactly that, so any binary linking the crate (the
engine included) can enable its seam with no code change and no
rebuild. No engine startup wiring was needed: per-call
consultation is the mechanism, the same one gguf and itak-loader
already ship.

| Crate | Env var | Setter (unchanged) |
| :--- | :--- | :--- |
| attention | `ATTENTION_DEBUG` | `set_debug_enabled` |
| cpu-compute | `CPU_COMPUTE_DEBUG` | `set_debug` |
| wgpu-compute | `WGPU_COMPUTE_DEBUG` | `set_debug` |
| torch-math | `TORCH_MATH_DEBUG` | `set_debug` |
| itak-kv | `ITAK_KV_DEBUG` | `set_debug_enabled` |
| tokenizer | `TOKENIZER_DEBUG` | `set_debug` (new) |

Names are the crate name uppercased plus `_DEBUG`, which is
exactly the convention the two already-wired crates follow
(`GGUF_DEBUG`, `ITAK_LOADER_DEBUG`). wgpu-compute's var is
deliberately distinct from `WGPU_COMPUTE_DIAG`, the per-dispatch
timestamp system in `timestamps.rs`; the diag.rs header says so,
and that system is untouched.

Emission shape: each macro crate's lines now flow through one
`emit_debug` path (the macro still checks `debug_enabled()` first
and evaluates no format arguments while off). In test builds
`emit_debug` also captures the exact formatted line into a
test-only buffer, the gguf channel's pattern, which is what makes
the capture tests below possible. No call site changed in any
crate.

tokenizer: converted to the runtime flag. The `debug-log`
feature is removed from its manifest (nothing enabled it; the
workspace builds clean without it), `dbglog!` now checks
`diag::debug_enabled()` at runtime, and its `diag` module became
`pub` like every sibling crate's (it was the only private one,
which is part of why the seam was unreachable). One test drives a
real production refusal (an unsupported tokenizer family) and
asserts the refusal line lands on the channel.

Trait seams (arch-llama, dflash, itak-spec, muse-glimmer) needed
no wiring change: a caller passes the sink, and the engine
already wires muse-glimmer's. What they lacked was the self-test;
each gains a capturing-sink test in the itak-engine shape.

## Before / after evidence

Failing before, executed: before the fix, each flag crate's new
`env_twin_enables_debug` test was added alone and run against the
unmodified seam. All five FAILED at the "must enable" assertion
(attention diag.rs:52, cpu-compute diag.rs:49, wgpu-compute
diag.rs:49, itak-kv diag.rs:52, torch-math lib.rs:210): the env
var was simply never read. The seat's grep evidence stands as the
before-proof for tokenizer (feature off in every dependent) and
for the missing self-tests.

Passing after, 17 new tests, all green:

- `env_twin_enables_debug` (attention, cpu-compute, wgpu-compute,
  itak-kv, torch-math `diag_tests`, tokenizer): var `1` enables,
  `0` and empty do not, removal restores off.
- `seam_fires_when_on_and_is_silent_when_off` (same six crates):
  a probe line through the crate's own macro is captured with its
  prefix while on, and produces nothing while off.
- `a_real_refusal_emits_through_the_seam` (tokenizer): the
  unsupported-family refusal path emits `refusing unsupported
  tokenizer model` while debug is on.
- `seam_delivers_events_to_a_sink` (arch-llama, dflash,
  itak-spec, muse-glimmer): a capturing sink receives the event
  verbatim; NullDiag drops silently; StderrDiag writes without
  panicking.

Diag tests within a crate hold a per-crate lock: the flag and the
env var are process-global, so parallel flips could otherwise
satisfy each other's assertions. Capture assertions search for
per-test unique probe strings, per the gguf buffer discipline.

## Gates, as observed on the lane tree

- `cargo test --workspace`: 781 passed, 0 failed, 1 ignored
  (68 suites). The 17 tests above are this lane's addition.
  Lib suites for the touched crates: arch-llama 7, attention 30,
  cpu-compute 11, dflash 25, itak-kv 23, itak-spec 70,
  muse-glimmer 16, tokenizer 21, torch-math 46, wgpu-compute 199.
- `cargo clippy --workspace --all-targets -- -D warnings`: clean.
- `cargo fmt --all -- --check`: clean (the fmt pass over the new
  test code is folded into the code commit).
- `cargo check --workspace`: clean.
- Platform (shared tree stayed whole): `go build ./...`,
  `go vet ./...` clean; `go test ./...` 115 tests, all packages ok.

Every default is still OFF and the off path is unchanged: with no
env set, `debug_enabled()` returns the flag exactly as before and
the macros evaluate nothing. Existing tests passed unmodified;
the only test changes are the additions above.

## RUNBOOK outcome

Included in this commit: a Diagnostics switches table at the end
of RUNBOOK's "Run locally" section (kept inside the file's
declared structure rather than as a new section) listing each
switch, its owner module, its default, and its effect, including
the pre-existing `WGPU_COMPUTE_DIAG`, `ITAK_KV_PREFIX_*`, and
`TORCH_LOG_LEVEL` switches the operability seat found
undocumented. That closes the Category H RUNBOOK MEDIUM.

## Deliberately left

- The Category H platform MEDIUM (slog coverage in the Go
  platform's auth, cache, chattemplate, engine and scheduler
  modules, and the platform README sentence) is out of this
  lane's scope and remains open.
- Supervised reach: a torchd-supervised engine child receives an
  allowlisted environment (PATH plus `ITAK_ENGINE_*` only,
  `platform/internal/supervisor/supervisor.go` `childEnv`), so
  the `*_DEBUG` switches reach a hand-run engine today but are
  filtered out before a supervised child. The RUNBOOK table
  states this plainly. Extending the allowlist (or adding an
  `ITAK_ENGINE_*` startup resolution in the engine) is follow-up
  work in the platform, not silently folded into this lane.
- The `WGPU_COMPUTE_DIAG` timestamps system and the examples'
  programmatic `set_debug` calls are unchanged; the env twin
  makes the second unnecessary for shipped binaries, which was
  the panel's point.
