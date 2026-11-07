# Mojo/MAX engine foundation, 2026-11-07

Branch `mojo-max-engine`, from main `b416987`. David's order:
pause Rust and Go, start over with MAX and Mojo. This lane is
the FOUNDATION ONLY: the branch, the skeleton, one kernel made
a citizen, and the first proof in the new layout. The Rust and
Go trees are untouched; every new file lives under
`mojo-engine/` plus this note.

Commits: `e5c2ab3` skeleton + design, `eab67c3` K1 kernel
module, `1dc2942` warning cleanup at the kernel's log call
sites, this note.

## What was built, file by file (all under mojo-engine/)

- `pixi.toml` + `pixi.lock`: the toolchain manifest, channels
  conda.modular.com/max + conda-forge, pinned to the versions
  proved by the kernel test: Mojo 1.2.0 (2016d0cd), MAX 26.7.
  Tasks: `engine`, `test`, `k1-fixture`.
- `DESIGN.md`: the stack (optional Go or MAX Serve front end;
  Mojo host; Mojo kernels, MAX's kernel library where it fits
  and hand-tuned custom kernels where it does not; MAX as the
  plumbing; vendor drivers at the bottom), the piece-by-piece
  build order (loader, tuned dequant+GEMV, attention, layers,
  generation loop, sampler, serving seam), the licensing facts
  as constraints, the parity law, and the speed law with the
  kernel test's numbers (wgpu K1 median 838.4 GB/s vs the
  straight Mojo port's 39.8; correctness rel 8.941e-08, cosine
  1.000000000).
- `README.md`: setup, run, test, layout, and the explicit
  not-built list.
- `src/main.mojo`: entry point. Prints `mojo-engine 1.1`
  (the new engine line starts at 1.1 in the repo's 1.<n>
  scheme), loads config, refuses cleanly with the config's
  named error and exit 2 on bad config.
- `src/config.mojo`: config as defaults in one place
  (DEFAULT_MODEL_PATH = the real Glimmer GGUF path,
  DEFAULT_CONTEXT_LEN = 131072), env overrides
  MOJO_ENGINE_MODEL_PATH / MOJO_ENGINE_CONTEXT_LEN, and
  `parse_context_len` refusing by name (`mojo_engine_config:
  ...`) on non-numeric or non-positive values.
- `src/diag.mojo`: the debug seam in the Rust tree's shape.
  MOJO_ENGINE_DEBUG is consulted on every call; any present
  value except "" and "0" enables. Lines are named
  `[mojo_engine][<phase>] ...`. `log_debug` returns the emitted
  line or "" so the seam is self-testable.
- `src/gguf/loader.mojo` (+ `__init__.mojo`): the loader's
  interface, fixed now: `loader_state()` returns
  "not_implemented"; `load_model` raises
  `gguf_loader_not_implemented` by name. The real loader is
  the next lane; nothing here pretends to load.
- `src/kernels/k1_gemv.mojo` (+ `__init__.mojo`): the kernel
  test's K1 Q4_K GEMV port carried over and made a citizen:
  debug flow through `src/diag.mojo` (MOJO_ENGINE_DEBUG, no
  ad hoc per-kernel flag), named REFUSAL + exit 2 on bad
  shapes, standalone fixture entry kept, and a new
  `k1_gemv_cpu` twin running the identical math on the host
  for GPU-less fixture tests. NOT tuned, deliberately; the
  tuning lane is later and DESIGN.md says why.
- `tests/`: `test_diag.mojo` (3), `test_config.mojo` (3),
  `test_k1_cpu.mojo` (2), `test_loader.mojo` (2),
  `config_probe.mojo` (process-level probe for the runner),
  `run_tests.sh` (the gate: TestSuite files in a clean env,
  the diag suite again with the seam on, config probe at
  defaults / env override / bad value, engine hello).

Toolchain note: Mojo 1.2 has no `mojo test` subcommand; the
toolchain's own runner is `std.testing.TestSuite`, discovered
and run by each test file's `main()` under `mojo run`. The
gate script drives exactly that.

## Toolchain setup, reproducible

On this VM nothing was installed beforehand. pixi 0.84.1 was
placed self-contained at `~/mojo-probe/bin/pixi` (the official
release tarball), then `pixi install` inside `mojo-engine/`
built the env from `pixi.toml`/`pixi.lock`: Mojo 1.2.0
(2016d0cd), MAX 26.7, the same versions the server proved.
Toolchains live outside the repo tree; the repo carries only
the manifest. On the server the kernel test's earlier
self-contained install (`~/mojo-probe/`, PIXI_HOME) was reused.

## Proof, half (a): this VM (no GPU)

`pixi run ./tests/run_tests.sh`: PASS, after `mojo format`
over all sources (clean).

- TestSuite tallies as observed: diag 3/3, config 3/3,
  k1_cpu 2/2, loader 2/2 in the clean-env pass (10 tests);
  diag 3/3 again with MOJO_ENGINE_DEBUG=1, its seam-on pass
  printing `[mojo_engine][test] seam probe` as designed.
- Config: defaults load (context_len 131072, default model
  path), env override lands (context_len 4096, model path),
  bad value `abc` refused as `REFUSAL mojo_engine_config:
  MOJO_ENGINE_CONTEXT_LEN is not a whole number: 'abc'`
  with exit 2 from the probe.
- Engine hello: `mojo-engine 1.1` / `config ok: context_len
  131072` / loader line naming the next lane.
- K1 CPU fixture: SYNTHETIC Q4_K data, named as such
  (deterministic LCG bytes, rows=8, cols=512, tame f16 scale
  headers), reference computed inside the test by a
  structurally different route (materialized super-block
  dequant, Float64 dot). Observed: cosine
  0.9999999999999999, max_rel 0.0 against that reference
  (bars were cosine >= 0.999999999, max_rel <= 1e-5).
  Plus the bad-shape refusal test (`k1_gemv_cpu_bad_shape`).
  The VM used the synthetic fixture because the real fixture
  lives on the server; the real fixture is half (b).

## Proof, half (b): the server, RX 7900 XTX

Staging: `~/gpu-mojo-engine` from `git archive` of the branch
at `eab67c3` (the kernel commit; the later `1dc2942` touches
only `_ =` prefixes at log call sites, no math), commit
stamped in `STAGED_COMMIT`, staged
`mojo-engine/src/kernels/k1_gemv.mojo` sha256 identical to
the worktree file (20cc3294...). Fixture: the kernel test's
own `~/gpu-mojo-test/fixture` (real Glimmer
blk.0.ffn_gate.weight [6656 x 19968], 74,760,192 bytes,
seed-42 input, recorded CPU reference in `ref_f32.bin`).

House rules held: phase lock throughout; tenants stopped
(cortex-embed.service, cortex-cortex-chat-1,
cortex-cortex-embed-1) and restored; VRAM proven at both
bookends (a short re-verification cycle after the main run,
because the first run's middle lines scrolled past the
captured tail): drained exactly 33,271,808 B, restored
exactly 8,932,147,200 B, cortex-embed.service active, both
containers Up and healthy. Port 8080: 0 listeners
throughout. Production Glimmer never started. No reboot.

- Engine hello on the server: ran (same three lines as VM).
- The kernel module, from the NEW branch layout
  (`mojo-engine/src/kernels/k1_gemv.mojo`), GPU path through
  MAX, debug seam on: named lines
  `[mojo_engine][k1_gemv.load]`, `[k1_gemv.device]`,
  `[k1_gemv.run]`, `[k1_gemv.timing]` fired in order.
- Correctness vs the recorded reference, judged by the
  kernel test's own compare.py: foundation output vs ref
  **rel 8.941e-08, cosine 1.000000000** (identical to the
  kernel test's Mojo arm); vs the wgpu output rel 1.788e-07,
  cosine 1.000000000; the fixture's gpu vs ref anchor
  reproduced at rel 1.267e-07, cosine 1.000000000.
- One timing line, recorded with no speed verdict in this
  lane (the kernel test's bar stands as the known starting
  point): 10 iters + 3 warmup, 747,601,920 bytes in
  0.019074 s = **39.19 GB/s**, in line with the kernel
  test's 39.8 GB/s median at 200 iters.

## Deliberately not built

The GGUF loader (interface only), every kernel past K1,
K1 tuning, attention, the generation loop, the sampler, any
serving front end, and any change to the Rust/Go trees.

## Next lane

The GGUF loader (DESIGN.md step 1): parse the real Glimmer
file in Mojo, metadata and tensor directory first, proven
against the Rust loader's recorded facts about the same file
(731 tensors, the known shapes and offsets), behind the
interface fixed in `src/gguf/loader.mojo`.
