# mojo-engine

The from-scratch Mojo/MAX inference engine, started 2026-11-07
on the `mojo-max-engine` branch. It pauses the Rust engine and
the Go platform (both stay on `main`, proven, as the reference
this engine must match) and rebuilds the engine piece by piece
in Mojo, on the MAX runtime. Design, build order, and the
parity and speed laws: [DESIGN.md](DESIGN.md).

## Setup

The toolchain is pinned by `pixi.toml` (Mojo 1.2.0, MAX 26.7,
the exact versions proved on the RX 7900 XTX by
`bench/mojo-kernel-test-2026-11-07.md`):

```sh
cd mojo-engine
pixi install
```

## Run

```sh
pixi run engine        # hello + config load (foundation scope)
```

Configuration is env over defaults in `src/config.mojo`:
`MOJO_ENGINE_MODEL_PATH`, `MOJO_ENGINE_CONTEXT_LEN`, and the
debug seam `MOJO_ENGINE_DEBUG` (any value but "" / "0" enables;
see `src/diag.mojo`).

## Test

```sh
pixi run test          # tests/run_tests.sh: mojo test, both seam modes
```

The suite covers config (defaults, env override, named
refusals), the diag seam (on fires, off silent, "0" does not
enable), and a CPU fixture test of the K1 kernel module on
synthetic Q4_K data with its own in-test reference. The real
Glimmer fixture runs on the server through the GPU path; see
`bench/mojo-engine-foundation-2026-11-07.md` for the foundation
proof and which fixture each half used.

## Layout

- `src/main.mojo` - entry point (hello, config, clean refusals)
- `src/config.mojo` - config: env over defaults, named refusals
- `src/diag.mojo` - the debug seam (MOJO_ENGINE_DEBUG)
- `src/gguf/loader.mojo` - loader interface; named
  not-implemented refusal until the loader lane lands
- `src/kernels/k1_gemv.mojo` - K1 Q4_K GEMV (GPU kernel + CPU
  twin), carried over from the kernel test, untuned by design
- `tests/` - the test suite and its runner

## Deliberately not built yet

The GGUF loader (next lane), every kernel past K1, attention,
the generation loop, the sampler, and any serving front end.
K1 is present as a correctness citizen, not a speed result:
its straight-port numbers and the tuning debt are recorded in
DESIGN.md under the speed law.
