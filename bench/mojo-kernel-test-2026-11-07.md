# Mojo kernel test: K1 Q4_K GEMV ported to Mojo and raced (2026-11-07)

## Verdict, against the pre-set rule

**NOT READY for a second kernel on this card today.** The pre-set
rule (written before the race, not moved): Mojo is ready only if
its correctness sits inside the shipped K1's tolerance (cosine
1.000000000 to the harness's printed precision, max rel err no
worse than 10x the shipped record's 1.267e-7, i.e. 1.267e-6)
AND its interleaved median speed is within 10 percent of the
wgpu arm's interleaved median in the same window, or beats it.

- Correctness leg: **PASS, with margin.** Mojo vs the CPU
  reference: cosine 1.000000000, rel 8.941e-08 (tighter than the
  shipped kernel's own 1.267e-7). Mojo vs wgpu directly: cosine
  1.000000000, rel 1.788e-07. Identical across all three reps.
- Speed leg: **FAIL, by a wide margin.** Interleaved medians:
  wgpu 838.4 GB/s, Mojo 39.8 GB/s. The within-10-percent bar was
  754.6 GB/s; Mojo reached 4.7 percent of the wgpu median. The
  wgpu arm was 21.1x faster.

Feasibility itself passed cleanly: a Mojo toolchain with MAX GPU
support exists today, it lists this card as a supported target,
and a checked Mojo GPU kernel ran on the RX 7900 XTX with an
exact result. What failed is not aim; it is speed of a first,
straightforward port against a hand-tuned kernel.

## Step 1: feasibility (the cheap gate)

Toolchain state found: no `mojo`, `max`, `pixi`, or `modular`
binary on this VM or on ubuntu-server. Prior research on file
(~/workspace/research_notes/mojo-critic-2026-10-28/REPORT.md):
Mojo 1.2.0 (2026-10-22), deps via pixi, channels
`https://conda.modular.com/max` plus conda-forge.

Modular's current hardware compatibility page lists RDNA3
`gfx1100` (Radeon RX 7900) as a supported arch target for Mojo
GPU programming, via the AMD HIP path (`libamdhip64.so`, AMD
driver 6.3.3 or later). The server carries
`/usr/lib/x86_64-linux-gnu/libamdhip64.so.7` system-wide.

Install, self-contained on the server under `~/mojo-probe/`
(nothing in the repo tree): pixi 0.84.1 with
`PIXI_HOME=~/mojo-probe/pixi` (the pixi installer appended its
PATH line to the server's `~/.bashrc`, its standard behavior),
project `~/mojo-probe/probe` resolving **Mojo 1.2.0 (2016d0cd)**
and **MAX 26.7** from the Modular channel; environment size
1.4 GB.

Probe: a throwaway Mojo vector add over 1,000,003 elements
(odd size, bounds-checked on the host afterward) under the
phase lock, tenants up (a probe is not a measurement; 24 GB
card, probe footprint is megabytes). Result:

```
PROBE: DeviceContext created
PROBE_RESULT mismatches = 0 of 1000003
```

Outcome (a): it runs on the card and the result checks out.
Footnote: the probe process printed `Warning: Resource leak
detected by SharedSignalPool, 77 Signals leaked.` on exit, a
runtime housekeeping warning with no effect on the result.

## Step 2: the port

`bench/mojo/k1_gemv.mojo` (this lane): a standalone Mojo program
running the same Q4_K GEMV math as the shipped K1 (the r2sg
shape's math, ported from the repo's own sources, not from
memory: `gguf::dequant::dequant_q4_k` and `scale_min_k4` for the
super-block dequant, the same 144-byte super-block layout, low
nibbles first then high). Workgroup shape is the port's own
choice: one 256-thread block per output row, each thread summing
26 strided columns, shared-memory tree reduction. Exact f16 to
f32 conversion (normals and subnormals both exact).

Fixture, from the repo's own harness so the comparison uses the
SAME data and the SAME bars: `k1_check`'s tensor
`blk.0.ffn_gate.weight` ([6656 x 19968], 74,760,192 bytes), its
seed-42 LCG input, and its CPU reference path (gguf dequantizer
plus torch-math matvec). A new bench example,
`crates/wgpu-compute/examples/k1_fixture_dump.rs`, writes the
raw Q4_K bytes, the input, the CPU reference output, and the
wgpu output into a fixture directory, printing k1_check's own
comparison line as provenance. No library code was touched and
no shipped crate depends on Mojo.

Debug law, from birth: `MOJO_KERNEL_DEBUG` (any value other
than 0/empty) enables named `[k1_gemv_mojo][phase]` log lines;
bad usage, bad shapes, and fixture size mismatches produce
named `REFUSAL k1_gemv_mojo: ...` lines and exit code 2.
Tooling: `mojo format` run on the final source (it reformatted;
the formatted file was rebuilt and re-proven against the
fixture), final `mojo build` warning-free. `compare.py`
(this lane) scores outputs with k1_check's exact definitions.
The runner is `bench/mojo/run_k1_race.sh` (arm selection via
`K1_BENCH=wgpu|mojo|both`).

## Step 3: the race

One cleared-card session on ubuntu-server, phase lock held,
tenants stopped (cortex-embed.service, cortex-cortex-chat-1,
cortex-cortex-embed-1) and restored after. VRAM: 1,680,437,248
bytes in use at lock acquisition (as observed), **33,271,808
after the clear (the exact drained figure)**, **8,932,147,200
after restore (the exact standing figure)**, embed service
active and both containers healthy at closeout. Production
Glimmer never started, port 8080 untouched, no reboot.

Correctness first, as observed:

| comparison | max rel err | cosine |
|---|---|---|
| wgpu vs CPU ref (fixture dump) | 1.267e-07 | 1.000000000 |
| wgpu vs CPU ref (k1_check anchor) | 1.267e-07 | 1.000000000 |
| Mojo vs CPU ref | 8.941e-08 | 1.000000000 |
| Mojo vs wgpu (direct) | 1.788e-07 | 1.000000000 |

The wgpu arm reproduces the shipped 2026-10-30 record
(1.267e-7, cosine 1.000000000) exactly on this fixture.

Speed, interleaved in one window, 200 iterations plus 20
warmup per rep, each iteration streaming the same 74,760,192
bytes (one resident tensor both arms; the wgpu arm is
`k1_bench --budget-mib 72`, whose greedy pack under that budget
is exactly one 74,760,192-byte tensor):

| rep | wgpu GB/s | Mojo GB/s |
|---|---|---|
| 1 | 864.0 | 39.78 |
| 2 | 838.4 | 40.14 |
| 3 | 831.3 | 39.77 |
| median | **838.4** | **39.8** |

Compile and startup costs, stated separately and excluded from
both timed loops (both loops run after warmup with data already
resident): Mojo clean-rebuild wall 1.10 s (`pixi run mojo
build`, compiled binary about 0.16 MB); the Rust bench examples
built once in 38.83 s for all three. Per-process startup
(DeviceContext creation, fixture upload) sits in the untimed
correctness section of each Mojo run, as does pipeline setup on
the wgpu side.

## Why the gap, and what a follow-up costs

Labeled analysis, not measurement: the port recomputes the f16
decode and the scale unpack for every element, loads weights
one byte at a time, and reduces through shared memory; the
shipped r2sg kernel's speed comes from packed scale extraction,
shared activation quads, subgroup butterfly reduction, and
2-rows-per-workgroup streaming. Nothing in this run shows MAX
cannot express those shapes; it shows a first port does not get
them for free. Closing a 21x gap is a tuning project, not a
tweak: mirror the r2sg structure inside the Mojo kernel
(vectorized loads, hoisted per-group dequant, subgroup
reduction), re-prove correctness, re-race. Realistic cost is
days of kernel work for K1 alone, with no guarantee the gap
closes inside the 10 percent bar; the cheap decision point is
one tuned attempt on K1 before any talk of a second kernel.

## Files and state

- Lane bench code: `bench/mojo/k1_gemv.mojo`,
  `bench/mojo/run_k1_race.sh`, `bench/mojo/compare.py`,
  `crates/wgpu-compute/examples/k1_fixture_dump.rs`
  (rustfmt clean, `cargo check` clean on the VM, release build
  clean on the server; the formatted example re-ran the fixture
  dump and reproduced the PASS line exactly).
- Server scratch, kept out of the repo: toolchain and probe
  project in `~/mojo-probe/`, staging tree and fixture in
  `~/gpu-mojo-test/` (session log `~/mojo-probe/race-session.log`).
- Nothing merged, nothing pushed, no engine flag, no default
  changed.
