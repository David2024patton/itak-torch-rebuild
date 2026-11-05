#!/bin/bash
# Shader autopsy: read-only diagnosis of the compiled form of
# every kernel family in the resident GPU path, plus the run
# environment, captured before any new kernel is written.
#
# Why: the streaming-rate research (bench plan in
# ~/workspace/research/gpu-streaming-rate-2026-11-04) found the
# shared-weight batched kernels streaming at 70-130 GB/s while
# the per-row GEMV reader reaches several hundred GB/s on the
# same card. Step 1 of that plan is to look at what the driver
# compiler (ACO, via RADV) actually emitted for each family:
# register counts, scratch (spills), LDS per workgroup, and
# load widths in the disassembly, with GEMV (fast) compared
# against K18 (slow). The Mesa issue 15132 pattern (redundant
# buffer loads from aliased multi-view access) is the named
# suspect; the stats and asm dumps are how it gets caught.
#
# What it does:
#   1. Captures the environment (GPU, Mesa/Vulkan driver,
#      toolchain, git SHA, flags) into env.txt.
#   2. Builds the verdict harness (itak-spec's
#      spec_generate_gpu example) and the mixed-fixture
#      resident_session test.
#   3. Dump arm: runs the verdict harness on the real Glimmer
#      files with RADV_DEBUG=shaderstats,asm,cs (compute stage
#      only) and a private MESA_SHADER_CACHE_DIR, so every
#      shader compiles cold and dumps. The opt-in kernel
#      families are switched ON for this arm only
#      (WGPU_COMPUTE_SMALL_BATCH_MAX_ROWS=16 for K18,
#      WGPU_COMPUTE_MMQ_WIDE_MIN_ROWS=32 for K15/K16) so all
#      families appear in one dump set. Also runs the
#      resident_session mixed fixture under dumps at SHIPPED
#      defaults: that test pins the exact dispatch count of
#      the shipped ladder (resident_session.rs), so running
#      it with K18 on trips the count pin by design (318 vs
#      306, observed 2026-11-04) after every token check has
#      already passed.
#   3b. Family isolation: runs wgpu-compute's shader_compile
#      tests one family at a time (K1, K2/K3, K12-K14,
#      K10/K11, K17, K15/K16, K18, K4-K8), each under the
#      same dump env with its own private cache, so every
#      stats block is attributable to exactly one family.
#      RADV's dump headers carry no shader name, and the
#      in-situ LDS values differ from the WGSL staging
#      constants by driver packing, so this isolation is
#      what makes the comparison table exact.
#   4. Perf arms: the same verdict run at shipped defaults
#      (K18 off, wide MMQ off), once plain and once with
#      RADV_PERFTEST=cswave32, each with its own private
#      shader cache. These are single data points, not
#      verdicts; the logs keep the full output either way.
#
# This script changes no engine code and sets no defaults.
# Dumps land on the harness's stderr (RADV writes them
# there); stdout and stderr are kept in separate logs.
#
# Usage:
#   bench/shader-autopsy.sh [MODEL] [DRAFT]
# MODEL/DRAFT default to the recorded Glimmer paths.
# Knobs: AUTOPSY_OUT (default /tmp/shader-autopsy/<date>-<sha>),
# AUTOPSY_NTOK (64), AUTOPSY_DEPTH (6), EXPECTED_SHA (recorded
# in env.txt when the tree has no .git, e.g. an archive stage).
#
# Run on ubuntu-server with the GPU free (the resident
# session takes the card). One arm at a time, sequential.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 2
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}"

MODEL="${1:-/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf}"
DRAFT="${2:-/mnt/models/Muse-Glimmer-30B/dflash-Muse-Glimmer-30B-Q4_K_M.gguf}"
NTOK="${AUTOPSY_NTOK:-64}"
DEPTH="${AUTOPSY_DEPTH:-6}"
SHA="$(git rev-parse HEAD 2>/dev/null || echo "${EXPECTED_SHA:-unknown}")"
STAMP="$(date +%F)-$(echo "$SHA" | cut -c1-8)"
OUT="${AUTOPSY_OUT:-/tmp/shader-autopsy/$STAMP}"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.txt"
: > "$SUMMARY"
note() { echo "$*" | tee -a "$SUMMARY"; }

note "=== shader autopsy $(date -Is) ==="
note "tree: $ROOT"
note "out:  $OUT"

# --- 1. environment -------------------------------------------------
{
  echo "date: $(date -Is)"
  echo "host: $(hostname)"
  echo "kernel: $(uname -srm)"
  echo "git_sha: $SHA"
  echo "expected_sha: ${EXPECTED_SHA:-same-as-git}"
  echo "cargo: $(cargo --version 2>/dev/null)"
  echo "rustc: $(rustc --version 2>/dev/null)"
  echo "--- vulkan devices ---"
  vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverName|driverInfo|apiVersion" \
    || echo "vulkaninfo: not available"
  echo "--- gpu (lspci) ---"
  lspci 2>/dev/null | grep -iE "vga|3d controller|display" || echo "lspci: not available"
  echo "--- mesa package ---"
  dpkg-query -W -f='${Package} ${Version}\n' 'libgl1-mesa-dri' 'mesa-vulkan-drivers' 2>/dev/null \
    || echo "dpkg mesa query: not available"
  echo "--- vram (rocm-smi, if present) ---"
  rocm-smi --showmeminfo vram 2>/dev/null | grep -iE "used|total" || echo "rocm-smi: not available"
  echo "--- run flags ---"
  echo "dump arm:  RADV_DEBUG=shaderstats,asm,cs WGPU_COMPUTE_SMALL_BATCH_MAX_ROWS=16 WGPU_COMPUTE_MMQ_WIDE_MIN_ROWS=32"
  echo "perf arms: shipped defaults; arm B adds RADV_PERFTEST=cswave32"
  echo "each arm:  private MESA_SHADER_CACHE_DIR under $OUT"
  echo "model: $MODEL"
  echo "draft: $DRAFT"
  echo "ntok: $NTOK depth: $DEPTH"
} > "$OUT/env.txt" 2>&1
note "env captured: $OUT/env.txt"

[ -f "$MODEL" ] || { note "SKIP: model not found: $MODEL"; exit 0; }
[ -f "$DRAFT" ] || { note "SKIP: draft not found: $DRAFT"; exit 0; }

# --- 2. build --------------------------------------------------------
note "--- build ---"
cargo build --release -p itak-spec --example spec_generate_gpu 2>&1 | tail -2 | tee -a "$SUMMARY"
GEN="$ROOT/target/release/examples/spec_generate_gpu"
[ -x "$GEN" ] || { note "FAIL: spec_generate_gpu did not build"; exit 1; }
cargo test --release -p wgpu-compute --test resident_session --no-run 2>&1 | tail -2 | tee -a "$SUMMARY"
FIXTURE="$(ls -t "$ROOT"/target/release/deps/resident_session-* 2>/dev/null | grep -v '\.d$' | head -1)"
[ -n "$FIXTURE" ] && [ -x "$FIXTURE" ] || { note "FAIL: resident_session test binary not found"; exit 1; }
note "fixture binary: $FIXTURE"
cargo test --release -p wgpu-compute --test shader_compile --no-run 2>&1 | tail -2 | tee -a "$SUMMARY"
SHADER_COMPILE="$(ls -t "$ROOT"/target/release/deps/shader_compile-* 2>/dev/null | grep -v '\.d$' | head -1)"
[ -n "$SHADER_COMPILE" ] && [ -x "$SHADER_COMPILE" ] || { note "FAIL: shader_compile test binary not found"; exit 1; }
note "shader_compile binary: $SHADER_COMPILE"

# Memory discipline for the model runs on the shared server
# (recorded server rule: arena caps keep the 17 GB load from
# fragmenting the box's tight RAM).
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=0

# --- 3. dump arm -----------------------------------------------------
note "--- dump arm (all kernel families on, RADV_DEBUG=shaderstats,asm,cs) ---"
mkdir -p "$OUT/cache-allon"
RADV_DEBUG=shaderstats,asm,cs \
MESA_SHADER_CACHE_DIR="$OUT/cache-allon" \
WGPU_COMPUTE_SMALL_BATCH_MAX_ROWS=16 \
WGPU_COMPUTE_MMQ_WIDE_MIN_ROWS=32 \
  "$GEN" "$MODEL" "$DRAFT" "$NTOK" "$DEPTH" \
  > "$OUT/allon.stdout.log" 2> "$OUT/allon.stderr.log"
note "verdict run exit: $? (dumps in allon.stderr.log, $(wc -c < "$OUT/allon.stderr.log") bytes)"
grep -E "IDENTITY|GOLDEN|SPEED|T-9.6|tok/s" "$OUT/allon.stdout.log" | tail -6 | tee -a "$SUMMARY" || true

mkdir -p "$OUT/cache-fixture"
RADV_DEBUG=shaderstats,asm,cs \
MESA_SHADER_CACHE_DIR="$OUT/cache-fixture" \
  "$FIXTURE" --nocapture \
  > "$OUT/fixture.stdout.log" 2> "$OUT/fixture.stderr.log"
note "mixed fixture exit: $? (dumps in fixture.stderr.log, $(wc -c < "$OUT/fixture.stderr.log") bytes)"

# --- 3b. family isolation ---------------------------------------------
note "--- family isolation (shader_compile, one family per run) ---"
for fam in k1_shader_compiles k2_k3_shaders_compile k12_k14_mmq_shaders_compile \
           k10_k11_gemm_shaders_compile k17_dflash_shaders_compile \
           k15_k16_mmq_wide_shaders_compile k18_small_batch_shaders_compile \
           k4_to_k8_pipelines_compile; do
  mkdir -p "$OUT/cache-fam-$fam"
  RADV_DEBUG=shaderstats,asm,cs \
  MESA_SHADER_CACHE_DIR="$OUT/cache-fam-$fam" \
    "$SHADER_COMPILE" "$fam" --nocapture \
    > "$OUT/family-$fam.stdout.log" 2> "$OUT/family-$fam.stderr.log"
  note "family $fam exit: $? stats blocks: $(grep -c 'SHADER STATS' "$OUT/family-$fam.stderr.log")"
done

# --- 4. perf arms (single data points) --------------------------------
run_perf() {
  local arm="$1"; shift
  mkdir -p "$OUT/cache-$arm"
  note "--- perf arm: $arm ($*) ---"
  env MESA_SHADER_CACHE_DIR="$OUT/cache-$arm" "$@" \
    "$GEN" "$MODEL" "$DRAFT" "$NTOK" "$DEPTH" \
    > "$OUT/perf-$arm.stdout.log" 2> "$OUT/perf-$arm.stderr.log"
  note "perf $arm exit: $?"
  grep -E "plain stream" "$OUT/perf-$arm.stdout.log" | tail -1 | tee -a "$SUMMARY" || true
  grep -E "IDENTITY|GOLDEN|SPEED|T-9.6" "$OUT/perf-$arm.stdout.log" | tail -4 | tee -a "$SUMMARY" || true
  grep -E "verify|draft|round" "$OUT/perf-$arm.stdout.log" | tail -6 | tee -a "$SUMMARY" || true
}
run_perf control
run_perf cswave32 RADV_PERFTEST=cswave32

# --- 5. stats extract ---------------------------------------------------
# Pull every stats block out of the dump logs into one small
# file for hand analysis: a block starts at a line naming
# shader statistics and runs until the next blank line.
python3 - "$OUT" <<'PYEOF' || note "stats extract: python3 pass failed, raw logs hold everything"
import re, sys, pathlib
out = pathlib.Path(sys.argv[1])
pat = re.compile(r"(vgpr|sgpr|spilled|scratch|lds|code size)", re.I)
import glob
names = ["allon.stderr.log", "fixture.stderr.log"] + \
        sorted(p.name for p in out.glob("family-*.stderr.log"))
for name in names:
    src = out / name
    if not src.exists():
        continue
    lines = src.read_text(errors="replace").splitlines()
    blocks, cur = [], None
    for ln in lines:
        if re.search(r"shader statistics|shader stats", ln, re.I):
            if cur: blocks.append(cur)
            cur = [ln]
        elif cur is not None:
            if ln.strip() == "" :
                blocks.append(cur); cur = None
            elif pat.search(ln):
                cur.append(ln)
    if cur: blocks.append(cur)
    dst = out / (name.replace(".stderr.log", "") + ".stats-extract.txt")
    dst.write_text("\n\n".join("\n".join(b) for b in blocks) + "\n")
    print(f"{name}: {len(blocks)} stats blocks -> {dst.name}")
PYEOF

note "AUTOPSY_DONE"
