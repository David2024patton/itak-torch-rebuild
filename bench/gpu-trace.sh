#!/bin/bash
# GPU trace wrapper: run one command with every form of GPU
# and host visibility this stack can produce switched on,
# and file the artifacts as one run directory.
#
# Why: the diag system gives per-kernel GPU durations but
# not where the card's waves actually stalled; the shader
# autopsy (bench/shader-autopsy.sh) captures compiled form
# only inside dedicated autopsy sessions; and the host
# readings that decided past verdicts (VRAM from amdgpu
# sysfs, process RSS from /proc) were taken by hand and
# lived in chat logs, not next to the runs they describe.
# The 2026-11-04 streaming-rate research named the four
# remaining forms: SQTT thread traces, hang dumps, a host
# telemetry timeline, and a bandwidth canary. This wrapper
# packages those four around any command.
#
# What the wrapped command gets (env applied to the child
# only, via env(1); the caller's shell is untouched):
#   MESA_VK_TRACE=rgp            SQTT capture for Radeon
#                                GPU Profiler, via RADV's
#                                built-in capture. The old
#                                RADV_THREAD_TRACE switches
#                                were removed from Mesa; do
#                                not use them.
#   MESA_VK_TRACE_PER_SUBMIT=1   per-submit capture, the
#                                documented knob for
#                                compute-only workloads.
#   RADV_DEBUG=shaderstats,asm,cs
#                                compiled-form dumps on the
#                                child's stderr (the same
#                                flags as the autopsy), plus
#                                `hang` when GPU_TRACE_HANG=1
#                                is set, plus any caller
#                                RADV_DEBUG appended.
#   MESA_SHADER_CACHE_DIR        private to the run dir, so
#                                shaders compile cold and the
#                                stats/asm dumps actually
#                                happen (a warm shared cache
#                                compiles nothing and prints
#                                nothing). A caller-set value
#                                is kept and recorded.
#
# Artifacts, in bench/traces/<UTC stamp>-<label>/:
#   stdout.log, stderr.log  the child's output, streamed
#                           live through tee and filed.
#   *.rgp                   the SQTT capture (or the
#                           per-submit set Mesa produced),
#                           collected after the run from the
#                           working directory, /tmp, and
#                           $HOME: Mesa's dump directory is
#                           not settable through a
#                           documented env var, so the
#                           wrapper sweeps. The sweep is
#                           why traced runs belong inside
#                           the phase lock like every other
#                           card session.
#   rgp-sources.txt         where each .rgp was found.
#   shaderstats.txt         stats blocks extracted from
#                           stderr.log (the autopsy
#                           extractor's shape).
#   asm/                    one numbered file per
#                           disassembly block (RADV dump
#                           headers carry no shader name),
#                           or asm/NO_ASM_FOUND.txt when no
#                           block matched the key.
#   telemetry.csv           host timeline at ~10 Hz for
#                           the child's lifetime: VRAM
#                           used, child RSS, CPU load,
#                           GPU edge temperature and power
#                           (amdgpu sysfs). Absent sources
#                           degrade to empty columns and
#                           are declared in the header.
#   canary.txt              vkpeak (or clpeak) output,
#                           read AFTER the child exits so
#                           the canary can never steal
#                           bandwidth from the measured
#                           run; or the recorded absence
#                           of both tools.
#   manifest.txt            command, env applied, git
#                           SHA, Mesa version if readable,
#                           exit code, artifact inventory.
# And one line per run appended to bench/traces/INDEX.md.
#
# Exit status: the wrapped command's, preserved exactly.
# The wrapper never times the child out. If the card
# wedges, telemetry keeps sampling until the operator
# kills the child; the wrapper then files everything.
# That is the hang-forensics case: see
# bench/gpu-trace-2026-11-06.md for the standing rule
# (one rerun under GPU_TRACE_HANG=1 before any reboot
# or reset).
#
# Usage:
#   bench/gpu-trace.sh LABEL [--] COMMAND [ARGS...]
# Knobs: GPU_TRACE_HANG=1     add RADV_DEBUG=hang
#        GPU_TRACE_INTERVAL  telemetry period in seconds
#                            (default 0.1, about 10 Hz)
#        GPU_TRACE_DIR       artifact root (default
#                            bench/traces under the repo)
#        EXPECTED_SHA        manifest SHA when the tree
#                            has no .git (archive stage;
#                            same convention as the
#                            autopsy script)
#
# shellcheck: NOT RUN for this script. shellcheck is not
# installed on the build VM (checked 2026-11-06:
# `command -v shellcheck` prints nothing) and this lane
# installs nothing. The script follows the quoting
# discipline of bench/shader-autopsy.sh and was exercised
# end to end on the VM, including the exit-code and
# telemetry demos recorded in bench/gpu-trace-2026-11-06.md.
# Run `shellcheck bench/gpu-trace.sh` on the server before
# the first traced verdict session and fix what it says.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 2

usage() {
  echo "usage: bench/gpu-trace.sh LABEL [--] COMMAND [ARGS...]" >&2
  echo "  runs COMMAND with SQTT capture, RADV shader dumps," >&2
  echo "  host telemetry, and a bandwidth canary; artifacts in" >&2
  echo "  bench/traces/<UTC stamp>-<label>/; exit code is the" >&2
  echo "  command's. Knobs: GPU_TRACE_HANG=1, GPU_TRACE_INTERVAL," >&2
  echo "  GPU_TRACE_DIR, EXPECTED_SHA. See bench/gpu-trace-2026-11-06.md." >&2
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi
[ $# -ge 2 ] || { usage; exit 2; }
LABEL_RAW="$1"; shift
if [ "${1:-}" = "--" ]; then shift; fi
[ $# -ge 1 ] || { usage; exit 2; }

LABEL="$(printf '%s' "$LABEL_RAW" | tr -c 'A-Za-z0-9._-' '-' | cut -c1-48)"
[ -n "$LABEL" ] || LABEL="run"
TRACES_DIR="${GPU_TRACE_DIR:-$ROOT/bench/traces}"
INTERVAL="${GPU_TRACE_INTERVAL:-0.1}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="$STAMP-$LABEL"
RUNDIR="$TRACES_DIR/$RUN_NAME"
n=2
while [ -e "$RUNDIR" ]; do
  RUN_NAME="$STAMP-$LABEL-$n"
  RUNDIR="$TRACES_DIR/$RUN_NAME"
  n=$((n + 1))
done
mkdir -p "$RUNDIR/asm" || { echo "FAIL: cannot create $RUNDIR" >&2; exit 2; }

CMD_STR="$(printf '%q ' "$@")"
SHA="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo "${EXPECTED_SHA:-unknown}")"

mesa_version() {
  local v
  v="$(vulkaninfo --summary 2>/dev/null | grep -E 'driverInfo' | head -1 | sed -E 's/^[^=]*=[[:space:]]*//')"
  if [ -n "$v" ]; then echo "$v (vulkaninfo driverInfo)"; return; fi
  v="$(dpkg-query -W -f='${Version}' mesa-vulkan-drivers 2>/dev/null)"
  if [ -n "$v" ]; then echo "$v (dpkg mesa-vulkan-drivers)"; return; fi
  echo "not readable on this host (no vulkaninfo, no dpkg mesa-vulkan-drivers)"
}

# Effective child environment (recorded in the manifest).
EFFECTIVE_RADV_DEBUG="shaderstats,asm,cs"
if [ "${GPU_TRACE_HANG:-}" = "1" ]; then
  EFFECTIVE_RADV_DEBUG="$EFFECTIVE_RADV_DEBUG,hang"
fi
CALLER_RADV_DEBUG="${RADV_DEBUG:-}"
if [ -n "$CALLER_RADV_DEBUG" ]; then
  EFFECTIVE_RADV_DEBUG="$EFFECTIVE_RADV_DEBUG,$CALLER_RADV_DEBUG"
fi
if [ -n "${MESA_SHADER_CACHE_DIR:-}" ]; then
  EFFECTIVE_CACHE="$MESA_SHADER_CACHE_DIR"
  CACHE_SOURCE="caller"
else
  EFFECTIVE_CACHE="$RUNDIR/shader-cache"
  CACHE_SOURCE="private (run dir)"
fi

MANIFEST="$RUNDIR/manifest.txt"
{
  echo "run: $RUN_NAME"
  echo "label: $LABEL"
  echo "start_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "start_epoch: $(date +%s)"
  echo "host: $(hostname)"
  echo "kernel: $(uname -srm)"
  echo "repo_root: $ROOT"
  echo "git_sha: $SHA"
  echo "command: $CMD_STR"
  echo "child_env: MESA_VK_TRACE=rgp MESA_VK_TRACE_PER_SUBMIT=1 RADV_DEBUG=$EFFECTIVE_RADV_DEBUG MESA_SHADER_CACHE_DIR=$EFFECTIVE_CACHE"
  echo "shader_cache_source: $CACHE_SOURCE"
  echo "caller_radv_debug: ${CALLER_RADV_DEBUG:-(unset)}"
  echo "hang_mode: ${GPU_TRACE_HANG:-0}"
  echo "telemetry_interval_s: $INTERVAL"
  echo "mesa_version: $(mesa_version)"
} > "$MANIFEST"

echo "[gpu-trace] run dir: $RUNDIR"
echo "[gpu-trace] command: $CMD_STR"
touch "$RUNDIR/.start-marker"
START_EPOCH="$(date +%s)"

# --- the wrapped command -----------------------------------
# Streams live through tee to the terminal and to the logs;
# backgrounded only so the telemetry sampler can be aimed
# at its PID. `wait` below returns the child's own status.
env MESA_VK_TRACE=rgp MESA_VK_TRACE_PER_SUBMIT=1 \
    RADV_DEBUG="$EFFECTIVE_RADV_DEBUG" \
    MESA_SHADER_CACHE_DIR="$EFFECTIVE_CACHE" \
    "$@" > >(tee "$RUNDIR/stdout.log") 2> >(tee "$RUNDIR/stderr.log" >&2) &
CHILD_PID=$!
echo "[gpu-trace] child pid: $CHILD_PID"

# --- telemetry sampler (embedded python3) -------------------
python3 - "$RUNDIR/telemetry.csv" "$CHILD_PID" "$RUNDIR/.stop" "$INTERVAL" <<'PYEOF' 2>"$RUNDIR/sampler.err" &
import glob, os, sys, time

out_path, pid, stop_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
interval = float(sys.argv[4])

def read_first(path):
    try:
        with open(path, "r", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return None

def find_vram():
    # The big card, by the same rule the bench runners use:
    # the amdgpu device whose VRAM total exceeds 20 GB.
    for dev in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
        total = read_first(os.path.join(dev, "mem_info_vram_total"))
        used = os.path.join(dev, "mem_info_vram_used")
        if total is None or not os.path.exists(used):
            continue
        try:
            if int(total) > 20000000000:
                return used
        except ValueError:
            continue
    return None

def find_hwmon():
    for h in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        if read_first(os.path.join(h, "name")) == "amdgpu":
            t = os.path.join(h, "temp1_input")
            p = os.path.join(h, "power1_average")
            return (t if os.path.exists(t) else None,
                    p if os.path.exists(p) else None)
    return None, None

vram_path = find_vram()
temp_path, power_path = find_hwmon()

def rss_kib():
    txt = read_first("/proc/%d/status" % pid)
    if txt is None:
        return ""
    for line in txt.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                return parts[1]
    return ""

def load1():
    txt = read_first("/proc/loadavg")
    return txt.split()[0] if txt else ""

def val_or_empty(path):
    if not path:
        return ""
    v = read_first(path)
    return v if v is not None else ""

f = open(out_path, "w", buffering=1)
f.write("# gpu-trace telemetry: host context for the wrapped run.\n")
f.write("# vram source: %s\n" % (vram_path or
        "ABSENT (no amdgpu mem_info counters on this host)"))
f.write("# temp source: %s\n" % (temp_path or
        "ABSENT (no amdgpu hwmon on this host)"))
f.write("# power source: %s\n" % (power_path or
        "ABSENT (no amdgpu hwmon on this host)"))
f.write("# rss source: /proc/%d/status VmRSS (wrapped process only;"
        " children it forks are not summed)\n" % pid)
f.write("# load source: /proc/loadavg field 1\n")
f.write("utc_iso,epoch_s,vram_used_bytes,proc_rss_kib,load1,"
        "gpu_edge_temp_millic,gpu_power_microw\n")

max_samples = int(21600 / max(interval, 0.01))  # about 6 h, anti-orphan
count = 0
while count < max_samples:
    if os.path.exists(stop_path):
        break
    if not os.path.exists("/proc/%d" % pid):
        break
    now = time.time()
    f.write("%s,%.3f,%s,%s,%s,%s,%s\n" % (
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        now,
        val_or_empty(vram_path),
        rss_kib(),
        load1(),
        val_or_empty(temp_path),
        val_or_empty(power_path)))
    count += 1
    delay = interval - (time.time() - now)
    end = time.time() + max(delay, 0.0)
    while time.time() < end:
        if os.path.exists(stop_path):
            break
        time.sleep(min(0.02, max(end - time.time(), 0.0)))
f.close()
PYEOF
SAMPLER_PID=$!

wait "$CHILD_PID"
CHILD_RC=$?
END_EPOCH="$(date +%s)"
echo "[gpu-trace] child exit: $CHILD_RC"

# Stop the sampler, then drain the tee processes.
touch "$RUNDIR/.stop"
for _ in $(seq 1 50); do
  kill -0 "$SAMPLER_PID" 2>/dev/null || break
  sleep 0.1
done
kill "$SAMPLER_PID" 2>/dev/null || true
wait "$SAMPLER_PID" 2>/dev/null || true
wait 2>/dev/null || true
if [ -s "$RUNDIR/sampler.err" ]; then
  echo "[gpu-trace] sampler wrote errors; kept as sampler.err"
else
  rm -f "$RUNDIR/sampler.err"
fi

# --- collect the SQTT captures ------------------------------
: > "$RUNDIR/rgp-sources.txt"
for base in "$ROOT" /tmp "$HOME"; do
  [ -d "$base" ] || continue
  find "$base" -maxdepth 1 -type f -name '*.rgp' \
       -newer "$RUNDIR/.start-marker" 2>/dev/null
done | sort -u | while IFS= read -r f; do
  case "$f" in
    "$RUNDIR"/*) continue ;;
  esac
  dest="$RUNDIR/$(basename "$f")"
  if mv "$f" "$dest" 2>/dev/null; then
    echo "$f -> $dest" >> "$RUNDIR/rgp-sources.txt"
  fi
done
RGP_COUNT="$(find "$RUNDIR" -maxdepth 1 -type f -name '*.rgp' | wc -l)"
echo "[gpu-trace] rgp files collected: $RGP_COUNT"

# --- extract shader stats and disassembly from stderr -------
python3 - "$RUNDIR" <<'PYEOF'
import re, sys, pathlib
rundir = pathlib.Path(sys.argv[1])
stderr = rundir / "stderr.log"
text = stderr.read_text(errors="replace") if stderr.exists() else ""
lines = text.splitlines()

# Shader stats blocks: the autopsy script's extractor shape.
pat = re.compile(r"(vgpr|sgpr|spilled|scratch|lds|code size)", re.I)
blocks, cur = [], None
for ln in lines:
    if re.search(r"shader statistics|shader stats", ln, re.I):
        if cur: blocks.append(cur)
        cur = [ln]
    elif cur is not None:
        if ln.strip() == "":
            blocks.append(cur); cur = None
        elif pat.search(ln):
            cur.append(ln)
if cur: blocks.append(cur)
stats_path = rundir / "shaderstats.txt"
if blocks:
    stats_path.write_text(
        "# shader stats blocks extracted from stderr.log by\n"
        "# bench/gpu-trace.sh. RADV dump headers carry no shader\n"
        "# name; order is stderr order; stderr.log is the ground\n"
        "# truth.\n\n" +
        "\n\n".join("\n".join(b) for b in blocks) + "\n")
else:
    stats_path.write_text(
        "# no shader stats blocks found in stderr.log\n"
        "# (causes: no Vulkan device on this host, the wrapped\n"
        "#  command compiled no shaders, or the dumps landed\n"
        "#  somewhere else; stderr.log is the ground truth)\n")

# Disassembly blocks: keyed on short header lines naming a
# disassembly; a stats header or the next disassembly header
# closes a block. RADV spells these headers itself, so the
# key is deliberately loose and NO_ASM_FOUND says so.
asmdir = rundir / "asm"
asmdir.mkdir(exist_ok=True)
asm_blocks, cur = [], None
for ln in lines:
    if re.search(r"disassembly", ln, re.I) and len(ln) < 200:
        if cur: asm_blocks.append(cur)
        cur = [ln]
    elif re.search(r"shader statistics|shader stats", ln, re.I):
        if cur: asm_blocks.append(cur)
        cur = None
    elif cur is not None:
        cur.append(ln)
if cur: asm_blocks.append(cur)
if asm_blocks:
    for i, b in enumerate(asm_blocks[:1000], 1):
        (asmdir / ("shader-%04d.asm" % i)).write_text("\n".join(b) + "\n")
else:
    (asmdir / "NO_ASM_FOUND.txt").write_text(
        "No disassembly blocks matched in stderr.log (keyed on\n"
        "short lines containing 'disassembly'). If RADV spelled\n"
        "its headers differently on this Mesa, the dumps are\n"
        "still in stderr.log; fix the key in bench/gpu-trace.sh\n"
        "and record it in the bench note for the run.\n")
(rundir / ".extract-counts").write_text("%d %d\n" % (len(blocks), len(asm_blocks)))
print("stats_blocks=%d asm_blocks=%d" % (len(blocks), len(asm_blocks)))
PYEOF
STATS_BLOCKS=0
ASM_BLOCKS=0
if [ -f "$RUNDIR/.extract-counts" ]; then
  read -r STATS_BLOCKS ASM_BLOCKS < "$RUNDIR/.extract-counts"
fi
echo "[gpu-trace] stats blocks: $STATS_BLOCKS; asm blocks: $ASM_BLOCKS"

# --- bandwidth canary (after the run, never concurrent) -----
run_canary_tool() {
  if command -v timeout >/dev/null 2>&1; then
    timeout 180 "$@" 2>&1
  else
    "$@" 2>&1
  fi
}
CANARY_TOOL="absent"
{
  echo "canary for run: $RUN_NAME"
  echo "timing: read after the wrapped command exited, never concurrent"
  if command -v vkpeak >/dev/null 2>&1; then
    CANARY_TOOL="vkpeak"
    echo "tool: vkpeak at $(command -v vkpeak)"
    run_canary_tool vkpeak
    echo "vkpeak exit: $?"
  elif command -v clpeak >/dev/null 2>&1; then
    CANARY_TOOL="clpeak"
    echo "tool: clpeak at $(command -v clpeak)"
    run_canary_tool clpeak
    echo "clpeak exit: $?"
  else
    echo "ABSENT: neither vkpeak nor clpeak is on PATH on $(hostname)"
    echo "at $(date -Is). No canary reading this run. Kernel stream"
    echo "rates from this run compare against the recorded references"
    echo "instead: card spec 960 GB/s; GEMV measured 649.5 GB/s and"
    echo "the MMQ pair 43.5 GB/s in the 2026-11-05 prefill profile."
  fi
} > "$RUNDIR/canary.txt" 2>&1
echo "[gpu-trace] canary: $CANARY_TOOL (see canary.txt)"

# --- finalize manifest and index ----------------------------
TELEM_ROWS="$(grep -cE '^[0-9]{4}-' "$RUNDIR/telemetry.csv" 2>/dev/null || true)"
# The scratch files have served their purpose by here (the
# sweep used the marker, the sampler used the stop file,
# the counts are read); remove them before the inventory
# so the inventory lists only real artifacts.
rm -f "$RUNDIR/.start-marker" "$RUNDIR/.stop" "$RUNDIR/.extract-counts"
{
  echo "end_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "end_epoch: $END_EPOCH"
  echo "child_pid: $CHILD_PID"
  echo "child_exit: $CHILD_RC"
  echo "child_wall_s: $((END_EPOCH - START_EPOCH))"
  echo "rgp_files_collected: $RGP_COUNT"
  echo "shaderstats_blocks: $STATS_BLOCKS"
  echo "asm_blocks: $ASM_BLOCKS"
  echo "telemetry_rows: ${TELEM_ROWS:-0}"
  echo "canary_tool: $CANARY_TOOL"
  echo "--- inventory (bytes path; manifest.txt itself excluded) ---"
  (cd "$RUNDIR" && find . -type f ! -name 'manifest.txt' -printf '%s %p\n' | sort -k2)
} >> "$MANIFEST"

INDEX="$TRACES_DIR/INDEX.md"
if [ ! -f "$INDEX" ]; then
  {
    echo "# GPU trace runs"
    echo
    echo "One line per bench/gpu-trace.sh run. Artifacts live in the"
    echo "run directory named on each line. See bench/gpu-trace-2026-11-06.md."
    echo
  } > "$INDEX"
fi
INDEX_CMD="$(printf '%s' "$CMD_STR" | tr '`' "'")"
echo "- \`$RUN_NAME\` — exit $CHILD_RC — \`$INDEX_CMD\` — rgp: $RGP_COUNT; stats blocks: $STATS_BLOCKS; asm blocks: $ASM_BLOCKS; telemetry rows: ${TELEM_ROWS:-0}; canary: $CANARY_TOOL" >> "$INDEX"

echo "[gpu-trace] done: $RUNDIR (child exit $CHILD_RC preserved)"
exit "$CHILD_RC"
