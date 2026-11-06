#!/bin/bash
# Failure bundle wrapper: run a bench or harness command
# and, if it fails, preserve EVERYTHING about the run in a
# bundle under bench/failures/ so the wrong data survives
# to build fixes on. Passing runs leave one line in
# bench/failures/INDEX.md and their bulky logs are deleted.
#
# Why this exists: David's standing want is every form of
# debugging data we can squeeze out, and the WRONG data kept
# in full. Until now wrong outputs were preserved only by
# hand (one example: the degenerate top-5 output a lane
# refused to commit and kept on the server as
# ~/overnight-golden/out/glimmer.top5.degenerate.jsonl).
# This wrapper makes the preservation automatic and
# uniform. The convention, the bundle format, and the list
# of historical wrong-data artifacts live in
# bench/failure-bundles-2026-11-06.md.
#
# A run FAILS when the command exits nonzero OR when its
# combined stdout+stderr matches any FAILURE_MARKERS entry
# (a clean exit with wrong data is still wrong data; the
# degenerate top-5 run exited 0). Either way a full bundle
# is assembled. The wrapper exits with the wrapped
# command's exit code in all cases, so a marker-only
# failure still exits 0: callers detect it through the
# bundle and the INDEX, not the exit code.
#
# Usage:
#   bench/failure-bundle.sh <slug> [outdir] -- <command...>
#
#   slug     short name for the run; sanitized to
#            [A-Za-z0-9._-] and used in the bundle
#            directory name.
#   outdir   optional directory the command writes its
#            output files into; on failure its contents
#            are copied into the bundle under outputs/.
#   command  the bench/harness command. It runs from the
#            repo root (this script cd's there, like the
#            other bench scripts) and must be a foreground
#            run; its stdout and stderr are tee'd to the
#            caller's terminal and to the bundle logs.
#
# Environment knobs:
#   FB_MODEL  model file path for the manifest's
#             path/size/sha256 block. Default: the first
#             command argument that is an existing *.gguf
#             file. Hashing is streamed (sha256sum) and
#             cached by path+size+mtime in the vault's
#             model-hash-index.tsv, so a 16 GB model is
#             hashed once, not once per run.
#   FB_VAULT  vault directory. Default: bench/failures
#             under the repo root.
#
# shellcheck: NOT RUN. shellcheck is not installed on the
# build VM (checked 2026-11-06: `command -v shellcheck`
# is empty), so this script carries the repo bench style
# (set -u, everything quoted) without a shellcheck pass.
# Run `shellcheck bench/failure-bundle.sh` on any host
# that has it before relying on a clean bill.

set -u

# Failure markers: fixed strings, case-sensitive, matched
# as substrings against the combined stdout+stderr of the
# wrapped command. This list decides what counts as wrong
# data; extend it in review, never silently.
FAILURE_MARKERS=("FAIL" "FAILED" "mismatch" "DIVERGES" "REFUSED" "degenerate")

usage() {
    echo "usage: bench/failure-bundle.sh <slug> [outdir] -- <command...>" >&2
    exit 64
}

[ $# -ge 1 ] || usage
slug_raw=$1
shift
slug=$(printf '%s' "$slug_raw" | tr -c 'A-Za-z0-9._-' '-')

outdir=""
if [ $# -ge 1 ] && [ "$1" != "--" ]; then
    outdir=$1
    shift
fi
[ $# -ge 1 ] && [ "$1" = "--" ] || usage
shift
[ $# -ge 1 ] || usage

caller_pwd=$PWD
# Resolve caller-relative paths before the cd to repo root.
if [ -n "$outdir" ]; then
    case "$outdir" in
        /*) ;;
        *) outdir="$caller_pwd/$outdir" ;;
    esac
fi
fb_model=${FB_MODEL:-}
if [ -n "$fb_model" ]; then
    case "$fb_model" in
        /*) ;;
        *) fb_model="$caller_pwd/$fb_model" ;;
    esac
fi

cd "$(dirname "$0")/.." || exit 64
repo_root=$PWD
vault=${FB_VAULT:-"$repo_root/bench/failures"}
case "$vault" in
    /*) ;;
    *) vault="$caller_pwd/$vault" ;;
esac
mkdir -p "$vault"

# --- start-of-run state ---------------------------------------------
utc_start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
local_start=$(date +%Y-%m-%dT%H:%M:%S%z)
epoch_start=$(date +%s)
host_name=$(hostname)
uname_line=$(uname -a)
git_sha=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
git_status=$(git status --porcelain=v1 2>/dev/null || echo "unavailable")
if [ -z "$git_status" ]; then
    git_dirty="clean"
else
    git_dirty="dirty"
fi

ram_bytes() {
    # MemAvailable from /proc/meminfo, in bytes; empty if unreadable.
    awk '/^MemAvailable:/ {print $2 * 1024}' /proc/meminfo 2>/dev/null
}
vram_used_bytes() {
    # First amdgpu-style sysfs counter that exists; empty otherwise
    # (this VM has no GPU at all, and that is recorded, not hidden).
    local f
    for f in /sys/class/drm/card*/device/mem_info_vram_used; do
        if [ -r "$f" ]; then
            cat "$f"
            return 0
        fi
    done
    return 0
}
human() {
    # Bytes to human size when numfmt exists; raw bytes otherwise.
    if [ -n "$1" ] && command -v numfmt >/dev/null 2>&1; then
        numfmt --to=iec "$1"
    else
        printf '%s' "$1"
    fi
}
ram_start=$(ram_bytes)
vram_start=$(vram_used_bytes)

# The resolved engine environment: every WGPU_COMPUTE_* and
# ITAK_* variable currently set (set-but-empty included).
env_lines=$(env | grep -E '^(WGPU_COMPUTE_|ITAK_)' | sort || true)
if [ -z "$env_lines" ]; then
    env_lines="(none set)"
fi

# Quote the command once, for the manifest, REPRO, and index.
cmd_quoted=""
for arg in "$@"; do
    printf -v q '%q' "$arg"
    cmd_quoted="$cmd_quoted $q"
done
cmd_quoted=${cmd_quoted# }

# --- run the command, tee'd ------------------------------------------
stage=$(mktemp -d)
fifo_dir="$stage/fifos"
mkdir -p "$fifo_dir"
mkfifo "$fifo_dir/out" "$fifo_dir/err"
tee "$stage/stdout.log" < "$fifo_dir/out" &
tee_out_pid=$!
tee "$stage/stderr.log" >&2 < "$fifo_dir/err" &
tee_err_pid=$!
"$@" > "$fifo_dir/out" 2> "$fifo_dir/err"
cmd_rc=$?
# The tees end when the command closes the fifos; wait reaps
# them, so both logs are complete and flushed before we read.
wait "$tee_out_pid" "$tee_err_pid" 2>/dev/null

# --- end-of-run state -------------------------------------------------
utc_end=$(date -u +%Y-%m-%dT%H:%M:%SZ)
local_end=$(date +%Y-%m-%dT%H:%M:%S%z)
epoch_end=$(date +%s)
duration=$((epoch_end - epoch_start))
ram_end=$(ram_bytes)
vram_end=$(vram_used_bytes)

cat "$stage/stdout.log" "$stage/stderr.log" > "$stage/combined.log"
matched_markers=""
for marker in "${FAILURE_MARKERS[@]}"; do
    count=$(grep -c -F -- "$marker" "$stage/combined.log" || true)
    if [ "$count" -gt 0 ]; then
        matched_markers="$matched_markers $marker($count)"
    fi
done
matched_markers=${matched_markers# }

verdict="PASS"
fail_reasons=""
if [ "$cmd_rc" -ne 0 ]; then
    verdict="FAIL"
    fail_reasons="exit code $cmd_rc"
fi
if [ -n "$matched_markers" ]; then
    verdict="FAIL"
    if [ -n "$fail_reasons" ]; then
        fail_reasons="$fail_reasons; failure markers matched: $matched_markers"
    else
        fail_reasons="failure markers matched: $matched_markers"
    fi
fi

# --- model file block (path/size/sha256, hash cached) -----------------
model_path=""
if [ -n "$fb_model" ] && [ -f "$fb_model" ]; then
    model_path=$fb_model
else
    for arg in "$@"; do
        case "$arg" in
            *.gguf)
                if [ -f "$arg" ]; then
                    model_path=$arg
                elif [ -f "$caller_pwd/$arg" ]; then
                    model_path="$caller_pwd/$arg"
                fi
                if [ -n "$model_path" ]; then
                    case "$model_path" in
                        /*) ;;
                        *) model_path="$repo_root/$model_path" ;;
                    esac
                    break
                fi
                ;;
        esac
    done
fi
model_size=""
model_mtime=""
model_sha=""
model_cache="none"
if [ -n "$model_path" ]; then
    model_size=$(stat -c %s "$model_path")
    model_mtime=$(stat -c %Y "$model_path")
    cache_file="$vault/model-hash-index.tsv"
    if [ -f "$cache_file" ]; then
        model_sha=$(awk -F '\t' -v p="$model_path" -v s="$model_size" -v m="$model_mtime" \
            '$1 == p && $2 == s && $3 == m {print $4}' "$cache_file")
    fi
    if [ -n "$model_sha" ]; then
        model_cache="hit"
    else
        model_sha=$(sha256sum "$model_path" | awk '{print $1}')
        printf '%s\t%s\t%s\t%s\n' "$model_path" "$model_size" "$model_mtime" "$model_sha" >> "$cache_file"
        model_cache="computed and cached"
    fi
fi

# --- index line (written for passes and failures alike) ---------------
index_file="$vault/INDEX.md"
if [ ! -f "$index_file" ]; then
    {
        echo "# Failure vault index"
        echo ""
        echo "One line per wrapped run (bench/failure-bundle.sh). FAIL rows"
        echo "point at a full bundle directory in this folder; PASS rows are"
        echo "the whole record, because a pass deletes its bulky logs by"
        echo "design. The convention is bench/failure-bundles-2026-11-06.md."
        echo ""
        echo "| UTC | slug | verdict | bundle | command |"
        echo "| --- | --- | --- | --- | --- |"
    } > "$index_file"
fi
index_cmd=$(printf '%s' "$cmd_quoted" | tr '|' '/')

if [ "$verdict" = "PASS" ]; then
    echo "| $utc_start | $slug | PASS | - | $index_cmd |" >> "$index_file"
    rm -rf "$stage"
    echo "[failure-bundle] PASS: $slug (index line only, logs deleted)" >&2
    exit "$cmd_rc"
fi

# --- assemble the failure bundle --------------------------------------
bundle_name="$(date -u +%Y%m%dT%H%M%SZ)-$slug"
bundle="$vault/$bundle_name"
n=2
while [ -e "$bundle" ]; do
    bundle="$vault/$bundle_name-$n"
    n=$((n + 1))
done
mkdir -p "$bundle"
mv "$stage/stdout.log" "$bundle/stdout.log"
mv "$stage/stderr.log" "$bundle/stderr.log"
printf '%s\n' "$env_lines" > "$bundle/env.txt"

outputs_note="not given"
outputs_count=0
if [ -n "$outdir" ]; then
    if [ -d "$outdir" ]; then
        mkdir -p "$bundle/outputs"
        cp -a "$outdir/." "$bundle/outputs/"
        outputs_count=$(find "$bundle/outputs" -type f | wc -l)
        outputs_note="captured from $outdir ($outputs_count files)"
    else
        outputs_note="given as $outdir but not a directory after the run; nothing captured"
    fi
fi

{
    echo "slug: $slug"
    echo "verdict: FAIL ($fail_reasons)"
    echo "command: $cmd_quoted"
    echo "exit_code: $cmd_rc"
    echo "markers_matched: ${matched_markers:-none}"
    echo "utc_start: $utc_start"
    echo "utc_end: $utc_end"
    echo "local_start: $local_start"
    echo "local_end: $local_end"
    echo "duration_seconds: $duration"
    echo "host: $host_name"
    echo "uname: $uname_line"
    echo "repo_git_sha: $git_sha"
    echo "repo_branch: $git_branch"
    echo "repo_dirty: $git_dirty"
    echo "model_path: ${model_path:-none detected}"
    echo "model_size_bytes: ${model_size:-}"
    echo "model_mtime_epoch: ${model_mtime:-}"
    echo "model_sha256: ${model_sha:-}"
    echo "model_hash_cache: $model_cache"
    echo "ram_available_start_bytes: ${ram_start:-unavailable}"
    echo "ram_available_start_human: $(human "$ram_start")"
    echo "ram_available_end_bytes: ${ram_end:-unavailable}"
    echo "ram_available_end_human: $(human "$ram_end")"
    echo "vram_used_start_bytes: ${vram_start:-unavailable}"
    echo "vram_used_start_human: $(human "$vram_start")"
    echo "vram_used_end_bytes: ${vram_end:-unavailable}"
    echo "vram_used_end_human: $(human "$vram_end")"
    echo "outputs_dir: $outputs_note"
    echo ""
    echo "engine environment (also in env.txt):"
    printf '%s\n' "$env_lines"
    echo ""
    echo "git status --porcelain at start of run:"
    if [ "$git_dirty" = "clean" ]; then
        echo "(clean)"
    else
        printf '%s\n' "$git_status"
    fi
} > "$bundle/manifest.txt"

{
    echo "# Repro: $slug"
    echo ""
    echo "## Command"
    echo ""
    echo '```sh'
    echo "$cmd_quoted"
    echo '```'
    echo ""
    echo "Run from the repo root (the wrapper's working directory),"
    echo "with the environment in this bundle's env.txt."
    echo ""
    echo "## Pass condition that failed"
    echo ""
    echo "Expected: exit code 0 and none of the failure markers"
    echo "(${FAILURE_MARKERS[*]}) in the combined stdout+stderr."
    echo ""
    echo "Observed: $fail_reasons."
    echo ""
    echo "## State at the run"
    echo ""
    echo "- git: $git_sha (branch $git_branch, working tree $git_dirty)"
    echo "- model: ${model_path:-none detected} sha256 ${model_sha:-n/a}"
    echo "- full output: stdout.log and stderr.log in this bundle"
    if [ -n "$outdir" ]; then
        echo "- command outputs: outputs/ in this bundle ($outputs_note)"
    fi
} > "$bundle/REPRO.md"

echo "| $utc_start | $slug | FAIL | $bundle_name | $index_cmd |" >> "$index_file"
rm -rf "$stage"
echo "[failure-bundle] FAIL: $slug -> bench/failures/$bundle_name ($fail_reasons)" >&2
exit "$cmd_rc"
