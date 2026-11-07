#!/bin/bash
# Local half of the cross-check: run the Rust probe over every vector
# set copied back from the server staging and diff against ggml's
# reference outputs. Usage: compare.sh <dir-with-bins-and-refs>
#
# Exit status: nonzero if any PRESENT set fails its diff (or its probe
# run fails), or if no set was present at all (a run that compared
# nothing proves nothing). Absent sets are skipped, named in the
# per-set summary, and do not by themselves fail the run, so a partial
# staging copied back for a quick check stays usable.
set -uo pipefail
DIR=${1:?usage: compare.sh <dir>}
cd "$DIR"
WT=~/workspace/lanes/overnight-q2q3
export PATH="$HOME/.cargo/bin:$PATH"

PASSED=()
FAILED=()
SKIPPED=()

run_set() {
    local ty=$1 bin=$2 block_bytes=$3
    local ref="${bin%.bin}_ref.f32"
    if [ ! -f "$bin" ]; then
        echo "$bin: absent, skipped"
        SKIPPED+=("$bin")
        return 0
    fi
    local size blocks elements
    size=$(stat -c %s "$bin")
    blocks=$((size / block_bytes))
    elements=$((blocks * 256))
    if ! cargo run --release --quiet --manifest-path "$WT/Cargo.toml" \
        -p gguf --example probe -- "$ty" "$bin" "$elements" "$elements" \
        > "${bin%.bin}_rust.txt" 2>/dev/null; then
        echo "$bin: probe run failed"
        FAILED+=("$bin (probe run failed)")
        return 0
    fi
    if python3 "$WT/bench/q2q3-xcheck/diff.py" "$ref" "${bin%.bin}_rust.txt" "${bin%.bin}"; then
        PASSED+=("$bin")
    else
        FAILED+=("$bin")
    fi
}

run_set q2_k q2k_synth.bin 84
run_set q3_k q3k_synth.bin 110
run_set q2_k q2k_dsweep.bin 84
run_set q3_k q3k_dsweep.bin 110
run_set q2_k q2k_quant.bin 84
run_set q3_k q3k_quant.bin 110
run_set q2_k q2k_real.bin 84
run_set q3_k q3k_real.bin 110

echo "---- compare.sh summary ----"
for bin in "${PASSED[@]}"; do echo "PASS    $bin"; done
for bin in "${FAILED[@]}"; do echo "FAIL    $bin"; done
for bin in "${SKIPPED[@]}"; do echo "SKIPPED $bin"; done
echo "sets: ${#PASSED[@]} passed, ${#FAILED[@]} failed, ${#SKIPPED[@]} skipped"
if [ "${#FAILED[@]}" -gt 0 ]; then
    exit 1
fi
if [ "${#PASSED[@]}" -eq 0 ]; then
    echo "compare.sh: no sets were compared; nothing was verified" >&2
    exit 1
fi
exit 0
