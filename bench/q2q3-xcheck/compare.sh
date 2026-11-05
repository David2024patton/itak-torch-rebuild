#!/bin/bash
# Local half of the cross-check: run the Rust probe over every vector
# set copied back from the server staging and diff against ggml's
# reference outputs. Usage: compare.sh <dir-with-bins-and-refs>
set -uo pipefail
DIR=${1:?usage: compare.sh <dir>}
cd "$DIR"
WT=~/workspace/lanes/overnight-q2q3
export PATH="$HOME/.cargo/bin:$PATH"

run_set() {
    local ty=$1 bin=$2 block_bytes=$3
    local ref="${bin%.bin}_ref.f32"
    if [ ! -f "$bin" ]; then echo "$bin: absent, skipped"; return 0; fi
    local size blocks elements
    size=$(stat -c %s "$bin")
    blocks=$((size / block_bytes))
    elements=$((blocks * 256))
    cargo run --release --quiet --manifest-path "$WT/Cargo.toml" \
        -p gguf --example probe -- "$ty" "$bin" "$elements" "$elements" \
        > "${bin%.bin}_rust.txt" 2>/dev/null
    python3 "$WT/bench/q2q3-xcheck/diff.py" "$ref" "${bin%.bin}_rust.txt" "${bin%.bin}"
}

run_set q2_k q2k_synth.bin 84
run_set q3_k q3k_synth.bin 110
run_set q2_k q2k_dsweep.bin 84
run_set q3_k q3k_dsweep.bin 110
run_set q2_k q2k_quant.bin 84
run_set q3_k q3k_quant.bin 110
run_set q2_k q2k_real.bin 84
run_set q3_k q3k_real.bin 110
