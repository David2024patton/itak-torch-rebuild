#!/bin/bash
# Gate wrapper with skip visibility: run the workspace test
# suite (the first of the RUNBOOK's plain gate commands) with
# per-test output captured, count what passed, what failed,
# and what SKIPPED, and fail the gate when tests skip on a
# machine that could have run them.
#
# Why this exists: the cold critic panel's Category J finding
# (bench/critic report 2026-11-07, merged verdict DO NOT SHIP
# on other grounds). Roughly one Rust test in six in this
# workspace needs a GPU adapter and, on a machine without
# one, prints a "skipping" line and passes vacuously. The
# plain gate (`cargo test --workspace`) captures a passing
# test's stderr and discards it, so the green result shows
# no trace that anything skipped: at the panel freeze the
# suite reported 758 passed while 119 skip lines sat hidden,
# covering effectively every kernel-parity, resident-session,
# prefix-KV, DFlash, coopmat, and F16-KV test. The standing
# lesson is the K21F mask error earlier in this project's
# history: a one-character shader defect shipped through
# green gates and was caught only on the card, because the
# tests that would have caught it were silently not running
# on the machines that ran the gates. Until CI exists (TODO
# T-1.7, still open; this wrapper is NOT that CI), a gate
# report must state the skip count beside the pass count,
# and on a machine WITH an adapter any adapter skip is a
# failure, not a footnote.
#
# What counts as a skip: the tests announce their own skips
# on stderr (visible only under --nocapture, which is how
# this wrapper runs the suite). Two adapter-absence idioms
# exist in the tree, surveyed file by file 2026-11-07:
#
#   A. "skipping <name>: no suitable adapter (enumerated:
#      [...])" — printed by the per-file backend_or_skip()
#      helpers and their inline variants when
#      WgpuCompute::new() refuses with NoSuitableAdapter.
#   B. "skipping <name>: wgpu enumerates no adapters here"
#      — the shader-compile suites' wording for the same
#      absence.
#
# A line counts as an ADAPTER SKIP when it contains
# "skipping" and either idiom's marker ("no suitable
# adapter" or "wgpu enumerates no adapters here"). Any
# other line containing "skipping" counts as an OTHER SKIP:
# the named exceptions below. Under --nocapture, parallel
# tests' output interleaves and a line can be garbled or
# merged (one such merged line was observed in the base
# measurement), so counts are exact to within a line or
# two; the adapter-machine rule (any adapter skip fails)
# and the printed beside-the-pass-count numbers are both
# robust at that tolerance.
#
# The adapterless-expected set (T-1.7's exception list,
# starting form). On a machine with no qualifying adapter,
# the adapter skips below are EXPECTED; they compose from:
#
#   - wgpu-compute integration suites, via backend_or_skip()
#     helpers: chunked_gemv, dflash_session, fused_kernels,
#     k10_k11_gemm, k12_k14_mmq, k15_k16_mmq_wide,
#     k18_small_batch, k19_mmv, k1_synthetic, k20_mmv,
#     k21_finish, k21_mmv, k21f_head_routing, k22_mmq,
#     k23_coopmat, k23_mmq, k24_coopmat, k2_k3_synthetic,
#     k4_k8_kernels, k8_argmax, prefix_kv_session,
#     proj_bench, resident_block, splitk_mmq.
#   - wgpu-compute integration suites with inline variants
#     of the same print: cross_backend, kv_f16,
#     resident_session (including its diag smoke, self-drain,
#     and off-check adapter arms), diag_timestamps.
#   - idiom B: shader_compile (14 print sites) and
#     k4_k8_kernels' shader-compile test.
#   - unit tests inside the crates: wgpu-compute's k21
#     finish registration test (src/compute.rs), itak-spec's
#     gpu arm speculator test (src/speculator.rs).
#   - itak-spec's ngram_gpu_runner integration test.
#
# Measured at base 34358e3 (2026-11-07) on the adapterless
# build VM: 126 adapter-skip lines beside 781 passed,
# 0 failed, 1 ignored (full survey and evidence:
# bench/gate-skip-visibility-2026-11-07.md). When the count
# drifts from EXPECTED_ADAPTER_SKIPS below on an adapterless
# machine, the wrapper still passes but prints the drift;
# update the enumeration in this header and the constant
# when the drift is reviewed, so T-1.7 inherits a current
# list instead of a stale one.
#
# Named exceptions (OTHER SKIPS: printed and counted, never
# gated). These fire by configuration or by device
# capability, not by adapter absence, so zero is not their
# expectation on any machine:
#
#   - env-arm routing pins in k21f_head_routing: each arm
#     prints a skip when the WGPU_COMPUTE_K21F_HEAD /
#     WGPU_COMPUTE_MMV_MAX_ROWS position it needs is not
#     the process position (3 lines at shipped defaults).
#   - diag capture arms that need an opt-in environment:
#     dflash_session's diag timing and capture tests
#     (DIAG_OUT / WGPU_COMPUTE_DIAG unset), resident_session's
#     diag smoke (WGPU_COMPUTE_DIAG_OUT unset) and diag
#     self-drain (WGPU_COMPUTE_DIAG not `timing`), and the
#     "this process runs with diag on" off-mode checks.
#   - device-capability gates, reachable only when an
#     adapter IS present: the k23/k24 "stock device (no
#     coopmat bridge)" and "device fails the kernel's
#     gates" prints, "timing requested but the device
#     granted ..." arms, and staged-flash / parity-leg
#     device refusals. On an adapter machine these are
#     reported in the other-skip count for a human to read;
#     the card sessions own their verdicts.
#
# Adapter detection: the wrapper must decide "could this
# machine have run them" with the SAME signal the tests
# use to skip. The tests skip when WgpuCompute::new()
# refuses: detect() (crates/wgpu-compute/src/detect.rs)
# enumerates Vulkan adapters through a wgpu instance and
# ranks them under DetectPolicy::default() — a discrete
# Vulkan adapter qualifies, an integrated or other adapter
# qualifies only under an explicit non-default policy, and
# a CPU-type adapter (llvmpipe) NEVER qualifies (BACKEND.md
# Section 6: a software adapter counts as no adapter). The
# wrapper probes with the crate's own readout example,
# adapter_features, which enumerates through the same wgpu
# Vulkan path and prints each adapter's facts; the machine
# counts as adapter-present when any printed adapter line
# carries device_type=DiscreteGpu, exactly the default
# policy's qualifying class. GATE_ASSUME_ADAPTER overrides
# the probe (see below) so the gate logic itself can be
# exercised on either class of machine.
#
# Usage:
#   bench/gate.sh
#
# Runs from the repo root (the script cd's there, like the
# other bench scripts). The suite's full combined output is
# tee'd to the terminal and to a log whose path is printed;
# the summary line names that log. The remaining RUNBOOK
# gates (clippy, fmt, cargo deny, the Go platform suite) are
# unchanged plain commands beside this wrapper; the wrapper
# wraps only the Rust test suite, which is where the skips
# live.
#
# Environment knobs:
#   GATE_ASSUME_ADAPTER  1: decide as if a qualifying
#                        adapter were present (the probe is
#                        not run). 0: decide as if none
#                        were. Unset: probe. Exists to test
#                        the gate logic; a forced decision
#                        is always named in the summary.
#
# Exit codes: 0 = gate passes; 1 = gate fails (tests
# failed, or adapter skips on an adapter machine, or the
# probe itself failed); 64 = usage error.
#
# shellcheck: NOT RUN. shellcheck is not installed on the
# build VM (checked 2026-11-07: `command -v shellcheck` is
# empty), so this script carries the repo bench style
# (set -u, everything quoted) without a shellcheck pass;
# `bash -n` was run clean. Run `shellcheck bench/gate.sh`
# on any host that has it before relying on a clean bill.

set -u

# Adapter-skip lines expected on an adapterless machine,
# measured at base 34358e3 (see the header enumeration).
EXPECTED_ADAPTER_SKIPS=126

usage() {
    echo "usage: bench/gate.sh" >&2
    exit 64
}

[ $# -eq 0 ] || usage

cd "$(dirname "$0")/.." || exit 64

adapter_present=""
adapter_source=""
case "${GATE_ASSUME_ADAPTER:-}" in
    1)
        adapter_present=1
        adapter_source="GATE_ASSUME_ADAPTER=1 (forced; probe not run)"
        ;;
    0)
        adapter_present=0
        adapter_source="GATE_ASSUME_ADAPTER=0 (forced; probe not run)"
        ;;
    "")
        probe_log=$(mktemp "${TMPDIR:-/tmp}/gate-probe.XXXXXX")
        if ! cargo run --quiet -p wgpu-compute --example adapter_features > "$probe_log" 2>/dev/null; then
            echo "gate: FAIL: the adapter probe (cargo run -p wgpu-compute --example adapter_features) did not run; the gate cannot establish whether this machine could have run the GPU tests" >&2
            echo "gate: probe output: $probe_log" >&2
            exit 1
        fi
        if grep -q 'device_type=DiscreteGpu' "$probe_log"; then
            adapter_present=1
        else
            adapter_present=0
        fi
        adapter_source="adapter_features probe ($probe_log)"
        ;;
    *)
        echo "gate: GATE_ASSUME_ADAPTER must be 0 or 1 when set (got: ${GATE_ASSUME_ADAPTER})" >&2
        usage
        ;;
esac

if [ "$adapter_present" -eq 1 ]; then
    echo "gate: adapter decision: PRESENT ($adapter_source)"
else
    echo "gate: adapter decision: ABSENT, no qualifying discrete Vulkan adapter ($adapter_source)"
fi

log=$(mktemp "${TMPDIR:-/tmp}/gate-test.XXXXXX")
echo "gate: running: cargo test --workspace -- --nocapture (full output: $log)"
cargo test --workspace -- --nocapture 2>&1 | tee "$log"
test_rc=${PIPESTATUS[0]}

counts=$(grep -E '^test result:' "$log" | awk '{for (i = 1; i <= NF; i++) {if ($i == "passed;") p += $(i - 1); if ($i == "failed;") f += $(i - 1); if ($i == "ignored;") g += $(i - 1)}} END {print p + 0, f + 0, g + 0}')
passed=${counts%% *}
rest=${counts#* }
failed=${rest%% *}
ignored=${rest##* }

adapter_skips=$(grep 'skipping' "$log" | grep -c 'no suitable adapter\|wgpu enumerates no adapters here' || true)
total_skips=$(grep -c 'skipping' "$log" || true)
other_skips=$((total_skips - adapter_skips))

verdict="PASS"
fail_reason=""
if [ "$test_rc" -ne 0 ] || [ "$failed" -ne 0 ]; then
    verdict="FAIL"
    fail_reason="cargo test exited $test_rc with $failed failed"
fi

if [ "$adapter_present" -eq 1 ]; then
    summary="gate: passed $passed, failed $failed, ignored $ignored; skipped $adapter_skips adapter (adapter present: expectation 0), $other_skips other (named exceptions, not gated)"
    if [ "$adapter_skips" -ne 0 ]; then
        verdict="FAIL"
        if [ -n "$fail_reason" ]; then
            fail_reason="$fail_reason; "
        fi
        fail_reason="${fail_reason}$adapter_skips adapter skips on a machine with a qualifying adapter"
        echo "gate: adapter skips observed (first 10):" >&2
        grep 'skipping' "$log" | grep 'no suitable adapter\|wgpu enumerates no adapters here' | head -10 >&2
    fi
else
    summary="gate: passed $passed, failed $failed, ignored $ignored; skipped $adapter_skips adapter (adapterless-expected: no qualifying adapter on this machine), $other_skips other (named exceptions, not gated)"
    if [ "$adapter_skips" -ne "$EXPECTED_ADAPTER_SKIPS" ]; then
        echo "gate: note: adapter skip count $adapter_skips differs from the enumerated expectation ($EXPECTED_ADAPTER_SKIPS at base 34358e3); when the drift is reviewed, update the enumeration in bench/gate.sh so T-1.7 inherits a current exception list"
    fi
fi

echo "$summary"
echo "gate: full output: $log"
if [ "$verdict" = "FAIL" ]; then
    echo "gate: FAIL: $fail_reason" >&2
    exit 1
fi
echo "gate: PASS"
exit 0
