#!/usr/bin/env bash
# run_tests.sh - the mojo-engine test gate.
#
# Runs the toolchain's own runner (std.testing.TestSuite
# via each test file's main under `mojo run`) over tests/ in a
# clean environment, then again with the debug seam ON (the diag
# suite asserts both modes against the same code), then drives
# the config probe at process level for the env-override and
# bad-value cases. Any failure exits nonzero.
# shellcheck: NOT RUN on the authoring VM (shellcheck absent
# there, same standing note as bench/gate.sh); bash -n clean.
set -u
cd "$(dirname "$0")/.."

fail() { echo "run_tests: FAIL: $*" >&2; exit 1; }

# Wave E hygiene (Lane E3, item 2): the timing harness verdict-line
# extraction self-test (synthetic logs; pure shell, no toolchain).
bash tests/wave_c_timing_extract_test.sh >/dev/null 2>&1 \
    || fail "wave_c verdict extraction self-test"

# Mojo 1.2 has no `mojo test` subcommand; the toolchain's own
# runner is std.testing.TestSuite, discovered and run by each
# test file's main() under `mojo run`.
echo "== TestSuite files (clean env, seam off) =="
for t in tests/test_diag.mojo tests/test_config.mojo tests/test_k1_cpu.mojo tests/test_loader.mojo tests/test_loader_full.mojo tests/test_q6k_cpu.mojo tests/test_q5k_cpu.mojo tests/test_f32_cpu.mojo tests/test_ops_rms_norm.mojo tests/test_ops_rope.mojo tests/test_ops_swiglu_residual.mojo tests/test_ops_softmax_logit.mojo tests/test_ops_embedding.mojo tests/test_k1_gpu_helpers.mojo tests/test_q6k_gpu_helpers.mojo tests/test_q5k_gpu_helpers.mojo tests/test_f32_gpu_helpers.mojo tests/test_tokenizer.mojo tests/test_tokenizer_gguf.mojo tests/test_kv_cache.mojo tests/test_attention_cpu.mojo tests/test_attention_split_cpu.mojo tests/test_generate.mojo tests/test_model_layer.mojo tests/test_gpu_plumbing.mojo tests/test_gpu_layer_vm.mojo tests/test_gpu_embedding_vm.mojo tests/test_gemm_cpu.mojo tests/test_gpu_prefill_vm.mojo tests/test_wave_e_batch_vm.mojo tests/test_attention_direct_cpu.mojo; do
    env -u MOJO_ENGINE_DEBUG -u MOJO_ENGINE_MODEL_PATH \
        -u MOJO_ENGINE_CONTEXT_LEN \
        mojo run -I src "$t" || fail "TestSuite $t (seam off)"
done

echo "== TestSuite diag (MOJO_ENGINE_DEBUG=1, seam on) =="
env MOJO_ENGINE_DEBUG=1 \
    mojo run -I src tests/test_diag.mojo || fail "TestSuite diag (seam on)"

echo "== config probe: defaults =="
out=$(env -u MOJO_ENGINE_MODEL_PATH -u MOJO_ENGINE_CONTEXT_LEN \
    mojo run -I src tests/config_probe.mojo) || fail "probe defaults"
echo "$out"
echo "$out" | grep -q "context_len 131072" || fail "probe default ctx"

echo "== config probe: env override =="
out=$(env MOJO_ENGINE_CONTEXT_LEN=4096 MOJO_ENGINE_MODEL_PATH=/tmp/m.gguf \
    mojo run -I src tests/config_probe.mojo) || fail "probe override"
echo "$out"
echo "$out" | grep -q "context_len 4096" || fail "probe override ctx"
echo "$out" | grep -q "model_path /tmp/m.gguf" || fail "probe override path"

echo "== config probe: bad value refused by name =="
if out=$(env MOJO_ENGINE_CONTEXT_LEN=abc \
    mojo run -I src tests/config_probe.mojo 2>&1); then
    fail "probe bad value exited 0"
fi
echo "$out"
echo "$out" | grep -q "REFUSAL mojo_engine_config" \
    || fail "probe bad value refusal unnamed"

echo "== engine hello =="
out=$(env -u MOJO_ENGINE_MODEL_PATH -u MOJO_ENGINE_CONTEXT_LEN \
    mojo run -I src src/main.mojo) || fail "engine hello"
echo "$out"
echo "$out" | grep -q "mojo-engine 1.6" || fail "hello missing name/version"

echo "run_tests: PASS"
