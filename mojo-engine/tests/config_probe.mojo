# config_probe.mojo - process-level config check for the runner.
#
# Not a `mojo test` file: tests/run_tests.sh runs this program
# with chosen env vars and asserts on its output and exit code.
# It prints the loaded config, or the named REFUSAL and exit 2
# when the config is bad, exactly as src/main.mojo behaves.

from std.sys import exit

from config import EngineConfig


def main():
    try:
        var cfg = EngineConfig.load()
        print("model_path", cfg.model_path)
        print("context_len", cfg.context_len)
        print("gpu_budget_bytes", cfg.gpu_budget_bytes)
    except e:
        print("REFUSAL", e)
        exit(2)
