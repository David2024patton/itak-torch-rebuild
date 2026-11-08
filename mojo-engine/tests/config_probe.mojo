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
        print("attn_tile_positions", cfg.attn_tile_positions)
        print("attn_max_splits", cfg.attn_max_splits)
        print("attn_split_span_positions", cfg.attn_split_span_positions)
        # Wave E Lane E3: Glimmer op defaults folded from src/ops/.
        print("rms_eps", cfg.rms_eps)
        print("rope_base", cfg.rope_base)
        print("logit_scale", cfg.logit_scale)
        print("logit_softcap", cfg.logit_softcap)
        print("head_dim", cfg.head_dim)
        print("hidden", cfg.hidden)
        print("vocab", cfg.vocab)
    except e:
        print("REFUSAL", e)
        exit(2)
