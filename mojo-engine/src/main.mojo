# main.mojo - mojo-engine entry point.
#
# Foundation scope only: print the named hello (engine name and
# version), load the config, and refuse cleanly with the config's
# named error when the config is bad. The generation loop, the
# loader, and serving are later lanes (see DESIGN.md).

from std.sys import exit

from config import ENGINE_NAME, ENGINE_VERSION, EngineConfig
from diag import log_debug


def run() raises:
    print(ENGINE_NAME, ENGINE_VERSION)
    var config = EngineConfig.load()
    _ = log_debug("config", "model_path=" + config.model_path)
    _ = log_debug("config", "context_len=" + String(config.context_len))
    print("config ok: context_len", config.context_len)
    print("loader: not implemented yet (next lane, see DESIGN.md)")


def main():
    try:
        run()
    except e:
        print("REFUSAL", e)
        exit(2)
