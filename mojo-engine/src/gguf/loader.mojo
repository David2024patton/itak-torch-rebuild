# loader.mojo - GGUF loader interface (NOT implemented).
#
# The real loader is the next lane on this branch (see DESIGN.md
# build order, step 1). This module fixes the interface now so
# callers and tests have a stable seam, and so the absence of a
# loader is a named refusal, never a silent stub.

from diag import log_debug


def loader_state() -> String:
    """The loader's implementation state, for hello/smoke output."""
    return "not_implemented"


def load_model(path: String) raises -> None:
    """Load the GGUF model at `path`. Not implemented yet.

    Raises the named refusal gguf_loader_not_implemented until the
    loader lane lands.
    """
    _ = log_debug("gguf", "load_model called for " + path)
    raise Error(
        "gguf_loader_not_implemented: the GGUF loader is the next"
        " lane on the mojo-max-engine branch (DESIGN.md step 1);"
        " refusing to pretend "
        + path
        + " was loaded"
    )
