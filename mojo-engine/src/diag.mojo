# diag.mojo - the Mojo engine's debug seam.
#
# Same shape as the Rust tree's diag seam: one env var,
# MOJO_ENGINE_DEBUG, enables the channel. The var is consulted on
# EVERY call (never cached at startup), so any binary or test that
# links this module can be switched at launch with no wiring and
# no rebuild. A present value enables, except "" and "0", which
# do not. While off, log lines are never printed and their message
# arguments are the caller's business alone, mirroring the Rust
# seam's no-format-while-off contract as closely as Mojo allows.
#
# Every emitted line is named: [mojo_engine][<phase>] <message>.

from std.os import getenv


def debug_value_enabled(value: String) -> Bool:
    """Pure predicate over a raw env value; the seam's whole rule.

    Split out from debug_enabled() so the rule itself is testable
    without mutating the process environment.
    """
    return value != "" and value != "0"


def debug_enabled() -> Bool:
    """True when MOJO_ENGINE_DEBUG currently enables the seam."""
    return debug_value_enabled(getenv("MOJO_ENGINE_DEBUG"))


def format_line(phase: String, msg: String) -> String:
    """The named line the seam emits for (phase, msg)."""
    return "[mojo_engine][" + phase + "] " + msg


def log_debug(phase: String, msg: String) -> String:
    """Emit one named debug line when the seam is on.

    Returns the emitted line, or "" when the seam is off (nothing
    printed). The return value is what the self-test asserts on:
    on fires (line returned and printed), off is silent.
    """
    if not debug_enabled():
        return ""
    var line = format_line(phase, msg)
    print(line)
    return line
