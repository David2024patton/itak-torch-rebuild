# stats.mojo - the serve layer's in-memory counters and the
# debug ring. Owned by the accept loop in serve_main; nothing
# persists across a restart (stated in /stats itself). All
# times are perf_counter_ns readings converted at build time.

from std.time import perf_counter_ns

from diag import format_line
from serve.json import json_escape


struct ServeStats(Copyable, Movable):
    var started_ns: Int
    var requests_served: Int
    var chat_requests: Int
    var prompt_tokens: Int
    var completion_tokens: Int
    var errors: Int
    var last_generation_ms: Int
    var last_total_ms: Int

    def __init__(out self):
        self.started_ns = Int(perf_counter_ns())
        self.requests_served = 0
        self.chat_requests = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.errors = 0
        self.last_generation_ms = 0
        self.last_total_ms = 0

    def uptime_seconds(self) -> Int:
        return (Int(perf_counter_ns()) - self.started_ns) // 1_000_000_000

    def note_request(mut self):
        self.requests_served += 1

    def note_error(mut self):
        self.errors += 1

    def note_chat(
        mut self, prompt_tokens: Int, completion_tokens: Int, generation_ms: Int
    ):
        self.chat_requests += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.last_generation_ms = generation_ms

    def to_json(self) -> String:
        var out = String("{")
        out += '"requests_served":' + String(self.requests_served)
        out += ',"chat_requests":' + String(self.chat_requests)
        out += ',"prompt_tokens":' + String(self.prompt_tokens)
        out += ',"tokens_generated":' + String(self.completion_tokens)
        out += ',"completion_tokens":' + String(self.completion_tokens)
        out += ',"errors":' + String(self.errors)
        out += ',"uptime_seconds":' + String(self.uptime_seconds())
        out += ',"last_generation_ms":' + String(self.last_generation_ms)
        out += ',"last_total_ms":' + String(self.last_total_ms)
        out += ',"persistence":"in-memory only; counters reset on restart"'
        out += "}"
        return out


struct DebugRing(Copyable, Movable):
    """The recent diag lines, oldest first, capped at the
    configured capacity (serve_debug_ring_lines). The serve
    layer records its own seam-shaped lines here (listen,
    request, refusal, generation) AND emits them through the
    diag seam via log_debug; engine-internal seam lines still
    go to the process stdout only - that limit is stated in
    the bench note, not hidden."""

    var lines: List[String]
    var capacity: Int

    def __init__(out self, capacity: Int):
        self.lines = List[String]()
        self.capacity = capacity if capacity > 0 else 1

    def push(mut self, phase: String, msg: String):
        self.lines.append(format_line(phase, msg))
        while len(self.lines) > self.capacity:
            _ = self.lines.pop(0)

    def to_json(self) -> String:
        var out = String("{")
        out += '"capacity":' + String(self.capacity)
        out += ',"count":' + String(len(self.lines))
        out += ',"lines":['
        for i in range(len(self.lines)):
            if i > 0:
                out += ","
            out += '"' + json_escape(self.lines[i]) + '"'
        out += "]}"
        return out
