# serve_main.mojo - the Mojo-only API endpoint's accept loop
# (DESIGN.md step 7, decided 2026-11-08: the HTTP server
# lives inside the engine process, written in Mojo; no MAX
# Serve, no Go socket parity).
#
# Shape: load config, load the tokenizer, parse the served
# model's GGUF for its fact sheet + sha256 fingerprint,
# build the model ONCE (GPU when MOJO_ENGINE_GPU selects it,
# CPU otherwise - the same switch as generate_main), then
# accept one request per connection and dispatch:
#
#   GET  /health              engine/model/backend status
#   GET  /v1/models           registry: fact sheets for the
#                             served model + every .gguf in
#                             the models directory
#   GET  /v1/models/<id>      one fact sheet by id
#   POST /v1/chat/completions greedy completion through a
#                             manual generate() twin that
#                             splits prefill from decode for
#                             the telemetry (same picks, same
#                             seam), then records the run
#   POST /v1/attachments/text attachment text gate (stores
#                             nothing)
#   GET  /stats               serve counters
#   GET  /debug/log           the serve diag ring
#   GET  /openapi.json        the static contract
#   GET  /                    the testing chat dashboard
#   GET  /history             run history (disk, JSON lines;
#                             ?source= / ?model= filters)
#   GET  /history/export      run bundle (format_version 1,
#                             UNSIGNED in v1 - known limit)
#   GET  /leaderboard         measured rows per model +
#                             hardware + reference rows
#   POST /v1/runs/ingest      import a run bundle (every
#                             record validated by name)
#   GET  /mcp/tools           MCP-style bridge tool list
#   POST /mcp/call            invoke one bridge tool
#   POST /mcp                 JSON-RPC subset: initialize,
#                             tools/list, tools/call
#
# STATE BETWEEN REQUESTS: the model is stateful (its KV
# cache holds the conversation), and there is no reset verb
# on the ForwardModel seam, so EVERY chat request rebuilds
# the model fresh (the Wave C runner's construction, about
# 32 s on the card for the GPU path). That is the honest v1
# contract for a test bench: runs are independent and
# comparable, at the price of the rebuild. The rebuild cost
# is stated here and in the bench note, not hidden.
#
# Every run that hits the endpoint is recorded (amendments):
# dashboard, CLI and MCP sources alike, tag carried in the
# request body ("source", default cli; the dashboard sends
# dashboard; the MCP bridge forces mcp). A failed history
# append never fails the request; it is logged to the ring.
#
# Diag seam: the serve layer emits its own seam lines
# (listen, request, refusal, generated) through log_debug
# AND into the /debug/log ring; engine-internal seam lines
# still go to stdout only (stated limit).
#
# Usage (from the mojo-engine/ directory):
#   mojo run -I src src/runner/serve_main.mojo
# Exit 0 on a clean shutdown (no shutdown verb in v1: the
# loop runs until killed), exit 2 with a REFUSAL line when
# startup refuses (bad config, bad model file, bind failure).

from std.ffi import external_call
from std.memory import alloc
from std.os import getenv
from std.sys import exit
from std.time import perf_counter_ns

from config import ENGINE_VERSION, EngineConfig, parse_ipv4
from diag import debug_value_enabled, log_debug
from generate.generate import ForwardModel
from generate.sampler import greedy_argmax
from model.gpu_model import GpuGlimmerModel
from model.model import GlimmerForwardAdapter, GlimmerModel
from serve.dashboard import dashboard_html
from serve.factsheet import fact_sheet, fact_sheet_json, scan_model_files
from serve.history import (
    DEBUG_LEVEL_TRACE,
    MIB_BYTES,
    ZERO_RECORD_HASH,
    RunRecord,
    append_record,
    bundle_json,
    compute_record_sha256,
    effective_bandwidth_gbps,
    file_sha256_hex,
    history_json,
    ids_csv,
    leaderboard_json,
    load_records,
    parse_bundle,
    parse_run_record,
    record_chain_valid,
    utc_now_seconds,
)
from serve.http import (
    ROUTE_ATTACHMENT,
    ROUTE_CHAT,
    ROUTE_DASHBOARD,
    ROUTE_DEBUG_LOG,
    ROUTE_HEALTH,
    ROUTE_HISTORY,
    ROUTE_HISTORY_EXPORT,
    ROUTE_LEADERBOARD,
    ROUTE_MCP,
    ROUTE_MCP_CALL,
    ROUTE_MCP_TOOLS,
    ROUTE_METHOD_NOT_ALLOWED,
    ROUTE_MODEL_DETAIL,
    ROUTE_MODELS,
    ROUTE_NOT_FOUND,
    ROUTE_OPENAPI,
    ROUTE_RUNS_INGEST,
    ROUTE_STATS,
    build_http_response,
    parse_http_request,
    query_param,
    route_request,
)
from serve.json import (
    has_json_key,
    json_escape,
    json_int_field,
    json_raw_field,
    json_string_field,
)
from serve.openapi import openapi_json
from serve.server import (
    attachment_result_json,
    chat_response_json,
    effective_prompt,
    error_code_of,
    error_json,
    health_json,
    ingest_result_json,
    mcp_tools_json,
    model_display_name,
    parse_attachment_request,
    parse_chat_request,
)
from serve.socket import TcpListener, close_conn, read_request, send_all
from serve.stats import DebugRing, ServeStats
from tokenizer.gguf_vocab import load_tokenizer_data
from tokenizer.tokenizer import Tokenizer


def _use_gpu() -> Bool:
    return debug_value_enabled(getenv("MOJO_ENGINE_GPU"))


def _mkdirs(path: String):
    """Create each prefix of path with mkdir(2), ignoring
    EEXIST (17) and naming nothing: startup verifies the
    history file by appending a probe record only when the
    operator asks; the append path refuses by name."""
    var bytes = path.as_bytes()
    var i = 1
    while i <= len(bytes):
        if i == len(bytes) or bytes[i] == 47:
            var part = String(path[byte=0:i])
            var buf = alloc[UInt8](part.byte_length() + 1)
            var pb = part.as_bytes()
            for j in range(len(pb)):
                buf[j] = pb[j]
            buf[len(pb)] = UInt8(0)
            _ = external_call["mkdir", Int32](buf, Int(0o755))
            buf.free()
        i += 1


struct ServeState(Movable):
    var config: EngineConfig
    var tok: Tokenizer
    var model_name: String
    var model_id: String
    var gpu: Bool
    var fingerprint: String
    var fingerprint_note: String
    var model_file_bytes: Int
    var last_record_sha: String
    var stats: ServeStats
    var ring: DebugRing
    var seq: Int

    def __init__(
        out self,
        var config: EngineConfig,
        var tok: Tokenizer,
        fingerprint: String,
        fingerprint_note: String,
    ):
        self.model_name = model_display_name(config.model_path)
        self.model_id = self.model_name
        self.gpu = _use_gpu()
        self.fingerprint = fingerprint
        self.fingerprint_note = fingerprint_note
        # Model file size for the record's effective
        # bandwidth (bytes-per-step convention); 0 when the
        # header read fails, and the slot stays null then.
        self.model_file_bytes = 0
        try:
            self.model_file_bytes = fact_sheet(config.model_path).file_bytes
        except:
            self.model_file_bytes = 0
        # Hash chain head (zero-trust gate 3): the last
        # chained record already on disk, so a restart
        # continues the chain instead of forking it. The
        # zero hash when history is empty or pre-chain.
        self.last_record_sha = String(ZERO_RECORD_HASH)
        var existing = load_records(config.serve_history_dir)
        for i in range(len(existing)):
            if existing[i].record_sha256 != "":
                self.last_record_sha = existing[i].record_sha256
        self.stats = ServeStats()
        self.ring = DebugRing(config.serve_debug_ring_lines)
        self.seq = 0
        self.config = config^
        self.tok = tok^

    def note(mut self, phase: String, msg: String):
        """One serve seam line: through the diag seam (stdout
        when enabled) AND into the /debug/log ring."""
        _ = log_debug(phase, msg)
        self.ring.push(phase, msg)


struct GenOutcome(Copyable, Movable):
    var ids: List[Int]
    var prefill_ms: Int
    var decode_ms: Int
    var prefill_ns: Int
    var decode_ns: Int
    var decode_steps: Int
    var reserved_mib: Int
    var kv_mib: Int

    def __init__(
        out self,
        var ids: List[Int],
        prefill_ms: Int,
        decode_ms: Int,
        prefill_ns: Int,
        decode_ns: Int,
        decode_steps: Int,
        reserved_mib: Int,
        kv_mib: Int,
    ):
        self.ids = ids^
        self.prefill_ms = prefill_ms
        self.decode_ms = decode_ms
        self.prefill_ns = prefill_ns
        self.decode_ns = decode_ns
        self.decode_steps = decode_steps
        self.reserved_mib = reserved_mib
        self.kv_mib = kv_mib


def _generate_timed[M: ForwardModel](
    mut model: M,
    prompt_ids: List[Int],
    max_tokens: Int,
    eos_id: Int,
    reserved_mib: Int,
    kv_mib: Int,
) raises -> GenOutcome:
    """generate()'s exact loop with the prefill/decode split
    timed: identical picks (trait prefill, logits, greedy
    argmax, EOS excluded, never fed back) - the split exists
    because the run record needs prefill_ms and decode_tok_s
    separately, which generate() does not surface."""
    if len(prompt_ids) == 0:
        raise Error("generate_empty_prompt: prompt has no token ids")
    var n = len(prompt_ids)
    var t0 = perf_counter_ns()
    var hidden = model.prefill(prompt_ids)
    var t1 = perf_counter_ns()
    var out = List[Int]()
    var steps = 0
    while True:
        var row = model.logits(hidden)
        var next_id = greedy_argmax(row)
        if next_id == eos_id:
            break
        out.append(next_id)
        if len(out) >= max_tokens:
            break
        steps += 1
        hidden = model.step(next_id, n + len(out) - 1)
    var t2 = perf_counter_ns()
    return GenOutcome(
        out^,
        Int((t1 - t0) // 1_000_000),
        Int((t2 - t1) // 1_000_000),
        Int(t1 - t0),
        Int(t2 - t1),
        steps,
        reserved_mib,
        kv_mib,
    )


def _run_generation(
    st: ServeState, prompt_ids: List[Int], max_tokens: Int
) raises -> GenOutcome:
    """Build the model FRESH for this request (see the header
    on state), run the timed generation, and let the model
    die with the call."""
    ref cfg = st.config
    if st.gpu:
        # The 9-arg constructor with every knob from config,
        # exactly as the proven runner builds it (open_attn
        # would pin the GEMM batch knobs at their defaults and
        # silently drop the MOJO_ENGINE_PREFILL_* overrides).
        var model = GpuGlimmerModel(
            GlimmerModel.open(cfg.model_path)^,
            cfg.context_len,
            cfg.gpu_budget_bytes,
            cfg.attn_tile_positions,
            cfg.attn_max_splits,
            cfg.attn_split_span_positions,
            cfg.prefill_gemm_min_batch,
            cfg.gemm_tiled_min_batch,
            cfg.prefill_batch_min_tokens,
        )
        var reserved = model.rt.total_reserved()
        var kv_mib = model.kv.total_bytes // MIB_BYTES
        return _generate_timed(
            model,
            prompt_ids,
            max_tokens,
            st.tok.eos_id(),
            reserved // MIB_BYTES,
            kv_mib,
        )
    var cpu = GlimmerModel.open(cfg.model_path)
    var adapter = GlimmerForwardAdapter(cpu^)
    return _generate_timed(
        adapter, prompt_ids, max_tokens, st.tok.eos_id(), 0, 0
    )


def _do_chat(
    mut st: ServeState, body: String, source_override: String
) raises -> String:
    """One chat completion end to end: parse, tokenize,
    generate, answer, record. Raises named refusals; the
    caller renders them."""
    var req = parse_chat_request(
        body,
        st.config.serve_default_max_tokens,
        st.config.serve_max_tokens,
        st.config.serve_max_context_bytes,
    )
    var source = req.source
    if source_override != "":
        source = source_override
    var text = effective_prompt(req)
    var prompt_ids = List[Int]()
    prompt_ids.append(st.tok.bos_id())
    var body_ids = st.tok.encode(text)
    for i in range(len(body_ids)):
        prompt_ids.append(body_ids[i])
    # Trace mode, the cheap part only (addendum): the debug
    # stream gets the tokenized prompt ids before generation,
    # and one plain line about what trace does NOT emit in
    # this build. Top-5 logits and per-layer checksums need
    # device-side work and land with the kernel wave.
    if req.debug_level == DEBUG_LEVEL_TRACE:
        var ids_text = String("prompt_token_ids=[")
        for i in range(len(prompt_ids)):
            if i > 0:
                ids_text += ", "
            ids_text += String(prompt_ids[i])
        ids_text += "]"
        st.note("serve.trace", ids_text)
        st.note(
            "serve.trace",
            "trace limits: top-5 logits and per-layer checksums"
            " are not emitted by this build; they land with the"
            " kernel wave (goal Piece 4B)",
        )
    var t_start = perf_counter_ns()
    var outcome = _run_generation(st, prompt_ids, req.max_tokens)
    var total_ms = Int((perf_counter_ns() - t_start) // 1_000_000)
    var content = st.tok.decode_lossy(outcome.ids.copy())
    var decode_tok_s = Float64(0.0)
    if outcome.decode_ms > 0 and len(outcome.ids) > 0:
        decode_tok_s = (
            Float64(len(outcome.ids)) * 1000.0 / Float64(outcome.decode_ms)
        )
    st.seq += 1
    st.stats.note_chat(
        len(prompt_ids), len(outcome.ids), outcome.prefill_ms + outcome.decode_ms
    )
    st.stats.last_total_ms = total_ms
    var finish = "stop" if len(outcome.ids) < req.max_tokens else "length"
    var resp = chat_response_json(
        "chatcmpl-mojo-" + String(st.seq),
        utc_now_seconds(),
        st.model_name,
        content,
        finish,
        len(prompt_ids),
        len(outcome.ids),
        total_ms,
        outcome.prefill_ms,
        decode_tok_s,
        req.debug_level,
    )
    # Effective decode bandwidth (addendum): measured from
    # the model file size, the decode steps actually run,
    # and the decode wall time. The percentage is against
    # the CONFIG peak and only on GPU runs; on the CPU path
    # the percentage would compare against the wrong card,
    # so it stays null with the reason recorded.
    var bw = effective_bandwidth_gbps(
        st.model_file_bytes, outcome.decode_steps, outcome.decode_ms
    )
    var has_bw = bw > 0.0
    var bw_pct = Float64(0.0)
    var has_pct = False
    var bw_note = String("")
    if not has_bw:
        bw_note = String(
            "no decode steps measured; nothing to stream against"
        )
    elif st.gpu:
        bw_pct = bw / st.config.gpu_peak_bandwidth_gbps * 100.0
        has_pct = True
    else:
        bw_note = String(
            "bandwidth_pct_of_peak is measured against the"
            " configured GPU peak; this run used the CPU path"
        )
    # Hash chain (zero-trust gate 3): compute this record's
    # link at write time, inside the binary, over the build
    # sha from config, the model fingerprint, the exact
    # token ids, the nanosecond timings, and the previous
    # record's hash.
    var prompt_csv = ids_csv(prompt_ids)
    var gen_csv = ids_csv(outcome.ids)
    var rec_sha = compute_record_sha256(
        st.config.build_sha,
        st.fingerprint,
        prompt_csv,
        gen_csv,
        outcome.prefill_ns,
        outcome.decode_ns,
        st.last_record_sha,
    )
    # Record the run (amendments: every endpoint run, tagged).
    var rec = RunRecord(
        utc_now_seconds(),
        source,
        st.model_id,
        st.fingerprint,
        st.fingerprint_note,
        st.config.serve_backend_target,
        len(prompt_ids),
        outcome.prefill_ms,
        decode_tok_s,
        len(outcome.ids),
        outcome.reserved_mib,
        outcome.kv_mib,
        req.quality_verdict,
        req.attachments.copy(),
        False,
        String(""),
        debug_level=req.debug_level,
        eff_bandwidth_gbps=bw,
        has_eff_bandwidth=has_bw,
        bandwidth_pct_of_peak=bw_pct,
        has_bandwidth_pct=has_pct,
        bandwidth_note=bw_note,
        build_sha=st.config.build_sha,
        prompt_ids_csv=prompt_csv,
        generated_ids_csv=gen_csv,
        prefill_ns=outcome.prefill_ns,
        decode_ns=outcome.decode_ns,
        prev_record_sha256=st.last_record_sha,
        record_sha256=rec_sha,
    )
    try:
        append_record(st.config.serve_history_dir, rec)
        st.last_record_sha = rec_sha
    except e:
        st.note("serve.history", "run record append failed: " + String(e))
    st.note(
        "serve.chat",
        "source="
        + source
        + " prompt_tokens="
        + String(len(prompt_ids))
        + " completion_tokens="
        + String(len(outcome.ids))
        + " prefill_ms="
        + String(outcome.prefill_ms)
        + " total_ms="
        + String(total_ms),
    )
    return resp


def _registry_json(st: ServeState) -> String:
    """GET /v1/models body: OpenAI-shaped data list PLUS the
    fact-sheet registry (the dashboard reads `models`)."""
    var out = String('{"object":"list","data":[{')
    out += '"id":"' + json_escape(st.model_name) + '"'
    out += ',"object":"model","created":0,"owned_by":"mojo-engine"}]'
    out += ',"models":['
    var first = True
    # The served model first, from its own file.
    try:
        var sheet = fact_sheet(st.config.model_path)
        out += fact_sheet_json(sheet, True)
        first = False
    except e:
        out += (
            '{"path":"'
            + json_escape(st.config.model_path)
            + '","error":"'
            + json_escape(String(e))
            + '"}'
        )
        first = False
    # Every other .gguf one level under the models directory.
    try:
        var files = scan_model_files(st.config.serve_models_dir)
        for i in range(len(files)):
            if files[i] == st.config.model_path:
                continue
            if not first:
                out += ","
            first = False
            try:
                var sheet2 = fact_sheet(files[i])
                out += fact_sheet_json(sheet2, False)
            except e:
                out += (
                    '{"path":"'
                    + json_escape(files[i])
                    + '","error":"'
                    + json_escape(String(e))
                    + '"}'
                )
    except e:
        out += (
            ',{"path":"'
            + json_escape(st.config.serve_models_dir)
            + '","error":"'
            + json_escape(String(e))
            + '"}'
        )
    out += "]}"
    return out


def _model_detail_json(st: ServeState, want_id: String) raises -> String:
    var candidates = List[String]()
    candidates.append(st.config.model_path)
    try:
        var files = scan_model_files(st.config.serve_models_dir)
        for i in range(len(files)):
            candidates.append(files[i])
    except:
        pass
    for i in range(len(candidates)):
        if model_display_name(candidates[i]) == want_id:
            var sheet = fact_sheet(candidates[i])
            return fact_sheet_json(sheet, candidates[i] == st.config.model_path)
    raise Error(
        "serve_model_not_found: no model with id '" + want_id + "' in the registry"
    )


def _mcp_result_json(tool: String, result_body: String) -> String:
    var out = String("{")
    out += '"tool":"' + json_escape(tool) + '"'
    out += ',"result":' + result_body
    out += "}"
    return out


def _mcp_dispatch(
    mut st: ServeState, tool: String, body: String
) raises -> String:
    """The bridge's tool dispatch: same handlers as the HTTP
    endpoints; chat runs carry source mcp into history."""
    if tool == "health":
        return _mcp_result_json(
            tool, health_json(st.config.model_path, True, st.gpu)
        )
    if tool == "stats":
        return _mcp_result_json(tool, st.stats.to_json())
    if tool == "debug_log":
        return _mcp_result_json(tool, st.ring.to_json())
    if tool == "history":
        var records = load_records(st.config.serve_history_dir)
        return _mcp_result_json(tool, history_json(records, String(""), String("")))
    if tool == "leaderboard":
        var records2 = load_records(st.config.serve_history_dir)
        return _mcp_result_json(tool, leaderboard_json(records2))
    if tool == "chat":
        return _mcp_result_json(tool, _do_chat(st, body, String("mcp")))
    raise Error(
        "serve_mcp_unknown_tool: no tool named '"
        + tool
        + "' (chat, health, stats, debug_log, history, leaderboard)"
    )


def _mcp_call(mut st: ServeState, body: String) raises -> String:
    """POST /mcp/call body: {"tool": name, "arguments": {...}}
    or {"tool": name, "input": {...}}; the chat tool takes
    the chat body itself as its arguments."""
    var tool = json_string_field(body, "tool")
    var args = json_raw_field(body, "arguments")
    if args == "":
        args = json_raw_field(body, "input")
    if args == "":
        args = String("{}")
    return _mcp_dispatch(st, tool, args)


def _jsonrpc(mut st: ServeState, body: String) raises -> String:
    """The /mcp JSON-RPC subset: initialize, tools/list,
    tools/call. Notifications (no id) answer 200 with an
    empty object. Anything else refuses serve_mcp_method by
    name - the subset is stated, not implied."""
    var method = json_string_field(body, "method")
    var id_raw = json_raw_field(body, "id")
    if method == "initialize":
        var out = String("{")
        out += '"jsonrpc":"2.0","id":' + (id_raw if id_raw != "" else "null")
        out += ',"result":{"protocolVersion":"2024-11-05"'
        out += ',"capabilities":{"tools":{}}'
        out += ',"serverInfo":{"name":"mojo-engine","version":"'
        out += ENGINE_VERSION + '"}}}'
        return out
    if method == "tools/list":
        var tools = mcp_tools_json()
        var out2 = String("{")
        out2 += '"jsonrpc":"2.0","id":' + (id_raw if id_raw != "" else "null")
        out2 += ',"result":' + tools + "}"
        return out2
    if method == "tools/call":
        var params = json_raw_field(body, "params")
        var name = json_string_field(params, "name")
        var args = json_raw_field(params, "arguments")
        if args == "":
            args = String("{}")
        var result = _mcp_dispatch(st, name, args)
        var out3 = String("{")
        out3 += '"jsonrpc":"2.0","id":' + (id_raw if id_raw != "" else "null")
        out3 += ',"result":{"content":[{"type":"text","text":'
        out3 += '"' + json_escape(result) + '"}]'
        out3 += ',"isError":false}}'
        return out3
    if id_raw == "":
        return String("{}")
    raise Error(
        "serve_mcp_method: JSON-RPC method '"
        + method
        + "' is not in the v1 subset (initialize, tools/list, tools/call)"
    )


def _dispatch(mut st: ServeState, raw: String) -> String:
    """Route one raw request to its full HTTP response."""
    var status = 200
    var ctype = String("application/json")
    var body = String("")
    try:
        var req = parse_http_request(raw)
        st.stats.note_request()
        var route = route_request(req.method, req.path)
        if route == ROUTE_HEALTH:
            body = health_json(st.config.model_path, True, st.gpu)
            body = _with_serve_fields(body, st)
        elif route == ROUTE_MODELS:
            body = _registry_json(st)
        elif route == ROUTE_MODEL_DETAIL:
            var want = String(
                req.path[byte = 11 : req.path.byte_length()]
            )
            body = _model_detail_json(st, want)
        elif route == ROUTE_STATS:
            body = st.stats.to_json()
        elif route == ROUTE_OPENAPI:
            body = openapi_json(st.model_name)
        elif route == ROUTE_DASHBOARD:
            ctype = "text/html; charset=utf-8"
            body = dashboard_html(
                st.model_name,
                st.config.serve_attach_warn_tokens,
                st.config.serve_max_context_bytes,
            )
        elif route == ROUTE_DEBUG_LOG:
            body = st.ring.to_json()
        elif route == ROUTE_HISTORY:
            var records = load_records(st.config.serve_history_dir)
            body = history_json(
                records,
                query_param(req.query, "source"),
                query_param(req.query, "model"),
            )
        elif route == ROUTE_HISTORY_EXPORT:
            var records2 = load_records(st.config.serve_history_dir)
            body = bundle_json(records2, st.config.serve_backend_target)
        elif route == ROUTE_LEADERBOARD:
            var records3 = load_records(st.config.serve_history_dir)
            body = leaderboard_json(records3)
        elif route == ROUTE_RUNS_INGEST:
            body = _do_ingest(st, req.body)
        elif route == ROUTE_ATTACHMENT:
            var parsed = parse_attachment_request(
                req.body, st.config.serve_max_context_bytes
            )
            body = attachment_result_json(parsed[0], parsed[1])
            st.note("serve.attachment", "validated " + parsed[0])
        elif route == ROUTE_CHAT:
            body = _do_chat(st, req.body, String(""))
        elif route == ROUTE_MCP_TOOLS:
            body = mcp_tools_json()
        elif route == ROUTE_MCP_CALL:
            body = _mcp_call(st, req.body)
        elif route == ROUTE_MCP:
            body = _jsonrpc(st, req.body)
        elif route == ROUTE_METHOD_NOT_ALLOWED:
            status = 405
            body = error_json(
                "serve_method_not_allowed",
                req.method + " is not allowed on " + req.path,
            )
        else:
            status = 404
            body = error_json(
                "serve_route_not_found", "no route for " + req.path
            )
        if status != 200:
            st.stats.note_error()
            st.note(
                "serve.refusal",
                req.method + " " + req.path + " -> " + String(status),
            )
        else:
            st.note(
                "serve.request",
                req.method + " " + req.path + " -> 200",
            )
    except e:
        var msg = String(e)
        var code = error_code_of(msg)
        status = 422
        if code == "serve_body_too_large":
            status = 413
        elif code == "serve_http_bad_request":
            status = 400
        elif code == "serve_model_not_found":
            status = 404
        elif code == "serve_ingest_bad_bundle":
            status = 400
        st.stats.note_error()
        st.note("serve.refusal", code + ": " + msg)
        body = error_json(code, msg)
    return build_http_response(status, ctype, body)


def _with_serve_fields(body: String, st: ServeState) -> String:
    """Health carries the bench's published knobs the
    dashboard needs (attach warn threshold, context cap,
    backend target, sampler object form)."""
    var extra = String("")
    extra += ',"backend_target":"' + json_escape(
        st.config.serve_backend_target
    ) + '"'
    extra += ',"attach_warn_tokens":' + String(st.config.serve_attach_warn_tokens)
    extra += ',"max_context_bytes":' + String(st.config.serve_max_context_bytes)
    extra += ',"history_dir":"' + json_escape(st.config.serve_history_dir) + '"'
    # Splice before the closing brace of the health body.
    var trimmed = String(body[byte = 0 : body.byte_length() - 1])
    return trimmed + extra + "}"


def _do_ingest(mut st: ServeState, body: String) raises -> String:
    var records_raw = parse_bundle(body)
    var accepted = 0
    var rejected = 0
    var names = List[String]()
    for i in range(len(records_raw)):
        try:
            var rec = parse_run_record(records_raw[i], i)
            # Chain verification before acceptance (gate 3):
            # a record carrying a hash whose inputs do not
            # recompute is refused BY NAME and never stored.
            # Pre-chain records (no hash) land as before and
            # read back as unverifiable_pre_chain.
            if rec.record_sha256 != "" and not record_chain_valid(rec):
                raise Error(
                    "serve_ingest_tampered_record: record "
                    + String(i)
                    + " carries a record_sha256 that does not"
                    + " recompute from its chain inputs"
                )
            append_record(st.config.serve_history_dir, rec^)
            accepted += 1
        except e:
            rejected += 1
            names.append(String(e))
    st.note(
        "serve.ingest",
        "accepted=" + String(accepted) + " rejected=" + String(rejected),
    )
    return ingest_result_json(accepted, rejected, names)


def run() raises:
    var config = EngineConfig.load()
    _mkdirs(config.serve_history_dir)
    _ = log_debug(
        "serve",
        "loading tokenizer + fact sheet for " + config.model_path,
    )
    var data = load_tokenizer_data(config.model_path)
    _ = log_debug("serve", "tokenizer data loaded; building tokenizer")
    var tok = Tokenizer.build(data^)
    _ = log_debug("serve", "tokenizer built")
    # Fingerprint the served model file (sha256 over its
    # bytes) so its run records are attributable.
    var fingerprint = String("")
    var fingerprint_note = String("")
    try:
        fingerprint = file_sha256_hex(config.model_path)
        _ = log_debug("serve", "model fingerprint sha256=" + fingerprint)
    except e:
        fingerprint_note = "sha256 failed at serve start: " + String(e)
        _ = log_debug("serve", "fingerprint failed: " + String(e))
    var st = ServeState(config^, tok^, fingerprint, fingerprint_note)
    var octets = parse_ipv4(st.config.serve_host)
    var listener = TcpListener.bind_listen(octets, st.config.serve_port)
    st.note(
        "serve.listen",
        "mojo-engine "
        + ENGINE_VERSION
        + " serving "
        + st.model_name
        + " on "
        + st.config.serve_host
        + ":"
        + String(st.config.serve_port)
        + " backend="
        + ("gpu" if st.gpu else "cpu")
        + " history_dir="
        + st.config.serve_history_dir,
    )
    print(
        "SERVING",
        st.model_name,
        "on",
        st.config.serve_host + ":" + String(st.config.serve_port),
        "backend=" + ("gpu" if st.gpu else "cpu"),
    )
    while True:
        var cfd = listener.accept_conn()
        try:
            var raw = read_request(cfd, st.config.serve_max_body_bytes)
            var response = _dispatch(st, raw)
            send_all(cfd, response)
        except e:
            st.note("serve.refusal", "connection: " + String(e))
            try:
                send_all(
                    cfd,
                    build_http_response(
                        400,
                        "application/json",
                        error_json(error_code_of(String(e)), String(e)),
                    ),
                )
            except:
                pass
        close_conn(cfd)


def main():
    try:
        run()
    except e:
        print("REFUSAL", e)
        exit(2)
