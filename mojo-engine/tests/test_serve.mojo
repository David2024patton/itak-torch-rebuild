# test_serve.mojo - the serve layer's suite (step 7 +
# amendments 2026-11-08): JSON reader/writer, HTTP parse +
# routing, chat parse with the greedy lock, stats + ring,
# the OpenAPI/route-table step check, the run-record schema
# (bucket rule, round trip, ingest validation by name), the
# bundle envelope, history filtering, the leaderboard shape
# (measured rows per model + hardware, reference rows marked
# and separate), and the fact sheet over the tiny fixture.
# All pure: no sockets, no model, no files except the tiny
# fixture GGUF under tests/fixtures/.

from std.testing import TestSuite, assert_true, assert_false, assert_equal

from gguf.loader import parse_gguf
from serve.dashboard import dashboard_html
from serve.factsheet import (
    compute_split,
    fact_sheet,
    fact_sheet_json,
    model_id_of,
)
from serve.history import (
    RunRecord,
    bundle_json,
    history_json,
    leaderboard_json,
    load_records,
    parse_bundle,
    parse_run_record,
    prompt_bucket_of,
    reference_rows,
)
from serve.http import (
    ROUTE_CHAT,
    ROUTE_HISTORY,
    ROUTE_LEADERBOARD,
    ROUTE_METHOD_NOT_ALLOWED,
    ROUTE_MODEL_DETAIL,
    ROUTE_NOT_FOUND,
    ROUTE_RUNS_INGEST,
    build_http_response,
    parse_http_request,
    query_param,
    route_request,
)
from serve.json import (
    json_escape,
    json_float_field,
    json_string_array_field,
    json_string_field,
    parse_json_string_at,
    split_json_array,
)
from serve.openapi import openapi_json
from serve.server import (
    chat_response_json,
    error_code_of,
    health_json,
    mcp_tools_json,
    model_display_name,
    parse_attachment_request,
    parse_chat_request,
)
from serve.stats import DebugRing, ServeStats

comptime TINY_FIXTURE = "tests/fixtures/tiny.gguf"


def _refusal_name(body: String) -> String:
    """Run parse_chat_request expecting a refusal; return its
    name (the token before the first ':')."""
    try:
        _ = parse_chat_request(body, 64, 512, 32768)
    except e:
        return error_code_of(String(e))
    return String("NO_REFUSAL")


# -- json.mojo ---------------------------------------------------------------


def test_json_escape_and_parse() raises:
    assert_equal(json_escape('a"b\\c\nd'), 'a\\"b\\\\c\\nd')
    var parsed = parse_json_string_at('"hi \\u0041\\uD83D\\uDE00"', 0)
    assert_equal(parsed[0], "hi A\U0001F600")
    var fields = '{"prompt": "x", "max_tokens": 7, "t": 0.5}'
    assert_equal(json_string_field(fields, "prompt"), "x")
    assert_true(abs(json_float_field(fields, "t", 0.0) - 0.5) < 1e-9)
    var arr = json_string_array_field('{"a":["x","y"]}', "a")
    assert_equal(len(arr), 2)
    assert_equal(arr[1], "y")


def test_split_json_array() raises:
    var parts = split_json_array('[{"a":1},{"b":[2,3]}, 7, "x]y"]')
    assert_equal(len(parts), 4)
    assert_equal(parts[3], '"x]y"')
    var empty = split_json_array(" [ ] ")
    assert_equal(len(empty), 0)


# -- http.mojo ---------------------------------------------------------------


def test_http_parse_and_query() raises:
    var raw = (
        "GET /history?source=mcp&model=foo HTTP/1.1\r\nHost: x\r\n\r\n"
    )
    var req = parse_http_request(raw)
    assert_equal(req.method, "GET")
    assert_equal(req.path, "/history")
    assert_equal(query_param(req.query, "source"), "mcp")
    assert_equal(query_param(req.query, "model"), "foo")
    assert_equal(query_param(req.query, "absent"), "")
    var resp = build_http_response(200, "application/json", "{}")
    assert_true(resp.startswith("HTTP/1.1 200 OK"))


def test_routing() raises:
    assert_equal(route_request("POST", "/v1/chat/completions"), ROUTE_CHAT)
    assert_equal(route_request("GET", "/history"), ROUTE_HISTORY)
    assert_equal(route_request("GET", "/leaderboard"), ROUTE_LEADERBOARD)
    assert_equal(route_request("POST", "/v1/runs/ingest"), ROUTE_RUNS_INGEST)
    assert_equal(route_request("GET", "/v1/models/abc"), ROUTE_MODEL_DETAIL)
    assert_equal(
        route_request("DELETE", "/health"), ROUTE_METHOD_NOT_ALLOWED
    )
    assert_equal(route_request("GET", "/nope"), ROUTE_NOT_FOUND)


def test_openapi_covers_routes() raises:
    var spec = openapi_json("SomeModel")
    for path in [
        "/health",
        "/v1/models",
        "/v1/models/{id}",
        "/v1/chat/completions",
        "/v1/attachments/text",
        "/stats",
        "/debug/log",
        "/openapi.json",
        "/",
        "/mcp/tools",
        "/mcp/call",
        "/mcp",
        "/history",
        "/history/export",
        "/v1/runs/ingest",
        "/leaderboard",
    ]:
        assert_true(spec.find(path) != -1)


# -- server.mojo: chat parse + greedy lock -----------------------------------


def test_chat_parse_shapes() raises:
    var a = parse_chat_request('{"prompt":"hello"}', 64, 512, 32768)
    assert_equal(a.prompt_text, "hello")
    assert_equal(a.max_tokens, 64)
    assert_equal(a.source, "cli")
    var b = parse_chat_request(
        '{"messages":[{"role":"user","content":"one"},'
        '{"role":"user","content":"two"}],"max_tokens":9,'
        '"source":"dashboard","attachments":["a.txt"],'
        '"quality_verdict":"golden PASS"}',
        64,
        512,
        32768,
    )
    assert_equal(b.prompt_text, "one\ntwo")
    assert_equal(b.max_tokens, 9)
    assert_equal(b.source, "dashboard")
    assert_equal(len(b.attachments), 1)
    assert_equal(b.quality_verdict, "golden PASS")


def test_greedy_lock() raises:
    # temperature 0.0 is the lock's own value: accepted.
    var ok = parse_chat_request(
        '{"prompt":"x","temperature":0.0}', 64, 512, 32768
    )
    assert_equal(ok.prompt_text, "x")
    assert_equal(
        _refusal_name('{"prompt":"x","temperature":0.7}'),
        "serve_sampler_not_available",
    )
    assert_equal(
        _refusal_name('{"prompt":"x","top_k":40}'),
        "serve_sampler_not_available",
    )
    assert_equal(
        _refusal_name('{"prompt":"x","top_p":0.9}'),
        "serve_sampler_not_available",
    )
    assert_equal(
        _refusal_name('{"prompt":"x","seed":42}'),
        "serve_sampler_not_available",
    )


def test_chat_refusals_by_name() raises:
    assert_equal(
        _refusal_name('{"prompt":"x","image":"data:..."}'),
        "vision_not_ready",
    )
    assert_equal(_refusal_name('{"max_tokens":3}'), "serve_missing_field")
    assert_equal(_refusal_name('{"prompt":""}'), "serve_empty_prompt")
    assert_equal(
        _refusal_name('{"prompt":"x","max_tokens":9999}'),
        "serve_max_tokens_exceeded",
    )
    assert_equal(
        _refusal_name('{"prompt":"x","source":"smoke-signal"}'),
        "serve_bad_json",
    )
    assert_equal(
        _refusal_name(
            '{"prompt":"x","context_text":"' + "a" * 40000 + '"}'
        ),
        "serve_context_too_large",
    )


def test_response_and_health_shapes() raises:
    var h = health_json("/tmp/Some-Model.gguf", True, True)
    assert_true(h.find('"sampler":{"name":"greedy","temperature":0.0') != -1)
    assert_true(h.find('"backend":"gpu"') != -1)
    var r = chat_response_json(
        "chatcmpl-mojo-1", 1700000000, "M", " Paris.", "stop", 6, 4, 2000, 900, 3.5
    )
    assert_true(r.find('"prefill_ms":900') != -1)
    assert_true(r.find('"ttft_ms":null') != -1)
    assert_true(r.find('"temperature":0.0') != -1)
    assert_equal(model_display_name("/a/b/Qwen3.5-9B.gguf"), "Qwen3.5-9B")


def test_attachment_gate() raises:
    var ok = parse_attachment_request(
        '{"filename":"notes.txt","text":"hi"}', 32768
    )
    assert_equal(ok[0], "notes.txt")
    assert_equal(ok[1], "hi")
    try:
        _ = parse_attachment_request(
            '{"filename":"evil.exe","text":"hi"}', 32768
        )
        assert_true(False)
    except e:
        assert_equal(error_code_of(String(e)), "serve_attachment_type")


# -- stats.mojo ---------------------------------------------------------------


def test_stats_and_ring() raises:
    var st = ServeStats()
    st.note_request()
    st.note_chat(6, 4, 900)
    var j = st.to_json()
    assert_true(j.find('"chat_requests":1') != -1)
    assert_true(j.find('"tokens_generated":4') != -1)
    var ring = DebugRing(3)
    for i in range(5):
        ring.push("serve", "line " + String(i))
    assert_equal(len(ring.lines), 3)
    assert_true(ring.to_json().find("line 4") != -1)
    assert_false(ring.to_json().find("line 0") != -1)


# -- history.mojo: schema, buckets, ingest, leaderboard -----------------------


def _sample_record() -> RunRecord:
    var atts = List[String]()
    atts.append("notes.txt")
    return RunRecord(
        1700000000,
        "dashboard",
        "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M",
        "ab12",
        "",
        "gfx1100 / MAX 26.7",
        1022,
        34852,
        20.35,
        16,
        19968,
        3484,
        "golden PASS",
        atts^,
        False,
        "",
    )


def test_bucket_rule() raises:
    assert_equal(prompt_bucket_of(6), 128)
    assert_equal(prompt_bucket_of(576), 128)
    assert_equal(prompt_bucket_of(577), 1024)
    assert_equal(prompt_bucket_of(1022), 1024)
    assert_equal(prompt_bucket_of(2560), 1024)
    assert_equal(prompt_bucket_of(2561), 4096)
    assert_equal(prompt_bucket_of(131072), 4096)


def test_record_round_trip() raises:
    var rec = _sample_record()
    var j = rec.to_json()
    assert_true(j.find('"prompt_bucket":1024') != -1)
    assert_true(j.find('"ttft_ms":null') != -1)
    assert_true(j.find('"peak_vram_mib":null') != -1)
    assert_true(j.find('"sampler":{"name":"greedy","temperature":0.0') != -1)
    assert_true(abs(rec.prefill_tok_s - 29.32) < 0.05)
    var back = parse_run_record(j, 0)
    assert_equal(back.model, rec.model)
    assert_equal(back.prompt_tokens, 1022)
    assert_true(back.imported)
    assert_equal(back.origin_backend, "gfx1100 / MAX 26.7")
    assert_equal(len(back.attachments), 1)


def test_ingest_validation_by_name() raises:
    try:
        _ = parse_run_record('{"timestamp":1,"source":"dashboard"}', 3)
        assert_true(False)
    except e:
        assert_equal(error_code_of(String(e)), "serve_ingest_bad_record")
    try:
        _ = parse_run_record(
            '{"timestamp":1,"source":"smoke","model":"m",'
            '"backend_target":"b","prompt_tokens":1,"prefill_ms":1,'
            '"decode_tok_s":1.0,"tokens_generated":1}',
            0,
        )
        assert_true(False)
    except e:
        assert_equal(error_code_of(String(e)), "serve_ingest_bad_record")
    # bundle envelope
    try:
        _ = parse_bundle('{"format":"other","format_version":1,"runs":[]}')
        assert_true(False)
    except e:
        assert_equal(error_code_of(String(e)), "serve_ingest_bad_bundle")
    var good = parse_bundle(
        '{"format":"mojo-engine-run-bundle","format_version":1,"runs":[]}'
    )
    assert_equal(len(good), 0)


def test_history_filter_and_bundle() raises:
    var records = List[RunRecord]()
    records.append(_sample_record())
    var other = _sample_record()
    other.source = "mcp"
    other.model = "OtherModel"
    records.append(other^)
    var all = history_json(records, "", "")
    assert_true(all.find('"count":2') != -1)
    var only_mcp = history_json(records, "mcp", "")
    assert_true(only_mcp.find('"count":1') != -1)
    var bundle = bundle_json(records, "gfx1100 / MAX 26.7")
    assert_true(bundle.find('"format_version":1') != -1)
    assert_true(bundle.find('"signed":false') != -1)
    var round = parse_bundle(bundle)
    assert_equal(len(round), 2)


def test_leaderboard_shape() raises:
    var records = List[RunRecord]()
    records.append(_sample_record())
    var foreign = _sample_record()
    foreign.backend_target = "sm_89 / MAX 26.7"
    foreign.decode_tok_s = 55.0
    foreign.imported = True
    records.append(foreign^)
    var board = leaderboard_json(records)
    # per model + hardware: never merged across cards
    assert_true(board.find("sm_89 / MAX 26.7") != -1)
    assert_true(board.find('"kind":"reference"') != -1)
    assert_true(board.find("44.59") != -1)
    assert_equal(len(reference_rows()), 2)
    var empty_dir = load_records("/tmp/nonexistent-mojo-serve-history-dir")
    assert_equal(len(empty_dir), 0)


# -- factsheet.mojo ------------------------------------------------------------


def test_fact_sheet_tiny_fixture() raises:
    var gf = parse_gguf(TINY_FIXTURE)
    assert_equal(gf.architecture(), "llama")
    assert_equal(model_id_of("/x/y/tiny.gguf"), "tiny")
    var sheet = fact_sheet(TINY_FIXTURE)
    assert_equal(sheet.architecture, "llama")
    assert_equal(sheet.block_count, 1)
    assert_false(sheet.vision_present)
    var j = fact_sheet_json(sheet, False)
    assert_true(j.find('"architecture":"llama"') != -1)
    assert_true(j.find('"vision":{') != -1)
    var split = compute_split(gf, "llama", 1)
    assert_equal(split.other_layers, 1)


# -- dashboard ------------------------------------------------------------------


def test_dashboard_wiring() raises:
    var page = dashboard_html("SomeModel", 4096, 32768)
    for marker in [
        "/v1/chat/completions",
        "/debug/log",
        "/v1/models",
        "/leaderboard",
        "/history",
        "--- FILE: ",
        "estimateTokens",
        "vision",
        "data-dl",
        "temperature:0.0",
    ]:
        assert_true(page.find(marker) != -1)


def test_mcp_tools_list() raises:
    var tools = mcp_tools_json()
    for name in ["chat", "health", "stats", "debug_log", "history", "leaderboard"]:
        assert_true(tools.find('"name":"' + name + '"') != -1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
