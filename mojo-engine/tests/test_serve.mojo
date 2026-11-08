# test_serve.mojo - the serve layer's suite (step 7 +
# amendments 2026-11-08): JSON reader/writer, HTTP parse +
# routing, chat parse with the greedy lock, stats + ring,
# the OpenAPI/route-table step check, the run-record schema
# (bucket rule, round trip, ingest validation by name), the
# bundle envelope, history filtering, the leaderboard shape
# (measured rows per model + hardware, reference rows marked
# and separate), and the fact sheet over the tiny fixture.
# All pure: no sockets, no model; the only files are the tiny
# fixture GGUF under tests/fixtures/ and two throwaway hash
# inputs the SHA file test writes under /tmp.

from std.testing import TestSuite, assert_true, assert_false, assert_equal

from gguf.loader import parse_gguf
from config import (
    DEFAULT_GPU_PEAK_BANDWIDTH_GBPS,
    parse_gpu_peak_bandwidth_gbps,
)
from serve.dashboard import dashboard_html
from serve.factsheet import (
    compute_split,
    fact_sheet,
    fact_sheet_json,
    model_id_of,
)
from serve.history import (
    ZERO_RECORD_HASH,
    RunRecord,
    bundle_json,
    chain_statuses,
    compute_record_sha256,
    effective_bandwidth_gbps,
    file_sha256_hex,
    history_json,
    ids_csv,
    leaderboard_json,
    load_records,
    parse_bundle,
    parse_run_record,
    prompt_bucket_of,
    record_chain_valid,
    reference_rows,
    sha256_hex_of_bytes,
    valid_debug_level,
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
    # The integer form 0 is the same zero (David's
    # clarification, 2026-11-08): standard OpenAI clients and
    # the MCP bridge may send either JSON form; both accepted.
    var ok_int = parse_chat_request(
        '{"prompt":"x","temperature":0}', 64, 512, 32768
    )
    assert_equal(ok_int.prompt_text, "x")
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


def test_debug_level_parse() raises:
    # Absent means benchmark, the default.
    var a = parse_chat_request('{"prompt":"x"}', 64, 512, 32768)
    assert_equal(a.debug_level, "benchmark")
    var b = parse_chat_request(
        '{"prompt":"x","debug_level":"profile"}', 64, 512, 32768
    )
    assert_equal(b.debug_level, "profile")
    var c = parse_chat_request(
        '{"prompt":"x","debug_level":"trace"}', 64, 512, 32768
    )
    assert_equal(c.debug_level, "trace")
    # Unknown values refuse BY NAME, never coerced.
    assert_equal(
        _refusal_name('{"prompt":"x","debug_level":"verbose"}'),
        "serve_bad_debug_level",
    )
    assert_true(valid_debug_level("benchmark"))
    assert_true(valid_debug_level("profile"))
    assert_true(valid_debug_level("trace"))
    assert_false(valid_debug_level(""))


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
    assert_true(r.find('"debug_level":"benchmark"') != -1)
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


def test_bandwidth_math() raises:
    # 1 GB streamed per step, 10 steps, 100 ms: 100 GB/s.
    assert_true(
        abs(effective_bandwidth_gbps(1_000_000_000, 10, 100) - 100.0) < 0.01
    )
    # Nothing measured means no number, never a guess.
    assert_equal(effective_bandwidth_gbps(0, 10, 100), 0.0)
    assert_equal(effective_bandwidth_gbps(1_000_000_000, 0, 100), 0.0)
    assert_equal(effective_bandwidth_gbps(1_000_000_000, 10, 0), 0.0)


def test_telemetry_slots_round_trip() raises:
    # A record with measured bandwidth and a trace level
    # keeps both through JSON and back through ingest.
    var atts = List[String]()
    var rec = RunRecord(
        1700000000,
        "dashboard",
        "M",
        "ab12",
        "",
        "gfx1100 / MAX 26.7",
        128,
        200,
        32.65,
        16,
        16739,
        260,
        "",
        atts^,
        False,
        "",
        debug_level="trace",
        eff_bandwidth_gbps=512.5,
        has_eff_bandwidth=True,
        bandwidth_pct_of_peak=53.4,
        has_bandwidth_pct=True,
    )
    var j = rec.to_json()
    assert_true(j.find('"debug_level":"trace"') != -1)
    assert_true(j.find('"effective_bandwidth_gbps":512.5') != -1)
    assert_true(j.find('"bandwidth_pct_of_peak":53.4') != -1)
    # GPU clock/temp/power are null with the named reason,
    # and the per-op buckets are null with theirs.
    assert_true(j.find('"gpu_clock_mhz_start":null') != -1)
    assert_true(j.find('"gpu_power_w_end":null') != -1)
    assert_true(j.find("not exposed by MAX 26.7") != -1)
    assert_true(j.find('"op_timings_ms":{"embedding":null') != -1)
    assert_true(j.find("kernel wave (goal Piece 4B)") != -1)
    var back = parse_run_record(j, 0)
    assert_equal(back.debug_level, "trace")
    assert_true(back.has_eff_bandwidth)
    assert_true(abs(back.eff_bandwidth_gbps - 512.5) < 0.01)
    assert_true(back.has_bandwidth_pct)
    assert_true(abs(back.bandwidth_pct_of_peak - 53.4) < 0.01)
    assert_false(back.has_gpu_telemetry)
    assert_false(back.has_op_timings)
    # An OLD record (no addendum keys) still ingests, with
    # the default level and null slots.
    var old = parse_run_record(_sample_record().to_json(), 0)
    assert_equal(old.debug_level, "benchmark")
    assert_false(old.has_eff_bandwidth)
    assert_false(old.has_bandwidth_pct)


def _chained_record(
    prev: String, prompt_csv: String, gen_csv: String, dec_ns: Int
) -> RunRecord:
    """A record the way serve_main writes one: hash
    computed at write time over its chain inputs."""
    var sha = compute_record_sha256(
        "testbuild", "ab12", prompt_csv, gen_csv, 1000, dec_ns, prev
    )
    var atts = List[String]()
    return RunRecord(
        1700000000,
        "dashboard",
        "M",
        "ab12",
        "",
        "gfx1100 / MAX 26.7",
        6,
        200,
        32.65,
        4,
        16739,
        260,
        "",
        atts^,
        False,
        "",
        build_sha="testbuild",
        prompt_ids_csv=prompt_csv,
        generated_ids_csv=gen_csv,
        prefill_ns=1000,
        decode_ns=dec_ns,
        prev_record_sha256=prev,
        record_sha256=sha,
    )


def test_record_hash_chain() raises:
    assert_equal(ids_csv(List[Int]()), "")
    var two = List[Int]()
    two.append(13796)
    two.append(26)
    assert_equal(ids_csv(two), "13796,26")
    # Genesis record: previous is the zero hash; verifies.
    var r1 = _chained_record(
        String(ZERO_RECORD_HASH), "1,2,3", "13796,26", 5000
    )
    assert_true(record_chain_valid(r1))
    var j = r1.to_json()
    assert_true(j.find('"prompt_token_ids":[1,2,3]') != -1)
    assert_true(j.find('"record_sha256":"') != -1)
    var back = parse_run_record(j, 0)
    assert_equal(back.record_sha256, r1.record_sha256)
    assert_equal(back.prev_record_sha256, String(ZERO_RECORD_HASH))
    assert_true(record_chain_valid(back))
    # Second record links to the first; the sequence and
    # the served history both verify.
    var r2 = _chained_record(r1.record_sha256, "1,2,3", "1573", 4000)
    var records = List[RunRecord]()
    records.append(r1.copy())
    records.append(r2.copy())
    var statuses = chain_statuses(records)
    assert_equal(statuses[0], "verified")
    assert_equal(statuses[1], "verified")
    var hj = history_json(records, "", "")
    assert_true(hj.find('"chain_status":"verified"') != -1)
    var lj = leaderboard_json(records)
    assert_true(lj.find('"verified":2') != -1)
    assert_true(lj.find('"tampered":0') != -1)


def test_record_hash_chain_tamper() raises:
    # A record whose stored hash was computed over OTHER
    # inputs (decode_ns 5001 vs the stored 5000) does not
    # recompute: TAMPERED_RECORD.
    var bad_sha = compute_record_sha256(
        "testbuild",
        "ab12",
        "1,2,3",
        "13796,26",
        1000,
        5001,
        String(ZERO_RECORD_HASH),
    )
    var atts = List[String]()
    var bad = RunRecord(
        1700000000,
        "dashboard",
        "M",
        "ab12",
        "",
        "gfx1100 / MAX 26.7",
        6,
        200,
        32.65,
        4,
        16739,
        260,
        "",
        atts^,
        False,
        "",
        build_sha="testbuild",
        prompt_ids_csv="1,2,3",
        generated_ids_csv="13796,26",
        prefill_ns=1000,
        decode_ns=5000,
        prev_record_sha256=String(ZERO_RECORD_HASH),
        record_sha256=bad_sha,
    )
    assert_false(record_chain_valid(bad))
    var one = List[RunRecord]()
    one.append(bad.copy())
    assert_equal(chain_statuses(one)[0], "TAMPERED_RECORD")
    # A self-consistent record pointing at the wrong
    # previous hash breaks the LINK: also tampered.
    var r1 = _chained_record(
        String(ZERO_RECORD_HASH), "1,2,3", "13796,26", 5000
    )
    var orphan = _chained_record(
        String(ZERO_RECORD_HASH), "1,2,3", "1573", 4000
    )
    var pair = List[RunRecord]()
    pair.append(r1.copy())
    pair.append(orphan.copy())
    var st = chain_statuses(pair)
    assert_equal(st[0], "verified")
    assert_equal(st[1], "TAMPERED_RECORD")
    # A pre-chain record is labelled, never silently
    # verified and never called tampered.
    var pre = List[RunRecord]()
    pre.append(_sample_record())
    assert_equal(chain_statuses(pre)[0], "unverifiable_pre_chain")
    assert_true(
        leaderboard_json(pre).find('"unverifiable_pre_chain":1') != -1
    )


def test_peak_bandwidth_config() raises:
    assert_equal(DEFAULT_GPU_PEAK_BANDWIDTH_GBPS, 960.0)
    assert_true(abs(parse_gpu_peak_bandwidth_gbps("960") - 960.0) < 0.01)
    assert_true(abs(parse_gpu_peak_bandwidth_gbps("1048.6") - 1048.6) < 0.01)
    try:
        _ = parse_gpu_peak_bandwidth_gbps("fast")
        assert_true(False)
    except e:
        assert_equal(error_code_of(String(e)), "mojo_engine_config")
    try:
        _ = parse_gpu_peak_bandwidth_gbps("0")
        assert_true(False)
    except e:
        assert_equal(error_code_of(String(e)), "mojo_engine_config")


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
        # Deep-telemetry addendum: the debug-level toggle
        # exists and rides the chat payload.
        'id="debugLevel"',
        "debug_level:",
        # readZip must close its function body; window 4
        # caught a one-brace-short ending via node --check.
        "return files;}}",
    ]:
        assert_true(page.find(marker) != -1)
    # Delimiter balance over the whole page: style and
    # script braces must net to zero or the browser drops
    # the script block silently.
    var opens = 0
    var closes = 0
    for b in page.as_bytes():
        if b == UInt8(123):
            opens += 1
        elif b == UInt8(125):
            closes += 1
    assert_equal(opens, closes)


def test_mcp_tools_list() raises:
    var tools = mcp_tools_json()
    for name in ["chat", "health", "stats", "debug_log", "history", "leaderboard"]:
        assert_true(tools.find('"name":"' + name + '"') != -1)


def test_sha256_fips_vector() raises:
    # FIPS 180-4 worked example: SHA-256("abc").
    var abc = List[UInt8]()
    for c in "abc".as_bytes():
        abc.append(c)
    assert_equal(
        sha256_hex_of_bytes(abc),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    )
    var empty = List[UInt8]()
    assert_equal(
        sha256_hex_of_bytes(empty),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    )
    # 56-byte FIPS example (exercises the two-block padding edge).
    var msg = String(
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
    )
    var bytes = List[UInt8]()
    for c in msg.as_bytes():
        bytes.append(c)
    assert_equal(
        sha256_hex_of_bytes(bytes),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
    )


def test_file_sha256_matches_bytes() raises:
    # The file path is what serve_main fingerprints the model
    # with; it must agree with the in-memory path exactly,
    # including across the 4 MiB read-chunk boundary with a
    # non-block-aligned remainder (the streaming carry path).
    var small = "/tmp/test_serve_sha_abc.bin"
    with open(small, "w") as f:
        f.write("abc")
    assert_equal(
        file_sha256_hex(small),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    )
    var block = String("")
    for i in range(90):
        block += String(chr(33 + i))
    var reps = 46604  # 46604 * 90 = 4,194,360 bytes
    var bytes = List[UInt8]()
    for _ in range(reps):
        for c in block.as_bytes():
            bytes.append(c)
    for i in range(9):
        bytes.append(UInt8(33 + i))
    var big = "/tmp/test_serve_sha_big.bin"
    with open(big, "w") as f:
        for _ in range(reps):
            f.write(block)
        var tail = String("")
        for i in range(9):
            tail += String(chr(33 + i))
        f.write(tail)
    assert_equal(len(bytes), 4194369)
    assert_equal(file_sha256_hex(big), sha256_hex_of_bytes(bytes))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
