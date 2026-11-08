# history.mojo - the run-record schema and everything built
# on it (amendments, David 2026-11-08): the on-disk history,
# the leaderboard view, and the export/import run bundles
# that let a second bench (Chris's 12 GB NVIDIA card) sync
# its measured runs into this one.
#
# THE SCHEMA (history and leaderboard share it; bundle format
# version 1). Every run record carries:
#   timestamp (unix seconds, from the serving host's clock;
#     imported records keep the origin host's timestamp),
#   source: "dashboard" | "cli" | "mcp",
#   model file name + model_fingerprint (sha256 of the model
#     FILE bytes when computed, else null + fingerprint_note
#     saying why - a 17 GB hash is an operator choice, never
#     silently faked; the config-model record at serve start
#     computes it, dashboard/CLI records carry what they were
#     given),
#   backend_target (chip + runtime string, config-stamped),
#   prompt_tokens, prompt_bucket (128 / 1024 / 4096 nominal,
#     nearest standard size with ties going to the smaller
#     bucket, clamped to the ends - the rule is in
#     prompt_bucket_of and nowhere else),
#   prefill_ms AND prefill_tok_s (both, always),
#   ttft_ms: null in v1 non-streaming - prefill ends when the
#     first token is picked, so TTFT == prefill_ms here and
#     the record says so via ttft_note instead of inventing a
#     second measurement,
#   decode_tok_s, tokens_generated,
#   peak_vram_mib: null + peak_vram_note in v1 (MAX 26.7
#     exposes no device-memory counter to Mojo; the engine's
#     own reservation ledger is the honest number and it is
#     recorded separately as reserved_vram_mib),
#   kv_cache_mib (the engine's budget accounting figure),
#   sampler {name, temperature, seed} - greedy, 0.0, null in
#     v1 by the greedy lock,
#   quality_verdict (string, "" when the run is not a check),
#   attachments (file names attached to the run),
#   imported flag + origin_backend for imported records.
#
# History persists as JSON lines at
# <serve_history_dir>/runs.jsonl; it survives restart because
# it is a file, not a counter. GET /history filters by source
# and model. GET /history/export returns the bundle
# {"format":"mojo-engine-run-bundle","format_version":1,
# "exported_by":..., "runs":[...]}. POST /v1/runs/ingest
# validates EVERY record (missing/ill-typed required fields
# refuse by name: serve_ingest_bad_record) - bad records are
# counted and reported, valid ones still land; imported runs
# keep their source, gain "imported":true and their origin
# backend, and leaderboard rows are per model + hardware,
# never merged across cards. The bundle is UNSIGNED in v1
# (known limit, recorded in the bench note); format_version
# is where signing will attach.
#
# The leaderboard adds REFERENCE rows (David's decision):
# llama.cpp and the paused Rust engine, clearly marked
# "reference" with their bench-note sources named. Reference
# rows are served alongside measured rows under their own
# "reference" kind; they are never history records and never
# mixed into measured rankings silently.

from serve.json import (
    has_json_key,
    json_bool_field,
    json_escape,
    json_float_field,
    json_int_field,
    json_raw_field,
    json_string_array_field,
    json_string_field,
    split_json_array,
)

comptime RUN_BUNDLE_FORMAT = "mojo-engine-run-bundle"
comptime RUN_BUNDLE_VERSION = 1
comptime RUNS_FILE_NAME = "runs.jsonl"

comptime MIB_BYTES = 1048576


def prompt_bucket_of(prompt_tokens: Int) -> Int:
    """The nominal prompt bucket: nearest of 128 / 1024 /
    4096, ties to the smaller bucket, clamped at the ends.
    576 is the 128/1024 midpoint and buckets to 128; 2560 is
    the 1024/4096 midpoint and buckets to 1024."""
    if prompt_tokens <= 576:
        return 128
    if prompt_tokens <= 2560:
        return 1024
    return 4096


def valid_source(source: String) -> Bool:
    return source == "dashboard" or source == "cli" or source == "mcp"


struct RunRecord(Copyable, Movable):
    var timestamp: Int
    var source: String
    var model: String
    var model_fingerprint: String
    var fingerprint_note: String
    var backend_target: String
    var prompt_tokens: Int
    var prompt_bucket: Int
    var prefill_ms: Int
    var prefill_tok_s: Float64
    var ttft_note: String
    var decode_tok_s: Float64
    var tokens_generated: Int
    var reserved_vram_mib: Int
    var kv_cache_mib: Int
    var quality_verdict: String
    var attachments: List[String]
    var imported: Bool
    var origin_backend: String

    def __init__(
        out self,
        timestamp: Int,
        source: String,
        model: String,
        model_fingerprint: String,
        fingerprint_note: String,
        backend_target: String,
        prompt_tokens: Int,
        prefill_ms: Int,
        decode_tok_s: Float64,
        tokens_generated: Int,
        reserved_vram_mib: Int,
        kv_cache_mib: Int,
        quality_verdict: String,
        var attachments: List[String],
        imported: Bool,
        origin_backend: String,
    ):
        self.timestamp = timestamp
        self.source = source
        self.model = model
        self.model_fingerprint = model_fingerprint
        self.fingerprint_note = fingerprint_note
        self.backend_target = backend_target
        self.prompt_tokens = prompt_tokens
        self.prompt_bucket = prompt_bucket_of(prompt_tokens)
        self.prefill_ms = prefill_ms
        if prefill_ms > 0:
            self.prefill_tok_s = (
                Float64(prompt_tokens) * 1000.0 / Float64(prefill_ms)
            )
        else:
            self.prefill_tok_s = Float64(0.0)
        # v1 non-streaming: see the module header.
        self.ttft_note = String("ttft_equals_prefill_non_streaming_v1")
        self.decode_tok_s = decode_tok_s
        self.tokens_generated = tokens_generated
        self.reserved_vram_mib = reserved_vram_mib
        self.kv_cache_mib = kv_cache_mib
        self.quality_verdict = quality_verdict
        self.attachments = attachments^
        self.imported = imported
        self.origin_backend = origin_backend

    def to_json(self) -> String:
        var out = String("{")
        out += '"timestamp":' + String(self.timestamp)
        out += ',"source":"' + json_escape(self.source) + '"'
        out += ',"model":"' + json_escape(self.model) + '"'
        if self.model_fingerprint != "":
            out += ',"model_fingerprint":"' + json_escape(
                self.model_fingerprint
            ) + '"'
        else:
            out += ',"model_fingerprint":null'
        out += ',"fingerprint_note":"' + json_escape(self.fingerprint_note) + '"'
        out += ',"backend_target":"' + json_escape(self.backend_target) + '"'
        out += ',"prompt_tokens":' + String(self.prompt_tokens)
        out += ',"prompt_bucket":' + String(self.prompt_bucket)
        out += ',"prefill_ms":' + String(self.prefill_ms)
        out += ',"prefill_tok_s":' + String(self.prefill_tok_s)
        out += ',"ttft_ms":null'
        out += ',"ttft_note":"' + json_escape(self.ttft_note) + '"'
        out += ',"decode_tok_s":' + String(self.decode_tok_s)
        out += ',"tokens_generated":' + String(self.tokens_generated)
        out += ',"peak_vram_mib":null'
        out += (
            ',"peak_vram_note":"not exposed by MAX 26.7 to Mojo;'
            ' reserved_vram_mib is the engine reservation ledger"'
        )
        out += ',"reserved_vram_mib":' + String(self.reserved_vram_mib)
        out += ',"kv_cache_mib":' + String(self.kv_cache_mib)
        out += ',"sampler":{"name":"greedy","temperature":0.0,"seed":null}'
        out += ',"quality_verdict":"' + json_escape(self.quality_verdict) + '"'
        out += ',"attachments":['
        for i in range(len(self.attachments)):
            if i > 0:
                out += ","
            out += '"' + json_escape(self.attachments[i]) + '"'
        out += "]"
        out += ',"imported":' + ("true" if self.imported else "false")
        if self.imported:
            out += ',"origin_backend":"' + json_escape(self.origin_backend) + '"'
        out += "}"
        return out


def _req_string(obj: String, key: String, idx: Int) raises -> String:
    if not has_json_key(obj, key):
        raise Error(
            "serve_ingest_bad_record: record "
            + String(idx)
            + " is missing '"
            + key
            + "'"
        )
    return json_string_field(obj, key)


def _req_int(obj: String, key: String, idx: Int) raises -> Int:
    if not has_json_key(obj, key):
        raise Error(
            "serve_ingest_bad_record: record "
            + String(idx)
            + " is missing '"
            + key
            + "'"
        )
    return json_int_field(obj, key, 0)


def parse_run_record(obj: String, idx: Int) raises -> RunRecord:
    """Validate one bundle record against the schema and
    build it. Required: timestamp, source (one of the three
    tags), model, backend_target, prompt_tokens, prefill_ms,
    decode_tok_s, tokens_generated. Everything else defaults
    honestly. Refusals are serve_ingest_bad_record BY NAME,
    with the record index and the field."""
    var timestamp = _req_int(obj, "timestamp", idx)
    var source = _req_string(obj, "source", idx)
    if not valid_source(source):
        raise Error(
            "serve_ingest_bad_record: record "
            + String(idx)
            + " has source '"
            + source
            + "', not dashboard|cli|mcp"
        )
    var model = _req_string(obj, "model", idx)
    var backend = _req_string(obj, "backend_target", idx)
    var prompt_tokens = _req_int(obj, "prompt_tokens", idx)
    var prefill_ms = _req_int(obj, "prefill_ms", idx)
    var tokens_generated = _req_int(obj, "tokens_generated", idx)
    if prompt_tokens < 0 or prefill_ms < 0 or tokens_generated < 0:
        raise Error(
            "serve_ingest_bad_record: record "
            + String(idx)
            + " has a negative count field"
        )
    if not has_json_key(obj, "decode_tok_s"):
        raise Error(
            "serve_ingest_bad_record: record "
            + String(idx)
            + " is missing 'decode_tok_s'"
        )
    var decode_tok_s = json_float_field(obj, "decode_tok_s", 0.0)
    var fingerprint = String("")
    var fingerprint_note = String("not_provided_by_origin")
    if has_json_key(obj, "model_fingerprint"):
        try:
            fingerprint = json_string_field(obj, "model_fingerprint")
            fingerprint_note = String("")
        except:
            fingerprint = String("")
    if has_json_key(obj, "fingerprint_note"):
        try:
            var note = json_string_field(obj, "fingerprint_note")
            if fingerprint == "":
                fingerprint_note = note
        except:
            pass
    var reserved = json_int_field(obj, "reserved_vram_mib", 0)
    var kv_mib = json_int_field(obj, "kv_cache_mib", 0)
    var verdict = String("")
    if has_json_key(obj, "quality_verdict"):
        verdict = json_string_field(obj, "quality_verdict")
    var attachments = json_string_array_field(obj, "attachments")
    # Imported records keep their origin backend for
    # attribution; the origin IS the record's backend_target.
    return RunRecord(
        timestamp,
        source,
        model,
        fingerprint,
        fingerprint_note,
        backend,
        prompt_tokens,
        prefill_ms,
        decode_tok_s,
        tokens_generated,
        reserved,
        kv_mib,
        verdict,
        attachments^,
        True,
        backend,
    )


def parse_bundle(body: String) raises -> List[String]:
    """Validate the bundle envelope and return the raw record
    texts. Refusals: serve_ingest_bad_bundle (not an object,
    wrong format name, unsupported format_version, no runs
    array)."""
    if not has_json_key(body, "format"):
        raise Error("serve_ingest_bad_bundle: no 'format' field")
    var fmt = json_string_field(body, "format")
    if fmt != RUN_BUNDLE_FORMAT:
        raise Error(
            "serve_ingest_bad_bundle: format '"
            + fmt
            + "' is not "
            + RUN_BUNDLE_FORMAT
        )
    var version = json_int_field(body, "format_version", -1)
    if version != RUN_BUNDLE_VERSION:
        raise Error(
            "serve_ingest_bad_bundle: format_version "
            + String(version)
            + " is not supported (this bench reads version "
            + String(RUN_BUNDLE_VERSION)
            + ")"
        )
    var runs_raw = json_raw_field(body, "runs")
    if runs_raw == "":
        raise Error("serve_ingest_bad_bundle: no 'runs' array")
    return split_json_array(runs_raw)


def bundle_json(runs: List[RunRecord], exported_by: String) -> String:
    var out = String("{")
    out += '"format":"' + RUN_BUNDLE_FORMAT + '"'
    out += ',"format_version":' + String(RUN_BUNDLE_VERSION)
    out += ',"exported_by":"' + json_escape(exported_by) + '"'
    out += ',"signed":false'
    out += ',"runs":['
    for i in range(len(runs)):
        if i > 0:
            out += ","
        out += runs[i].to_json()
    out += "]}"
    return out


def read_all_lines(path: String) -> String:
    """A history file's whole text; "" when it does not exist
    yet (a fresh bench has an empty history, not an error)."""
    try:
        with open(path, "r") as f:
            return f.read()
    except:
        return String("")


def load_records(dir_path: String) -> List[RunRecord]:
    """Parse the JSON-lines history back into records.
    Lines that no longer validate are SKIPPED here (they stay
    in the file; the ingest path is the one that refuses by
    name) - history display must never take the bench down."""
    var out = List[RunRecord]()
    var text = read_all_lines(dir_path + "/" + RUNS_FILE_NAME)
    var start = 0
    var bytes = text.as_bytes()
    var i = 0
    while i <= len(bytes):
        if i == len(bytes) or bytes[i] == 10:
            if i > start:
                var line = String(text[byte=start:i])
                try:
                    # Local lines were written imported=false;
                    # re-mark on load from the stored flag.
                    var rec = parse_run_record(line, len(out))
                    if has_json_key(line, "imported"):
                        rec.imported = json_bool_field(line, "imported", False)
                        if rec.imported and has_json_key(
                            line, "origin_backend"
                        ):
                            rec.origin_backend = json_string_field(
                                line, "origin_backend"
                            )
                        if not rec.imported:
                            rec.origin_backend = String("")
                    out.append(rec^)
                except:
                    pass
            start = i + 1
        i += 1
    return out^


def append_record(dir_path: String, rec: RunRecord) raises:
    """Append one record line. The directory is created by
    the caller (serve_main) at startup; a missing directory
    here refuses serve_history_io by name."""
    var path = dir_path + "/" + RUNS_FILE_NAME
    try:
        with open(path, "a") as f:
            f.write(rec.to_json() + "\n")
    except e:
        raise Error(
            "serve_history_io: cannot append to " + path + ": " + String(e)
        )


def history_json(
    records: List[RunRecord], source_filter: String, model_filter: String
) -> String:
    var out = String('{"count":')
    var shown = List[String]()
    for i in range(len(records)):
        if source_filter != "" and records[i].source != source_filter:
            continue
        if model_filter != "" and records[i].model != model_filter:
            continue
        shown.append(records[i].to_json())
    out += String(len(shown))
    out += ',"runs":['
    for i in range(len(shown)):
        if i > 0:
            out += ","
        out += shown[i]
    out += "]}"
    return out


struct LeaderRow(Copyable, Movable):
    var model: String
    var backend_target: String
    var kind: String
    var runs: Int
    var best_decode_tok_s: Float64
    var best_prefill_tok_s: Float64
    var last_timestamp: Int
    var quality_verdicts: String
    var source_note: String

    def __init__(
        out self,
        model: String,
        backend_target: String,
        kind: String,
        runs: Int,
        best_decode_tok_s: Float64,
        best_prefill_tok_s: Float64,
        last_timestamp: Int,
        quality_verdicts: String,
        source_note: String,
    ):
        self.model = model
        self.backend_target = backend_target
        self.kind = kind
        self.runs = runs
        self.best_decode_tok_s = best_decode_tok_s
        self.best_prefill_tok_s = best_prefill_tok_s
        self.last_timestamp = last_timestamp
        self.quality_verdicts = quality_verdicts
        self.source_note = source_note


def reference_rows() -> List[LeaderRow]:
    """The reference engines (David's decision): llama.cpp
    Vulkan and the paused Rust engine on the same Glimmer
    model + card, marked reference with their bench records
    named. These are NOT measurements of this bench and are
    never stored in history."""
    var rows = List[LeaderRow]()
    rows.append(
        LeaderRow(
            "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M",
            "gfx1100 / llama.cpp Vulkan",
            "reference",
            0,
            44.59,
            995.99,
            0,
            "",
            "llama.cpp baseline record,"
            " bench/wave20-session-2026-11-07.md era (llama-bench,"
            " 1,022-token prefill 995.99 tok/s, decode 44.59 tok/s)",
        )
    )
    rows.append(
        LeaderRow(
            "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M",
            "gfx1100 / Rust wgpu (paused)",
            "reference",
            0,
            34.5,
            0.0,
            0,
            "",
            "paused Rust engine plain decode ~34.5 tok/s,"
            " bench/wave20-session-2026-11-07.md",
        )
    )
    return rows^


def _row_json(row: LeaderRow) -> String:
    var out = String("{")
    out += '"model":"' + json_escape(row.model) + '"'
    out += ',"backend_target":"' + json_escape(row.backend_target) + '"'
    out += ',"kind":"' + json_escape(row.kind) + '"'
    out += ',"runs":' + String(row.runs)
    out += ',"best_decode_tok_s":' + String(row.best_decode_tok_s)
    out += ',"best_prefill_tok_s":' + String(row.best_prefill_tok_s)
    out += ',"last_timestamp":' + String(row.last_timestamp)
    out += ',"quality_verdicts":"' + json_escape(row.quality_verdicts) + '"'
    out += ',"source_note":"' + json_escape(row.source_note) + '"'
    out += "}"
    return out


def leaderboard_json(records: List[RunRecord]) -> String:
    """Measured rows grouped per model + hardware (never
    merged across cards), ranked by best decode tok/s,
    followed by the reference rows under their own kind."""
    var rows = List[LeaderRow]()
    for i in range(len(records)):
        ref rec = records[i]
        var key_backend = rec.backend_target
        var found = -1
        for j in range(len(rows)):
            if rows[j].model == rec.model and rows[j].backend_target == key_backend:
                found = j
                break
        if found == -1:
            rows.append(
                LeaderRow(
                    rec.model,
                    key_backend,
                    "measured",
                    0,
                    0.0,
                    0.0,
                    0,
                    "",
                    "imported runs included" if rec.imported else "",
                )
            )
            found = len(rows) - 1
        rows[found].runs += 1
        if rec.decode_tok_s > rows[found].best_decode_tok_s:
            rows[found].best_decode_tok_s = rec.decode_tok_s
        if rec.prefill_tok_s > rows[found].best_prefill_tok_s:
            rows[found].best_prefill_tok_s = rec.prefill_tok_s
        if rec.timestamp > rows[found].last_timestamp:
            rows[found].last_timestamp = rec.timestamp
        if rec.quality_verdict != "":
            if rows[found].quality_verdicts != "":
                rows[found].quality_verdicts += "; "
            rows[found].quality_verdicts += rec.quality_verdict
        if rec.imported and rows[found].source_note == "":
            rows[found].source_note = "imported runs included"
    # insertion sort by best decode, descending (rows are few)
    for i in range(1, len(rows)):
        var j = i
        while j > 0 and rows[j].best_decode_tok_s > rows[j - 1].best_decode_tok_s:
            var tmp = rows[j - 1].copy()
            rows[j - 1] = rows[j].copy()
            rows[j] = tmp^
            j -= 1
    var out = String('{"rows":[')
    var first = True
    for i in range(len(rows)):
        if not first:
            out += ","
        first = False
        out += _row_json(rows[i])
    var refs = reference_rows()
    for i in range(len(refs)):
        if not first:
            out += ","
        first = False
        out += _row_json(refs[i])
    out += "]}"
    return out


def _sha256_k(t: Int) -> UInt32:
    var words = List[UInt32]()
    words.extend([0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5])
    words.extend([0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174])
    words.extend([0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA])
    words.extend([0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967])
    words.extend([0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85])
    words.extend([0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070])
    words.extend([0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3])
    words.extend([0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2])
    return words[t]


def _rotr32(x: UInt32, n: Int) -> UInt32:
    return (x >> UInt32(n)) | (x << UInt32(32 - n))


def _sha256_block(h: List[UInt32], data: List[UInt8], base: Int) -> List[UInt32]:
    var w = List[UInt32](capacity=64)
    for t in range(16):
        var o = base + t * 4
        w.append(
            (UInt32(data[o]) << UInt32(24))
            | (UInt32(data[o + 1]) << UInt32(16))
            | (UInt32(data[o + 2]) << UInt32(8))
            | UInt32(data[o + 3])
        )
    for t in range(16, 64):
        var s0 = (
            _rotr32(w[t - 15], 7) ^ _rotr32(w[t - 15], 18) ^ (w[t - 15] >> 3)
        )
        var s1 = _rotr32(w[t - 2], 17) ^ _rotr32(w[t - 2], 19) ^ (w[t - 2] >> 10)
        w.append(w[t - 16] + s0 + w[t - 7] + s1)
    var a = h[0]
    var b = h[1]
    var c = h[2]
    var d = h[3]
    var e = h[4]
    var f = h[5]
    var g = h[6]
    var hh = h[7]
    for t in range(64):
        var s1 = _rotr32(e, 6) ^ _rotr32(e, 11) ^ _rotr32(e, 25)
        var ch = (e & f) ^ ((~e) & g)
        var t1 = hh + s1 + ch + _sha256_k(t) + w[t]
        var s0 = _rotr32(a, 2) ^ _rotr32(a, 13) ^ _rotr32(a, 22)
        var maj = (a & b) ^ (a & c) ^ (b & c)
        var t2 = s0 + maj
        hh = g
        g = f
        f = e
        e = d + t1
        d = c
        c = b
        b = a
        a = t1 + t2
    return [
        h[0] + a, h[1] + b, h[2] + c, h[3] + d,
        h[4] + e, h[5] + f, h[6] + g, h[7] + hh,
    ]


def sha256_hex_of_bytes(data: List[UInt8]) -> String:
    """Hex SHA-256 (FIPS 180-4) of an in-memory byte list,
    computed in pure Mojo. The fingerprint path cannot use
    OpenSSL: the MAX link line carries no libcrypto, so the
    SHA256_* symbols are undefined at link time (proven on
    the server build, 2026-11-08). Suite-pinned against the
    FIPS "abc" test vector."""
    var h: List[UInt32] = [
        0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
        0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
    ]
    var total = len(data)
    var buf = List[UInt8]()
    for i in range(len(data)):
        buf.append(data[i])
    buf.append(UInt8(0x80))
    while len(buf) % 64 != 56:
        buf.append(UInt8(0))
    var bit_len = UInt64(total) * 8
    for shift in [56, 48, 40, 32, 24, 16, 8, 0]:
        buf.append(UInt8((bit_len >> UInt64(shift)) & 0xFF))
    var base = 0
    while base < len(buf):
        h = _sha256_block(h, buf, base)
        base += 64
    var hexd = "0123456789abcdef"
    var hd = hexd.as_bytes()
    var out = String("")
    for i in range(8):
        var word = h[i]
        for s in range(4):
            var byte = Int((word >> UInt32(24 - 8 * s)) & 0xFF)
            out += String(chr(Int(hd[byte >> 4])))
            out += String(chr(Int(hd[byte & 15])))
    return out^


def file_sha256_hex(path: String) raises -> String:
    """SHA-256 of a file's bytes (pure Mojo, chunked reads).
    Used for the served model's fingerprint at serve start;
    a 17 GB file hashes in a few seconds."""
    var h: List[UInt32] = [
        0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
        0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
    ]
    var tail = List[UInt8]()
    var total: Int = 0
    try:
        with open(path, "r") as f:
            while True:
                var chunk = f.read_bytes(4194304)
                if len(chunk) == 0:
                    break
                total += len(chunk)
                for i in range(len(chunk)):
                    tail.append(chunk[i])
                var full = (len(tail) // 64) * 64
                if full > 0:
                    var base = 0
                    var hh = h.copy()
                    while base < full:
                        hh = _sha256_block(hh, tail, base)
                        base += 64
                    h = hh^
                    var rest = List[UInt8]()
                    for i in range(full, len(tail)):
                        rest.append(tail[i])
                    tail = rest^
    except e:
        raise Error("serve_fingerprint: cannot read " + path + ": " + String(e))
    tail.append(UInt8(0x80))
    while len(tail) % 64 != 56:
        tail.append(UInt8(0))
    var bit_len = UInt64(total) * 8
    for shift in [56, 48, 40, 32, 24, 16, 8, 0]:
        tail.append(UInt8((bit_len >> UInt64(shift)) & 0xFF))
    var base2 = 0
    while base2 < len(tail):
        h = _sha256_block(h, tail, base2)
        base2 += 64
    var hexd = "0123456789abcdef"
    var hd = hexd.as_bytes()
    var out = String("")
    for i in range(8):
        var word = h[i]
        for s in range(4):
            var byte = Int((word >> UInt32(24 - 8 * s)) & 0xFF)
            out += String(chr(Int(hd[byte >> 4])))
            out += String(chr(Int(hd[byte & 15])))
    return out^


def utc_now_seconds() -> Int:
    """Wall-clock unix seconds: libc time() through FFI (the
    same route the socket layer uses; Mojo 1.2's std.time is
    monotonic only). Falls back to 0 only if the call itself
    fails, which the record then shows honestly."""
    from std.ffi import external_call as _ec

    var t = _ec["time", Int](Int(0))
    if t < 0:
        return 0
    return t
