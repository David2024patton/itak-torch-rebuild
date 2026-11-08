# server.mojo - pure dispatch bodies for the serve layer:
# every response body that does not need the model is built
# here, and the chat request is parsed here, so the accept
# loop in serve_main stays a thin wire: read, route, answer.
#
# Named refusals raised here: vision_not_ready (an image was
# attached; the engine has no vision path yet - the plumbing
# lands, the refusal is honest), serve_context_too_large,
# serve_max_tokens_exceeded, serve_empty_prompt,
# serve_sampler_not_available (the GREEDY LOCK, amendments
# 2026-11-08: v1 generation is greedy only; a request naming
# a nonzero temperature, top_k, top_p, or any seed refuses BY
# NAME instead of being silently ignored, so repeat runs and
# golden verdicts stay deterministic), plus the
# serve_bad_json / serve_missing_field family from json.mojo.

from config import ENGINE_NAME, ENGINE_VERSION
from serve.json import (
    has_json_key,
    json_content_strings,
    json_escape,
    json_float_field,
    json_int_field,
    json_string_array_field,
    json_string_field,
)


def model_display_name(model_path: String) -> String:
    """The served model id: the GGUF file's base name without
    the .gguf suffix (the OpenAI `model` field convention is
    a name, not a path; the full path stays in /health)."""
    var base = model_path
    var slash = -1
    var bytes = model_path.as_bytes()
    for i in range(len(bytes)):
        if bytes[i] == 47:  # '/'
            slash = i
    if slash != -1:
        base = String(model_path[byte = slash + 1 : model_path.byte_length()])
    if base.endswith(".gguf"):
        var trimmed = String(base[byte = 0 : base.byte_length() - 5])
        base = trimmed
    return base


def health_json(model_path: String, model_loaded: Bool, gpu: Bool) -> String:
    var out = String("{")
    out += '"status":"' + ("ok" if model_loaded else "degraded") + '"'
    out += ',"engine":"' + ENGINE_NAME + '"'
    out += ',"version":"' + ENGINE_VERSION + '"'
    out += ',"model":"' + json_escape(model_display_name(model_path)) + '"'
    out += ',"model_path":"' + json_escape(model_path) + '"'
    out += ',"model_loaded":' + ("true" if model_loaded else "false")
    out += ',"backend":"' + ("gpu" if gpu else "cpu") + '"'
    out += ',"sampler":{"name":"greedy","temperature":0.0,"seed":null}'
    out += ',"streaming":false'
    out += "}"
    return out


def models_json(model_path: String) -> String:
    var name = json_escape(model_display_name(model_path))
    var out = String('{"object":"list","data":[{')
    out += '"id":"' + name + '"'
    out += ',"object":"model"'
    out += ',"created":0'
    out += ',"owned_by":"mojo-engine"'
    out += "}]}"
    return out


def error_json(code: String, message: String) -> String:
    var out = String('{"error":{')
    out += '"message":"' + json_escape(message) + '"'
    out += ',"type":"serve_error"'
    out += ',"code":"' + json_escape(code) + '"'
    out += "}}"
    return out


def error_code_of(err: String) -> String:
    """The refusal name at the start of an error string, up to
    the first ':' or space - how a named refusal becomes the
    JSON error code without a second channel."""
    var bytes = err.as_bytes()
    for i in range(len(bytes)):
        if bytes[i] == 58 or bytes[i] == 32:
            return String(err[byte=0:i])
    return err


struct ChatRequest(Copyable, Movable):
    var prompt_text: String
    var context_text: String
    var max_tokens: Int
    var source: String
    var attachments: List[String]
    var quality_verdict: String

    def __init__(
        out self,
        prompt_text: String,
        context_text: String,
        max_tokens: Int,
        source: String,
        var attachments: List[String],
        quality_verdict: String,
    ):
        self.prompt_text = prompt_text
        self.context_text = context_text
        self.max_tokens = max_tokens
        self.source = source
        self.attachments = attachments^
        self.quality_verdict = quality_verdict


def chat_request_has_image(body: String) raises -> Bool:
    """True when the request carries an image in any of the
    standard field shapes: a top-level image / image_url /
    images field, or a message content part typed as an
    image. Detection is by field presence, per the v1
    contract: the dashboard sends exactly these shapes."""
    if has_json_key(body, "image_url"):
        return True
    if has_json_key(body, "image"):
        return True
    if has_json_key(body, "images"):
        return True
    if has_json_key(body, "input_image"):
        return True
    return False


def _check_greedy_lock(body: String) raises:
    """The greedy lock (amendments): temperature must be 0
    when present; top_k / top_p / seed refuse by name when
    present at all (v1 has no sampler for them, and a seed
    with greedy decoding would be a lie)."""
    if has_json_key(body, "temperature"):
        var t = json_float_field(body, "temperature", 0.0)
        if t != 0.0:
            raise Error(
                "serve_sampler_not_available: temperature "
                + String(t)
                + " requested; v1 generation is greedy only"
                " (temperature 0.0)"
            )
    for key in ["top_k", "top_p", "seed"]:
        if has_json_key(body, key):
            raise Error(
                "serve_sampler_not_available: '"
                + key
                + "' requested; v1 generation is greedy only"
            )


def parse_chat_request(
    body: String,
    default_max_tokens: Int,
    max_tokens_cap: Int,
    max_context_bytes: Int,
) raises -> ChatRequest:
    """Parse a /v1/chat/completions body.

    Shapes accepted: {"prompt": "..."} or {"messages":[...]}
    (message contents joined with newlines, roles not
    interpreted - the engine has no chat template), plus an
    optional {"context_text": "..."} attachment text, an
    optional integer max_tokens, and the bench fields
    {"source": "dashboard"|"cli"|"mcp", "attachments": [names],
    "quality_verdict": "..."} the run record consumes. An
    attached image refuses vision_not_ready BY NAME: no
    vision path exists yet. Sampler fields are governed by
    the greedy lock (see _check_greedy_lock).
    """
    _check_greedy_lock(body)
    if chat_request_has_image(body):
        raise Error(
            "vision_not_ready: image input is plumbed but the engine"
            " has no vision path yet; resend as text"
        )
    var prompt = String("")
    if has_json_key(body, "prompt"):
        prompt = json_string_field(body, "prompt")
    elif has_json_key(body, "messages"):
        var parts = json_content_strings(body)
        for i in range(len(parts)):
            if i > 0:
                prompt += "\n"
            prompt += parts[i]
    else:
        raise Error(
            "serve_missing_field: chat request needs 'prompt' or 'messages'"
        )
    if prompt.byte_length() == 0:
        raise Error("serve_empty_prompt: chat prompt text is empty")
    var context = String("")
    if has_json_key(body, "context_text"):
        context = json_string_field(body, "context_text")
        if context.byte_length() > max_context_bytes:
            raise Error(
                "serve_context_too_large: attachment context is "
                + String(context.byte_length())
                + " bytes, cap is "
                + String(max_context_bytes)
            )
    var max_tokens = json_int_field(body, "max_tokens", default_max_tokens)
    if max_tokens <= 0:
        raise Error(
            "serve_bad_json: max_tokens must be positive, got "
            + String(max_tokens)
        )
    if max_tokens > max_tokens_cap:
        raise Error(
            "serve_max_tokens_exceeded: max_tokens "
            + String(max_tokens)
            + " exceeds the configured cap "
            + String(max_tokens_cap)
        )
    var source = String("cli")
    if has_json_key(body, "source"):
        source = json_string_field(body, "source")
        if source != "dashboard" and source != "cli" and source != "mcp":
            raise Error(
                "serve_bad_json: source '"
                + source
                + "' is not dashboard|cli|mcp"
            )
    var attachments = json_string_array_field(body, "attachments")
    var verdict = String("")
    if has_json_key(body, "quality_verdict"):
        verdict = json_string_field(body, "quality_verdict")
    return ChatRequest(prompt, context, max_tokens, source, attachments^, verdict)


def effective_prompt(req: ChatRequest) -> String:
    """The text actually tokenized: attachment context first
    (as context), then the prompt, when context is present."""
    if req.context_text.byte_length() == 0:
        return req.prompt_text
    return req.context_text + "\n\n" + req.prompt_text


def chat_response_json(
    completion_id: String,
    created: Int,
    model_name: String,
    content: String,
    finish_reason: String,
    prompt_tokens: Int,
    completion_tokens: Int,
    total_ms: Int,
    prefill_ms: Int,
    decode_tok_s: Float64,
) -> String:
    """OpenAI-shaped chat.completion plus a timings object
    and the sampler settings actually used (the greedy lock:
    temperature 0.0, echoed on every response).

    ttft_ms is null on purpose: v1 is non-streaming; prefill
    ends when the first token is picked, so TTFT == prefill
    in this shape and prefill_ms carries the number instead
    of a second invented measurement (the run record says
    the same via ttft_note)."""
    var prefill_tok_s = Float64(0.0)
    if prefill_ms > 0:
        prefill_tok_s = Float64(prompt_tokens) * 1000.0 / Float64(prefill_ms)
    var out = String("{")
    out += '"id":"' + json_escape(completion_id) + '"'
    out += ',"object":"chat.completion"'
    out += ',"created":' + String(created)
    out += ',"model":"' + json_escape(model_name) + '"'
    out += ',"choices":[{"index":0,"message":{"role":"assistant"'
    out += ',"content":"' + json_escape(content) + '"}'
    out += ',"finish_reason":"' + finish_reason + '"}]'
    out += ',"usage":{"prompt_tokens":' + String(prompt_tokens)
    out += ',"completion_tokens":' + String(completion_tokens)
    out += ',"total_tokens":' + String(prompt_tokens + completion_tokens) + "}"
    out += ',"timings":{"total_ms":' + String(total_ms)
    out += ',"generation_ms":' + String(total_ms)
    out += ',"prefill_ms":' + String(prefill_ms)
    out += ',"prefill_tok_s":' + String(prefill_tok_s)
    out += ',"tokens_per_second":' + String(decode_tok_s)
    out += ',"decode_tok_s":' + String(decode_tok_s)
    out += ',"ttft_ms":null}'
    out += ',"sampler":{"name":"greedy","temperature":0.0,"seed":null}'
    out += "}"
    return out


def mcp_tools_json() -> String:
    """The MCP-style bridge tool list (GET /mcp/tools). The
    chat tool runs carry source "mcp" into the run history
    (amendments: every endpoint run is recorded, tagged)."""
    var out = String('{"tools":[')
    out += (
        '{"name":"chat","description":"Run one greedy chat completion on the'
        ' engine","input":{"prompt":"string","max_tokens":"int, optional"}}'
    )
    out += (
        ',{"name":"health","description":"Engine health JSON (same body as GET'
        ' /health)","input":{}}'
    )
    out += (
        ',{"name":"stats","description":"Serve counters JSON (same body as GET'
        ' /stats)","input":{}}'
    )
    out += (
        ',{"name":"debug_log","description":"Recent diag ring lines (same body'
        ' as GET /debug/log)","input":{}}'
    )
    out += (
        ',{"name":"history","description":"Run history JSON (same body as GET'
        ' /history)","input":{}}'
    )
    out += (
        ',{"name":"leaderboard","description":"Leaderboard JSON (same body as'
        ' GET /leaderboard)","input":{}}'
    )
    out += "]}"
    return out


def ingest_result_json(
    accepted: Int, rejected: Int, rejection_names: List[String]
) -> String:
    var out = String("{")
    out += '"accepted":' + String(accepted)
    out += ',"rejected":' + String(rejected)
    out += ',"rejections":['
    for i in range(len(rejection_names)):
        if i > 0:
            out += ","
        out += '"' + json_escape(rejection_names[i]) + '"'
    out += "]}"
    return out


def attachment_result_json(filename: String, text: String) -> String:
    var out = String("{")
    out += '"filename":"' + json_escape(filename) + '"'
    out += ',"chars":' + String(text.byte_length())
    out += ',"text":"' + json_escape(text) + '"'
    out += "}"
    return out


def parse_attachment_request(
    body: String, max_context_bytes: Int
) raises -> Tuple[String, String]:
    """Validate a POST /v1/attachments/text body: the
    dashboard extracts text browser-side and posts it here
    for the size/type gate; the server stores nothing.
    Returns (filename, text)."""
    var filename = json_string_field(body, "filename")
    var lower = filename.lower()
    var ok_type = (
        lower.endswith(".txt")
        or lower.endswith(".md")
        or lower.endswith(".pdf")
        or lower.endswith(".zip")
        or lower.endswith(".7z")
    )
    if not ok_type:
        raise Error(
            "serve_attachment_type: only .txt/.md/.pdf/.zip/.7z"
            " attachments are accepted, got '"
            + filename
            + "'"
        )
    var text = json_string_field(body, "text")
    if text.byte_length() > max_context_bytes:
        raise Error(
            "serve_context_too_large: attachment text is "
            + String(text.byte_length())
            + " bytes, cap is "
            + String(max_context_bytes)
        )
    return (filename, text)
