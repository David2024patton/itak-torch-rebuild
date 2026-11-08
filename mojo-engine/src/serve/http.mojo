# http.mojo - HTTP/1.1 request parsing, response building, and
# routing for the serve layer. Pure functions over strings:
# the socket module (socket.mojo) owns the bytes on the wire,
# this module owns what they MEAN. One request per connection,
# Connection: close on every response (v1 contract; a test
# bench does not need keep-alive).
#
# Named refusals: serve_http_bad_request (no header end, a
# request line that is not "METHOD PATH HTTP/...", a bad
# Content-Length), serve_body_too_large is raised by the
# socket reader and by parse_chat_request's caller, not here.

from serve.json import _find_substring
from tokenizer.byte_level import string_from_bytes_lossy

comptime ROUTE_NOT_FOUND = 0
comptime ROUTE_HEALTH = 1
comptime ROUTE_MODELS = 2
comptime ROUTE_CHAT = 3
comptime ROUTE_STATS = 4
comptime ROUTE_OPENAPI = 5
comptime ROUTE_DASHBOARD = 6
comptime ROUTE_METHOD_NOT_ALLOWED = 7
comptime ROUTE_DEBUG_LOG = 8
comptime ROUTE_MCP_TOOLS = 9
comptime ROUTE_MCP_CALL = 10
comptime ROUTE_MCP = 11
comptime ROUTE_ATTACHMENT = 12
comptime ROUTE_HISTORY = 13
comptime ROUTE_HISTORY_EXPORT = 14
comptime ROUTE_LEADERBOARD = 15
comptime ROUTE_RUNS_INGEST = 16
comptime ROUTE_MODEL_DETAIL = 17


struct HttpRequest(Copyable, Movable):
    var method: String
    var path: String
    var query: String
    var body: String
    var content_length: Int

    def __init__(
        out self,
        method: String,
        path: String,
        query: String,
        body: String,
        content_length: Int,
    ):
        self.method = method
        self.path = path
        self.query = query
        self.body = body
        self.content_length = content_length


def _slice(text: String, start: Int, end: Int) -> String:
    var bytes = text.as_bytes()
    var out = List[UInt8]()
    var i = start
    while i < end and i < len(bytes):
        out.append(bytes[i])
        i += 1
    return string_from_bytes_lossy(out)


def find_header_end(raw: String) -> Int:
    """Index just past the blank line ending the headers, or
    -1 when the header block is not complete yet."""
    var at = _find_substring(raw, "\r\n\r\n", 0)
    if at != -1:
        return at + 4
    at = _find_substring(raw, "\n\n", 0)
    if at != -1:
        return at + 2
    return -1


def header_content_length(raw: String) raises -> Int:
    """The Content-Length header value, 0 when absent.

    Case-insensitive name match; a non-numeric value refuses
    serve_http_bad_request by name.
    """
    var lower = raw.lower()
    var needle = String("content-length:")
    var at = _find_substring(lower, needle, 0)
    if at == -1:
        return 0
    var i = at + needle.byte_length()
    var bytes = raw.as_bytes()
    while i < len(bytes) and bytes[i] == 32:
        i += 1
    var value = 0
    var digits = 0
    while i < len(bytes) and 48 <= Int(bytes[i]) and Int(bytes[i]) <= 57:
        value = value * 10 + (Int(bytes[i]) - 48)
        digits += 1
        i += 1
    if digits == 0:
        raise Error("serve_http_bad_request: bad Content-Length header")
    return value


def parse_http_request(raw: String) raises -> HttpRequest:
    """Parse one complete raw request (headers + body)."""
    var hend = find_header_end(raw)
    if hend == -1:
        raise Error("serve_http_bad_request: header block incomplete")
    var head = _slice(raw, 0, hend)
    # Request line: METHOD SP PATH SP HTTP/x
    var line_end = _find_substring(head, "\r\n", 0)
    if line_end == -1:
        line_end = _find_substring(head, "\n", 0)
    if line_end == -1:
        raise Error("serve_http_bad_request: no request line")
    var line = _slice(head, 0, line_end)
    var sp1 = _find_substring(line, " ", 0)
    if sp1 <= 0:
        raise Error("serve_http_bad_request: malformed request line")
    var sp2 = _find_substring(line, " ", sp1 + 1)
    if sp2 == -1:
        raise Error("serve_http_bad_request: malformed request line")
    var method = _slice(line, 0, sp1)
    var target = _slice(line, sp1 + 1, sp2)
    if not _slice(line, sp2 + 1, line.byte_length()).startswith("HTTP/"):
        raise Error("serve_http_bad_request: request line has no HTTP version")
    # Routing is by path; the query string rides along for
    # the handlers that filter (/history?source=&model=).
    var q = _find_substring(target, "?", 0)
    var path = target if q == -1 else _slice(target, 0, q)
    var query = String("") if q == -1 else _slice(target, q + 1, target.byte_length())
    var clen = header_content_length(raw)
    var body = _slice(raw, hend, raw.byte_length())
    return HttpRequest(method, path, query, body, clen)


def query_param(query: String, key: String) -> String:
    """One query parameter's raw value ("" when absent; no
    percent-decoding in v1 - filters are plain ASCII tags)."""
    var needle = key + "="
    var from_idx = 0
    while True:
        var at = _find_substring(query, needle, from_idx)
        if at == -1:
            return String("")
        if at == 0 or query.as_bytes()[at - 1] == 38:  # '&'
            var start = at + needle.byte_length()
            var end = _find_substring(query, "&", start)
            if end == -1:
                end = query.byte_length()
            return _slice(query, start, end)
        from_idx = at + 1


def route_request(method: String, path: String) -> Int:
    """The routing table. A known path with the wrong method
    is ROUTE_METHOD_NOT_ALLOWED, not a 404: the bench should
    say the endpoint exists."""
    if path == "/health":
        return ROUTE_HEALTH if method == "GET" else ROUTE_METHOD_NOT_ALLOWED
    if path == "/v1/models":
        return ROUTE_MODELS if method == "GET" else ROUTE_METHOD_NOT_ALLOWED
    if path.startswith("/v1/models/") and path.byte_length() > 11:
        return (
            ROUTE_MODEL_DETAIL if method == "GET" else ROUTE_METHOD_NOT_ALLOWED
        )
    if path == "/history":
        return ROUTE_HISTORY if method == "GET" else ROUTE_METHOD_NOT_ALLOWED
    if path == "/history/export":
        return (
            ROUTE_HISTORY_EXPORT if method == "GET"
            else ROUTE_METHOD_NOT_ALLOWED
        )
    if path == "/leaderboard":
        return (
            ROUTE_LEADERBOARD if method == "GET" else ROUTE_METHOD_NOT_ALLOWED
        )
    if path == "/v1/runs/ingest":
        return (
            ROUTE_RUNS_INGEST if method == "POST" else ROUTE_METHOD_NOT_ALLOWED
        )
    if path == "/v1/chat/completions":
        return ROUTE_CHAT if method == "POST" else ROUTE_METHOD_NOT_ALLOWED
    if path == "/stats":
        return ROUTE_STATS if method == "GET" else ROUTE_METHOD_NOT_ALLOWED
    if path == "/openapi.json":
        return ROUTE_OPENAPI if method == "GET" else ROUTE_METHOD_NOT_ALLOWED
    if path == "/":
        return ROUTE_DASHBOARD if method == "GET" else ROUTE_METHOD_NOT_ALLOWED
    if path == "/debug/log":
        return ROUTE_DEBUG_LOG if method == "GET" else ROUTE_METHOD_NOT_ALLOWED
    if path == "/mcp/tools":
        return ROUTE_MCP_TOOLS if method == "GET" else ROUTE_METHOD_NOT_ALLOWED
    if path == "/mcp/call":
        return ROUTE_MCP_CALL if method == "POST" else ROUTE_METHOD_NOT_ALLOWED
    if path == "/mcp":
        return ROUTE_MCP if method == "POST" else ROUTE_METHOD_NOT_ALLOWED
    if path == "/v1/attachments/text":
        return (
            ROUTE_ATTACHMENT if method == "POST" else ROUTE_METHOD_NOT_ALLOWED
        )
    return ROUTE_NOT_FOUND


def status_reason(status: Int) -> String:
    if status == 200:
        return "OK"
    if status == 400:
        return "Bad Request"
    if status == 404:
        return "Not Found"
    if status == 405:
        return "Method Not Allowed"
    if status == 413:
        return "Payload Too Large"
    if status == 422:
        return "Unprocessable Entity"
    if status == 500:
        return "Internal Server Error"
    if status == 501:
        return "Not Implemented"
    return "Status"


def build_http_response(
    status: Int, content_type: String, body: String
) -> String:
    """One complete HTTP/1.1 response, Connection: close."""
    return (
        "HTTP/1.1 "
        + String(status)
        + " "
        + status_reason(status)
        + "\r\nContent-Type: "
        + content_type
        + "\r\nContent-Length: "
        + String(body.byte_length())
        + "\r\nConnection: close\r\n\r\n"
        + body
    )
