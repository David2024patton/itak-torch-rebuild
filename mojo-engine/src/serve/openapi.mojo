# openapi.mojo - the static OpenAPI 3 description served at
# GET /openapi.json. This JSON is the contract the dashboard
# (and any other client) reads: it is built from the same
# route table http.mojo dispatches on, and a suite test keeps
# the two in step (every path here must route, every route
# must appear here). Non-streaming is stated in the chat
# operation's description, per the v1 contract.

from config import ENGINE_NAME, ENGINE_VERSION
from serve.json import json_escape


def openapi_json(model_name: String) -> String:
    var out = String("{")
    out += '"openapi":"3.0.3"'
    out += ',"info":{"title":"' + ENGINE_NAME + ' API"'
    out += ',"version":"' + ENGINE_VERSION + '"'
    out += (
        ',"description":"Mojo-only inference endpoint. Greedy sampler only in'
        " v1 (the greedy lock: temperature must be 0.0; top_k, top_p and seed"
        " refuse serve_sampler_not_available); chat completions are"
        " non-streaming. Image input is accepted in the request shape and"
        " refused by name (vision_not_ready) until a vision path lands."
        " Every run is recorded to the on-disk history with its source tag"
        ' (dashboard, cli, mcp) and full telemetry."}'
    )
    out += ',"paths":{'
    out += (
        '"/health":{"get":{"summary":"Engine name, version, model,'
        ' status","responses":{"200":{"description":"Health JSON"}}}}'
    )
    out += (
        ',"/v1/models":{"get":{"summary":"Model registry: every model file in'
        " the configured models directory plus the served model, each with"
        ' its fact sheet","responses":{"200":{"description":"Registry'
        ' JSON"}}}}'
    )
    out += (
        ',"/v1/models/{id}":{"get":{"summary":"One model fact sheet by id'
        ' (file base name without .gguf)","responses":{"200":{"description":'
        '"Fact sheet JSON"},"404":{"description":"serve_model_not_found"}}}}'
    )
    out += (
        ',"/history":{"get":{"summary":"Run history (JSON lines on disk,'
        ' survives restart), filterable by ?source= and'
        ' ?model=","responses":{"200":{"description":"History JSON"}}}}'
    )
    out += (
        ',"/history/export":{"get":{"summary":"Run bundle export'
        ' (mojo-engine-run-bundle, format_version 1, unsigned in'
        ' v1)","responses":{"200":{"description":"Bundle JSON"}}}}'
    )
    out += (
        ',"/v1/runs/ingest":{"post":{"summary":"Import a run bundle: every'
        " record validated against the schema, bad records refused by name"
        " and counted, imported rows marked imported with origin"
        ' hardware","responses":{"200":{"description":"Accepted/rejected'
        ' counts with refusal names"}}}}'
    )
    out += (
        ',"/leaderboard":{"get":{"summary":"Leaderboard: measured rows per'
        " model + hardware ranked by best decode tok/s, plus reference rows"
        " (llama.cpp, paused Rust) marked"
        ' reference","responses":{"200":{"description":"Leaderboard'
        ' JSON"}}}}'
    )
    out += (
        ',"/v1/chat/completions":{"post":{"summary":"OpenAI-shaped chat'
        " completion (non-streaming, greedy). Accepts prompt or messages,"
        " optional context_text (attachment text), max_tokens, and the bench"
        " fields source, attachments (file names) and quality_verdict for"
        ' the run record.","responses":{"200":{"description":"chat.completion'
        " JSON with usage, timings (prefill_ms and prefill_tok_s; ttft_ms is"
        ' null in the non-streaming shape) and the sampler settings used'
        ' (greedy, temperature 0.0)"},"422":{"description":"Named refusal,'
        " e.g. vision_not_ready or"
        ' serve_sampler_not_available"}}}}'
    )
    out += (
        ',"/v1/attachments/text":{"post":{"summary":"Validate browser-extracted'
        " attachment text (.txt/.md/.pdf/.zip/.7z) against the context cap;"
        ' stores nothing","responses":{"200":{"description":"Filename, char'
        ' count, text"}}}}'
    )
    out += (
        ',"/stats":{"get":{"summary":"In-memory serve'
        ' counters","responses":{"200":{"description":"Stats JSON"}}}}'
    )
    out += (
        ',"/debug/log":{"get":{"summary":"Recent diag-seam lines from the serve'
        ' ring buffer","responses":{"200":{"description":"Debug log JSON"}}}}'
    )
    out += (
        ',"/openapi.json":{"get":{"summary":"This'
        ' document","responses":{"200":{"description":"OpenAPI JSON"}}}}'
    )
    out += (
        ',"/":{"get":{"summary":"Testing chat dashboard (static HTML, renders'
        " from the JSON"
        ' endpoints)","responses":{"200":{"description":"HTML"}}}}'
    )
    out += (
        ',"/mcp/tools":{"get":{"summary":"MCP-style bridge: tool'
        ' list","responses":{"200":{"description":"Tools JSON"}}}}'
    )
    out += (
        ',"/mcp/call":{"post":{"summary":"MCP-style bridge: invoke one tool by'
        " name (chat, health, stats,"
        ' debug_log)","responses":{"200":{"description":"Tool result JSON"}}}}'
    )
    out += (
        ',"/mcp":{"post":{"summary":"MCP-style bridge: JSON-RPC initialize /'
        " tools/list / tools/call"
        ' subset","responses":{"200":{"description":"JSON-RPC response"}}}}'
    )
    out += "}"
    out += ',"x-model":"' + json_escape(model_name) + '"'
    out += "}"
    return out
