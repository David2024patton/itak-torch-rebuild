# serve package: the Mojo-only API endpoint (DESIGN.md step 7,
# decided 2026-11-08: no MAX Serve, no Go socket parity; the
# HTTP server lives inside the engine process, written in
# Mojo). One module per job (grid-and-tiles): socket (POSIX
# transport over libc FFI), http (request parse + response
# build + routing), json (the small JSON reader/writer the
# endpoints need), stats (in-memory counters), openapi (the
# static /openapi.json contract), dashboard (the static page
# at /), server (pure dispatch + response bodies), factsheet
# (per-model fact sheets from a GGUF file's own metadata and
# tensor directory), history (the run-record schema, the
# on-disk JSON-lines history, the leaderboard view, and the
# export/import run bundles). The amendments of 2026-11-08
# (David) folded history/leaderboard/fact sheets in here.
# The accept loop and the generation wiring live in
# src/runner/serve_main.mojo.
