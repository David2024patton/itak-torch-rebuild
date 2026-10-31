<!--
INSTRUCTIONS FOR THIS FILE (user and AI: read before editing. Keep this header.)

WHAT THIS FILE IS: the operations book for this project, at docs/RUNBOOK.md.
How to build it, run it, deploy it, verify it, and where everything lives,
so anyone can operate the project without tribal knowledge.

EXACT STRUCTURE:
- "## Build": the real commands, copied from the working build.
- "## Run locally": how to start it and what you should see.
- "## Deploy": where it is hosted, how a deploy happens, how to roll back.
- "## Verify": the checks that prove it works after a change or deploy.
- "## Where things live": hosts, services, dashboards, and WHERE
  credentials are stored (names of vault entries or connectors only).
- "## Gotchas": the traps already learned the hard way, each with its fix.

RULES:
- NEVER a credential value in this file: no passwords, keys, or tokens.
  Pointers to secure storage only.
- Commands are copied from what actually works; if a command changes,
  this file changes in the same commit.
- No secrets, passwords, or tokens in this file, ever.
-->
# RUNBOOK: iTaK Torch Rebuild

## Build

Toolchains are pinned: `rust-toolchain.toml` at the repo root (rust 1.99.1) and the platform's `go.mod` toolchain line (go1.26.10). Build both halves from the repo root:

```bash
# Rust workspace: the engine binary plus the eleven library crates
cargo build --release -p itak-engine

# Go platform: the torchd daemon
(cd platform && go build -o torchd ./cmd/torchd)
```

## Run locally

The pair serves as two processes meeting on a Unix socket. Either start the engine by hand and point the platform at its socket, or let torchd supervise the engine. The full operator walkthrough is in SERVING.md; the short form:

```bash
# Engine by hand (loads the model; ready when the log says "serving on ...")
./target/release/itak-engine \
  --model /path/to/model.gguf \
  --socket /tmp/itak-engine.sock

# Platform against the socket (binds 127.0.0.1:8091)
TORCH_API_KEY=<your-key> ./platform/torchd \
  -engine-socket /tmp/itak-engine.sock -models glimmer
```

Engine flags, each with an `ITAK_ENGINE_*` environment twin; nothing is compiled in:

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--model` | (required) | GGUF file to load |
| `--socket` | `/tmp/itak-engine.sock` | Unix socket to bind; created owner-only (0600) |
| `--model-name` | file metadata name, else file stem | Name requests must address |
| `--default-max-tokens` | 128 | Generation cap when a request omits `max_tokens` |
| `--slots` | 1 | Concurrent generations, each with its own compute and KV state |
| `--max-queued` | 8 | Jobs that may wait for a slot; past capacity refuses as `engine_busy` |
| `--max-frame-bytes` | 4194304 | Largest single NDJSON frame the engine accepts |
| `--max-embed-inputs` | 256 | Largest embed batch in one request |
| `--max-embed-tokens` | 8192 | Largest token count in one embed input |
| `--log-content` | off | Also log prompt ids and generated text; counts and timings only by default |

The socket refuses what it should: a first frame that is not a hello, a peer running under a different uid, an oversized or unterminated frame, and JSON nested past the parser's depth cap all get named refusals, and the engine never deletes a non-socket file found at the socket path.

torchd flags (the serving-relevant set; `torchd -h` prints them all): `-addr` (127.0.0.1:8091), `-engine-socket`, `-engine-bin` with `-engine-model` and `-engine-name` for supervised mode, `-engine-slots`, `-engine-max-queued`, `-engine-max-restarts` (5) and `-engine-restart-window` (5m) for the restart budget, `-models`, `-max-concurrent` (4), `-max-tokens` (8192 platform cap), `-generation-timeout` (1h), `-max-body-bytes` (1 MiB), `-read-header-timeout` (10s), `-idle-timeout` (90s), `-max-waiters-per-lane` (128), `-cache-ttl` (5m), `-cache-max-entries` (1024), `-cache-purge-interval` (1m). The daemon refuses to start without `TORCH_API_KEY` (minimum 16 characters) unless `TORCH_ALLOW_NO_AUTH=1` is set deliberately.

Expected: an authenticated chat completion streams or returns from the engine; without the key, 401; with no engine configured, engine routes answer 503 `engine_unavailable`. The recorded smoke transcript is bench/serving-smoke-2026-10-30.md.

## Deploy

There is no automated deploy for this stack, and production Glimmer serving is down on the owner's explicit order until the engine work is finished; it stays down. Nothing in this repo restarts it. When serving is approved, a deploy is: build both binaries on the target host, start the engine (by hand or supervised by torchd), and run the Verify section against it. Rollback is the previous release binaries, or `git revert` and a rebuild; the pair holds no state that a rollback can strand (the response cache is in-memory, engine state is per-process).

## Verify

The gates, as plain commands (there are no gate scripts; do not invent any):

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo deny check
(cd platform && go build ./... && go vet ./... && go test ./... && govulncheck ./...)
```

Under the pinned go1.26.10 toolchain, govulncheck reports no vulnerabilities in the platform (this closes the panel's GO-2026-6809 finding, which was a standard library advisory against the older toolchain). After a serving change, repeat the smoke from SERVING.md and compare against bench/serving-smoke-2026-10-30.md. Benchmarks: compare new runs against docs/research/2026-10-26-glimmer-7900xtx-benchmarks.md and the bench/ notes for the same harness shape; anchors are harness-relative (DECISIONS.md, 2026-10-30), so a result more than a few percent off the recorded numbers for the same harness is a finding, not a rounding error. Accuracy: bench/accuracy-prompts.md at temp 0; empty content means the reasoning default trap is back.

## Where things live

- Home server: reached over Tailscale by its SSH config alias (ts-server). It runs Dokploy and live business sites; treat it as production at all times. The server is never rebooted on an agent's call.
- Model weights: on the server at /mnt/models/ (the 2TB Kioxia NVMe). The Glimmer file set lives in /mnt/models/Muse-Glimmer-30B/ (the Q4_K_M target, its mmproj, and the DFlash draft). The weights were never deleted. Never put model-sized files on the root disk.
- Reference stack on the server: llama.cpp trees build-vulkan (the reference serving path) and build-rocm. The server runs the Ubuntu native ROCm 7.1 stack; llama.cpp works on both Vulkan and HIP there, and Vulkan is the serving pick on the RX 7900 XTX (HIP wins prompt processing, Vulkan wins decode).
- The original engine: GitHub repos David2024patton/iTaKTorchQ4 (public) and itak-torch (private twin), plus the GOAgent repo, which still hosts the pkg/debug and pkg/eventbus ancestors Phase 1 ports from.
- Credentials: GitHub access goes through the stored custom.github connector credential. No credential value appears in this repo, ever.

## Gotchas

- **ROCm was broken, and is fixed; do not re-litigate it.** The 2026-10-26 symptom set (HIP loads, then spins at 0% GPU busy) traced to a mixed-version ROCm userland. The fix landed 2026-10-28: a single Ubuntu native ROCm 7.1 stack, proven from hello-world to a full 30B bench (STATUS.md; TODO T-P.2). Vulkan remains the serving pick on this card on decode speed, not because HIP is broken.
- **Total context is split across slots in the reference stack.** llama-server with -c 131072 and -np 4 gives each slot 32768 tokens. A poisoned oversized request can leave its slot failing with context-size errors until eviction; do not read that as a model fault.
- **Reasoning models eat the budget by default.** At default reasoning strength, Glimmer spent the entire 512-token cap thinking and returned empty content on code and logic prompts. Inject reasoning_strength low (chat template kwargs) at the proxy by default.
- **Never strip special tokens.** The model's channel protocol rides on its special tokens; a proxy that strips them collapses reasoning, tool calls, and replies into one mangled stream. Always run llama-server with --jinja, and the rebuild's own template renderer keeps them by construction.
- **Root disk discipline.** Model-sized files on the root NVMe filled the disk on 2026-10-27 and took Dokploy down. Weights go on the models drive, full stop.
- **A dead engine under supervision stays down when the budget is spent.** torchd restarts a crashed engine within its restart budget (default 5 restarts per 5 minutes). A model that can no longer load exhausts the budget and stays down, visibly, in /debug under engine_debug.supervisor; that is the designed behavior, not a hang.
- **The supervisor's child gets an allowlisted environment.** torchd passes the engine only PATH and ITAK_ENGINE_* variables. If a supervised engine seems to miss a setting, check the name is an ITAK_ENGINE_* twin before suspecting the flag wiring.
