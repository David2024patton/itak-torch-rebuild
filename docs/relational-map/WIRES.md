# Relational map: wires

Every connection between the nodes in NODES.md, traced both ways:
the wire itself, and the backlink (the reverse relationship, named,
or "none"). Built from the code at branch head b5522f2
(mojo-max-engine), 2026-11-18: Cargo manifests for the Rust wires,
imports for the Go and Mojo wires.

Wires to or from PLANNED and ABSENT nodes are marked in the Kind
column. They describe the shape the plan promises, not code that
runs today.

## Rust workspace wires

| ID | From | To | Kind | Backlink |
| :--- | :--- | :--- | :--- | :--- |
| W01 | RS-ENGINE | RS-SEAM | hosts (serves the protocol on the socket) | RS-SEAM is served by RS-ENGINE |
| W02 | RS-ENGINE | RS-GGUF | calls (parse and dequantize) | RS-GGUF hands parsed files to RS-ENGINE |
| W03 | RS-ENGINE | RS-TOK | calls (encode and decode) | RS-TOK tokenizes for RS-ENGINE |
| W04 | RS-ENGINE | RS-LOADER | calls (load model behind the coverage gate) | RS-LOADER loads models for RS-ENGINE |
| W05 | RS-ENGINE | RS-GLIMMER | calls (architecture forward) | RS-GLIMMER returns logits to RS-ENGINE |
| W06 | RS-ENGINE | RS-CPU | calls (Compute backend; the only backend the binary depends on) | RS-CPU executes the forward for RS-ENGINE |
| W07 | RS-LOADER | RS-GGUF | reads (implements traits over the parsed file) | RS-GGUF is the file layer RS-LOADER bridges from |
| W08 | RS-LOADER | RS-GLIMMER | writes (implements its Metadata and TensorSource traits) | RS-GLIMMER receives metadata and tensors through RS-LOADER |
| W09 | RS-LOADER | RS-DFLASH | reads (draft file census and conditioning inputs) | RS-DFLASH files are inventoried through RS-LOADER |
| W10 | RS-CPU | RS-GLIMMER | reads (implements its Compute trait) | RS-GLIMMER runs on RS-CPU |
| W11 | RS-CPU | RS-MATH | calls (reference f32 arithmetic) | RS-MATH returns activations to RS-CPU |
| W12 | RS-CPU | RS-GGUF | calls (dequantization) | RS-GGUF dequantizes for RS-CPU |
| W13 | RS-CPU | RS-ATTN | reads and writes (KV storage) | RS-ATTN holds cache state for RS-CPU |
| W14 | RS-WGPU | RS-GLIMMER | reads (implements its Compute trait) | RS-GLIMMER can run on RS-WGPU (bench path) |
| W15 | RS-WGPU | RS-GGUF | calls (dequantization and upload) | RS-GGUF supplies weights to RS-WGPU |
| W16 | RS-WGPU | RS-ATTN | reads (KV shapes and masks) | RS-ATTN defines the cache layout RS-WGPU mirrors on device |
| W17 | RS-WGPU | RS-KV | reads and writes (prefix snapshots through the store seam) | RS-KV serves snapshot capture and restore to the RS-WGPU resident session |
| W18 | RS-WGPU | RS-DFLASH | calls (device DFlash session) | RS-DFLASH drafts on device for RS-WGPU |
| W19 | RS-KV | RS-ATTN | reads and writes (snapshots captured from and restored into KvCache) | RS-ATTN holds the cache state RS-KV persists |
| W20 | RS-ARCH | RS-GGUF | reads (metadata and tensors) | RS-GGUF supplies RS-ARCH |
| W21 | RS-ARCH | RS-ATTN | reads and writes (KV cache) | RS-ATTN holds cache state for RS-ARCH |
| W22 | RS-ARCH | RS-MATH | calls (reference f32 arithmetic) | RS-MATH returns activations to RS-ARCH |
| W23 | RS-DFLASH | RS-MATH | calls (reference f32 arithmetic) | RS-MATH returns activations to RS-DFLASH |
| W24 | RS-SPEC | RS-DFLASH | calls (block drafting) | RS-DFLASH drafts blocks for RS-SPEC |
| W25 | RS-SPEC | RS-GLIMMER | calls (target forward for verification) | RS-GLIMMER verifies drafts for RS-SPEC |
| W26 | RS-SPEC | RS-CPU | calls (Compute backend) | RS-CPU executes for RS-SPEC |
| W27 | RS-SPEC | RS-WGPU | calls (resident GPU target driver) | RS-WGPU runs the resident target for RS-SPEC |
| W28 | RS-TOK | RS-GGUF | reads (vocabulary and merges data; no crate dependency by design) | RS-GGUF supplies the vocabulary RS-TOK builds from |
| W29 | RS-GGUF | EXT-MODELS | reads (mmap of the model file) | EXT-MODELS are opened by RS-GGUF |
| W30 | RS-WGPU | EXT-SERVER | runs on (Vulkan via wgpu) | EXT-SERVER hosts RS-WGPU |
| W31 | RS-ENGINE | EXT-LLAMA | compared against (benchmarks and goldens) | none |

## Go platform wires

| ID | From | To | Kind | Backlink |
| :--- | :--- | :--- | :--- | :--- |
| W32 | EXT-CLIENT | GO-SERVER | calls (HTTPS API) | GO-SERVER returns responses and streams tokens (SSE) to EXT-CLIENT |
| W33 | GO-TORCHD | GO-SERVER | hosts (builds and runs the HTTP server) | GO-SERVER is wired up by GO-TORCHD |
| W34 | GO-TORCHD | GO-SUPER | hosts (starts supervision when an engine binary is configured) | GO-SUPER reports lifecycle state to GO-TORCHD |
| W35 | GO-TORCHD | GO-ENGINE | calls (DialUnix when a socket is configured) | GO-ENGINE connects GO-TORCHD to the seam |
| W36 | GO-SERVER | GO-AUTH | calls (middleware wraps every route but /healthz) | GO-AUTH guards GO-SERVER |
| W37 | GO-SERVER | GO-CACHE | reads and writes (exact match response cache) | GO-CACHE serves cached responses to GO-SERVER |
| W38 | GO-SERVER | GO-TEMPLATE | calls (renders the model's chat template) | GO-TEMPLATE renders prompts for GO-SERVER |
| W39 | GO-SERVER | GO-DIAG | writes (request, route, and error counters) | GO-DIAG is read back through the GO-SERVER /debug route |
| W40 | GO-SERVER | GO-SCHED | calls (Acquire before generation) | GO-SCHED grants generation slots to GO-SERVER |
| W41 | GO-SERVER | GO-ENGINE | calls (Client interface: Generate, Embed, Health) | GO-ENGINE returns tokens, vectors, and health to GO-SERVER |
| W42 | GO-SUPER | GO-ENGINE | reads (implements Client by delegating to the socket client) | GO-ENGINE is the surface GO-SUPER presents as |
| W43 | GO-SUPER | RS-ENGINE | hosts (subprocess: start, health wait, restart within budget) | RS-ENGINE reports lifecycle state to GO-SUPER |
| W44 | GO-ENGINE | RS-SEAM | writes (NDJSON requests over the unix socket) | RS-SEAM returns token frames and engine health to GO-ENGINE |
| W45 | GO-AUTH | GO-DIAG | calls (FailureRecorder interface, no import) | GO-DIAG records auth failures for GO-AUTH |
| W46 | EXT-ORIG | GO-SERVER | informs (successor scope and audit record) | GO-SERVER is part of the successor of EXT-ORIG |
| W47 | GO-SERVER | EXT-SERVER | hosts on (the daemon runs on the home server) | EXT-SERVER hosts GO-SERVER |

## Mojo engine wires

| ID | From | To | Kind | Backlink |
| :--- | :--- | :--- | :--- | :--- |
| W48 | MJ-MAIN | MJ-CONFIG | calls (EngineConfig.load) | MJ-CONFIG configures MJ-MAIN |
| W49 | MJ-MAIN | MJ-GGUF | calls (loader state report) | MJ-GGUF reports through MJ-MAIN |
| W50 | MJ-RUNNER | MJ-GEN | calls (generate and generate_device) | MJ-GEN generates for MJ-RUNNER |
| W51 | MJ-RUNNER | MJ-MODEL | calls (builds GpuGlimmerModel and the CPU model) | MJ-MODEL runs forwards for MJ-RUNNER |
| W52 | MJ-RUNNER | MJ-TOK | calls (vocab load, encode, decode) | MJ-TOK tokenizes for MJ-RUNNER |
| W53 | MJ-RUNNER | MJ-SERVE | hosts (the serve accept loop dispatches into the serve modules) | MJ-SERVE is served by MJ-RUNNER |
| W54 | MJ-MODEL | MJ-GGUF | reads (tensor reads through the loader) | MJ-GGUF supplies MJ-MODEL |
| W55 | MJ-MODEL | MJ-KV | reads and writes (CPU KV cache) | MJ-KV holds cache state for the CPU forward in MJ-MODEL |
| W56 | MJ-MODEL | MJ-ATTN | calls (attention, CPU twin and GPU path) | MJ-ATTN attends for MJ-MODEL |
| W57 | MJ-MODEL | MJ-KERNELS | calls (GEMV and GEMM kernels) | MJ-KERNELS multiply for MJ-MODEL |
| W58 | MJ-MODEL | MJ-OPS | calls (norm, RoPE, SwiGLU, embedding, final logit) | MJ-OPS returns rows to MJ-MODEL |
| W59 | MJ-MODEL | MJ-GPU | calls (device runtime, device KV, resident weights) | MJ-GPU hosts the device side of MJ-MODEL |
| W60 | MJ-MODEL | MJ-GEN | reads (implements the ForwardModel trait the loop drives) | MJ-GEN drives MJ-MODEL through the trait |
| W61 | MJ-GPU | MJ-KV | reads (device KV mirrors the cache layout) | MJ-KV defines the layout MJ-GPU mirrors |
| W62 | MJ-ATTN | MJ-OPS | calls (rope_cpu and softmax_cpu) | MJ-OPS serves MJ-ATTN |
| W63 | MJ-TOK | MJ-GGUF | reads (vocab from the loader) | MJ-GGUF supplies MJ-TOK |
| W64 | MJ-SERVE | MJ-MODEL | calls (chat completions run the model) | MJ-MODEL generates for MJ-SERVE |
| W65 | MJ-SERVE | MJ-GEN | calls (generation loop and greedy sampler) | MJ-GEN generates for MJ-SERVE |
| W66 | MJ-SERVE | MJ-FACT | calls (model fact sheets for /v1/models) | MJ-FACT describes models for MJ-SERVE |
| W67 | MJ-SERVE | MJ-HIST | reads and writes (run records, history, leaderboard, stats) | MJ-HIST is served through MJ-SERVE |
| W68 | MJ-SERVE | MJ-CONFIG | reads (serve limits and caps) | MJ-CONFIG bounds MJ-SERVE |
| W69 | MJ-DASH | MJ-SERVE | reads (fetches /openapi.json and the JSON feeds over HTTP) | MJ-SERVE feeds MJ-DASH |
| W70 | EXT-CLIENT | MJ-SERVE | calls (HTTP API and the MCP style bridge) | MJ-SERVE returns responses to EXT-CLIENT |
| W71 | MJ-GPU | EXT-SERVER | runs on (MAX device on the RX 7900 XTX) | EXT-SERVER hosts MJ-GPU |
| W72 | MJ-GGUF | EXT-MODELS | reads (staged tensor reads from the model file) | EXT-MODELS are opened by MJ-GGUF |
| W73 | MJ-MODEL | MJ-DIAG | emits (phase lines through the diag seam; most modules do the same) | MJ-DIAG carries MJ-MODEL diagnostics out |
| W74 | MJ-RUNNER | EXT-LLAMA | compared against (dated bench notes, same card, same file) | none |

## Support wires

| ID | From | To | Kind | Backlink |
| :--- | :--- | :--- | :--- | :--- |
| W75 | SUP-GOLDEN | RS-GLIMMER | verifies (golden token records and layer diffs) | RS-GLIMMER is checked against SUP-GOLDEN |
| W76 | SUP-GOLDEN | MJ-GEN | verifies (the golden first 8 tokens the runner must match) | MJ-GEN is checked against SUP-GOLDEN |
| W77 | SUP-BENCH | RS-WGPU | measures (gate.sh runs the workspace tests; notes record kernel benches) | RS-WGPU results land in SUP-BENCH |
| W78 | SUP-BENCH | MJ-GPU | measures (wave notes record card windows, witness logs, speeds) | MJ-GPU results land in SUP-BENCH |
| W79 | SUP-DOCS | MJ-MODEL | informs (DESIGN.md is the design record the engine is built from) | MJ-MODEL work updates SUP-DOCS |

## Planned and absent wires

| ID | From | To | Kind | Backlink |
| :--- | :--- | :--- | :--- | :--- |
| W80 | GO-SERVER | PL-CORE | reads (PLANNED: contracts, health, events) | PL-CORE is imported by GO-SERVER |
| W81 | EXT-GOAGENT | PL-CORE | informs (PLANNED: debug and event ports descend from it) | PL-CORE descends from EXT-GOAGENT |
| W82 | PL-TRAIN | RS-ENGINE | writes (PLANNED: trained heads and adapters) | RS-ENGINE loads artifacts written by PL-TRAIN |
| W83 | GO-SERVER | PL-TRAIN | calls (PLANNED: training API) | PL-TRAIN reports progress to GO-SERVER |
| W84 | PL-DISSECT | RS-GGUF | writes (PLANNED: .itf files the loader reads) | RS-GGUF reads what PL-DISSECT writes |
| W85 | GO-SERVER | EXT-REG | reads (PLANNED: model pulls; no pull code exists today) | EXT-REG serves pulls to GO-SERVER |
| W86 | MJ-SERVE | PL-VISION | refuses (PLANNED path: images answered vision_not_ready until it lands) | PL-VISION lands behind the MJ-SERVE refusal |
| W87 | MJ-GEN | PL-SAMPLER | extends (PLANNED: the sampler family grows out of the greedy sampler) | PL-SAMPLER completes MJ-GEN |
| W88 | MJ-MODEL | PL-QWEN | extends (PLANNED: the qwen35 architecture lands in the model module) | PL-QWEN runs inside MJ-MODEL |
| W89 | MJ-KERNELS | PL-F1B | extends (PLANNED: striping lands in the GEMV kernels) | PL-F1B speeds up MJ-KERNELS |
| W90 | MJ-MODEL | PL-F1B | extends (PLANNED: QKV and Gate/Up fusions land in the layer assembly) | PL-F1B speeds up MJ-MODEL |
| W91 | MJ-KERNELS | PL-F2 | extends (PLANNED: the WMMA prefill path lands in the kernels) | PL-F2 speeds up MJ-KERNELS |
| W92 | MJ-SERVE | PL-STREAM | extends (PLANNED: streaming grows out of the v1 non-streaming API) | PL-STREAM completes MJ-SERVE |
| W93 | MJ-GEN | PL-SPEC | extends (PLANNED: draft based decoding grows out of the generation loop) | PL-SPEC speeds up MJ-GEN |
| W94 | RS-ATTN | PL-KVT | extends (PLANNED: the advanced KV tiers grow out of the built cache) | PL-KVT extends RS-ATTN |
| W95 | RS-WGPU | PL-BACKENDS | extends (PLANNED: new backends land beside the wgpu backend, behind the same Compute trait) | PL-BACKENDS sit beside RS-WGPU |
| W96 | GO-SERVER | PL-GOPHASE6 | extends (PLANNED: the Phase 6 growth lands in the platform) | PL-GOPHASE6 grows GO-SERVER |
| W97 | GO-ENGINE | AB-SHMEM | writes (ABSENT: only a comment promises it; the real plane is W44) | none |
| W98 | SUP-DOCS | AB-CI | informs (ABSENT: RULES.md names the gates as target state) | none |
| W99 | SUP-DOCS | AB-FUZZ | informs (ABSENT: RULES.md names fuzz targets as target state) | none |
| W100 | GO-SERVER | AB-DB | reads and writes (ABSENT: no database calls exist; state is in memory) | none |

## Totals

100 wires: 79 live wires between LIVE nodes, 17 planned wires,
4 absent wires.
