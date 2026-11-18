# Relational map: nodes

Compact node table for the relational diagram. The full inventory
(what each node owns, and its source file) is INVENTORY.md in this
folder. The connections are WIRES.md. The pictures are DIAGRAM.md.

Built from the code at branch head b5522f2 (mojo-max-engine),
2026-11-18.

State key: LIVE means code exists and does the job. LIVE, paused
means code exists on a line paused by David's 2026-11-07 order
(the Rust and Go trees are the parity scorekeeper). PLANNED means
a doc names it and no code exists. ABSENT means the repo speaks
of it as if present and it is not in the tree.

## External

| ID | Node | Kind | State |
| :--- | :--- | :--- | :--- |
| EXT-CLIENT | Client | external | LIVE |
| EXT-SERVER | RX 7900 XTX server | external | LIVE |
| EXT-MODELS | Model files | external | LIVE |
| EXT-REG | Model registries | external | LIVE |
| EXT-LLAMA | llama.cpp | external | LIVE |
| EXT-ORIG | Original Torch repos | external | LIVE |
| EXT-GOAGENT | GOAgent repo | external | LIVE |

## Rust workspace (paused)

| ID | Node | Kind | State |
| :--- | :--- | :--- | :--- |
| RS-ROOT | Cargo workspace root | workspace | LIVE, paused |
| RS-ENGINE | itak-engine | binary crate | LIVE, paused |
| RS-SEAM | The seam | module | LIVE, paused |
| RS-GGUF | gguf | library crate | LIVE, paused |
| RS-TOK | tokenizer | library crate | LIVE, paused |
| RS-MATH | torch-math | library crate | LIVE, paused |
| RS-ATTN | attention | library crate | LIVE, paused |
| RS-GLIMMER | muse-glimmer | library crate | LIVE, paused |
| RS-ARCH | arch-llama | library crate | LIVE, paused |
| RS-LOADER | itak-loader | library crate | LIVE, paused |
| RS-CPU | cpu-compute | library crate | LIVE, paused |
| RS-WGPU | wgpu-compute | library crate | LIVE, paused |
| RS-DFLASH | dflash | library crate | LIVE, paused |
| RS-KV | itak-kv | library crate | LIVE, paused |
| RS-SPEC | itak-spec | library crate | LIVE, paused |

## Go platform (paused)

| ID | Node | Kind | State |
| :--- | :--- | :--- | :--- |
| GO-TORCHD | torchd | binary | LIVE, paused |
| GO-SERVER | server | package | LIVE, paused |
| GO-AUTH | auth | package | LIVE, paused |
| GO-ENGINE | engine (client) | package | LIVE, paused |
| GO-SCHED | scheduler | package | LIVE, paused |
| GO-CACHE | cache | package | LIVE, paused |
| GO-TEMPLATE | chattemplate | package | LIVE, paused |
| GO-SUPER | supervisor | package | LIVE, paused |
| GO-DIAG | diag | package | LIVE, paused |

## Mojo engine (live line)

| ID | Node | Kind | State |
| :--- | :--- | :--- | :--- |
| MJ-MAIN | main.mojo entry | module | LIVE |
| MJ-CONFIG | config | module | LIVE |
| MJ-DIAG | diag seam | module | LIVE |
| MJ-GGUF | gguf loader (Mojo) | module | LIVE |
| MJ-TOK | tokenizer (Mojo) | module | LIVE |
| MJ-KV | kv cache (Mojo) | module | LIVE |
| MJ-ATTN | attention (Mojo) | module | LIVE |
| MJ-OPS | ops | module | LIVE |
| MJ-KERNELS | kernels | module | LIVE |
| MJ-GPU | gpu runtime | module | LIVE |
| MJ-MODEL | model forward | module | LIVE |
| MJ-GEN | generate and sampler | module | LIVE |
| MJ-RUNNER | runner entry points | module | LIVE |
| MJ-SERVE | serve API | module | LIVE |
| MJ-DASH | dashboard page | page | LIVE |
| MJ-FACT | fact sheets | module | LIVE |
| MJ-HIST | history and stats | module | LIVE |
| MJ-TESTS | Mojo tests and proofs | tests | LIVE |

## Support

| ID | Node | Kind | State |
| :--- | :--- | :--- | :--- |
| SUP-BENCH | bench notes and gate | records | LIVE |
| SUP-GOLDEN | golden fixtures | records | LIVE |
| SUP-DOCS | planning docs | docs | LIVE |

## Planned

| ID | Node | Kind | State |
| :--- | :--- | :--- | :--- |
| PL-F1B | Wave F1-B striping and fusions | module | PLANNED |
| PL-F2 | Wave F2 WMMA prefill | module | PLANNED |
| PL-SAMPLER | Full sampler family (Mojo) | module | PLANNED |
| PL-QWEN | Qwen3.5 architecture port | module | PLANNED |
| PL-VISION | Vision generation path | module | PLANNED |
| PL-STREAM | Streaming chat (Mojo serve) | module | PLANNED |
| PL-SPEC | Mojo speculative decoding wave | module | PLANNED |
| PL-CORE | itak-core (Go contracts) | module | PLANNED |
| PL-TYPES | itak-types (Rust contracts) | module | PLANNED |
| PL-SSM | itak-ssm | module | PLANNED |
| PL-MOE | itak-moe | module | PLANNED |
| PL-TRAIN | itak-train | module | PLANNED |
| PL-DISSECT | itak-dissect and .itf | tool | PLANNED |
| PL-KVT | Advanced KV tiers (Rust) | module | PLANNED |
| PL-BACKENDS | More backends and multi-GPU | module | PLANNED |
| PL-GOPHASE6 | Go platform Phase 6 expansion | service | PLANNED |

## Absent

| ID | Node | Kind | State |
| :--- | :--- | :--- | :--- |
| AB-CI | CI and commit gate script | service | ABSENT |
| AB-FUZZ | Fuzz targets | tests | ABSENT |
| AB-DB | Platform database | service | ABSENT |
| AB-SHMEM | Shared memory data plane | module | ABSENT |

## Totals

72 nodes: 52 LIVE (28 on the live Mojo and support lines, 24 on
the paused Rust and Go lines), 16 PLANNED, 4 ABSENT.
