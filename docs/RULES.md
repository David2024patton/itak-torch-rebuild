<!--
INSTRUCTIONS FOR THIS FILE (user and AI: read before editing. Keep this header.)

WHAT THIS FILE IS: the build law for this repo in one page, at docs/RULES.md.
It states how every module in this project gets built, gated, and proven.
Project-specific numbers and additions go in a "Project specifics" section
at the end; the law sections above it come from the github-repo-standard
skill template and change only when the skill changes.

RULES:
- If a project practice conflicts with this page, this page wins until
  David changes it.
- No secrets, passwords, or tokens in this file, ever.
-->
# RULES: how this project gets built

The build law for this repo, in one page. These rules come from David
Patton and apply to every module, every language, every time.

## Modularity: the grid and the tiles
- The main program or page is a GRID. Every section is a MODULE plugged
  into a grid slot: header, footer, each content block, each tile.
  Modular to the full max, never partial.
- Every feature is its own isolated module with a defined interface.
  One feature, one module, one commit, one push.
- Modules are independently replaceable and updatable on the fly. A
  broken module gets fixed or swapped alone; the system never rebuilds
  around it.

## Build 20 years ahead
- Design for the bottleneck that is COMING, not just today's.
- Interface seams at every change-prone boundary: model providers,
  executors, verifiers, storage, transports.
- Operational limits (timeouts, caps, retry counts, quotas) are config
  with sane defaults, never constants compiled into logic.
- Data formats carry version fields so future versions read today's data.

## Debugging and linting, in every module, from birth
- They are not the same thing. Linting reads the code for mistakes
  without running it. Debugging is finding out why the program
  misbehaved when it ran, backed by diagnostics built into the module.
- Every module ships with BOTH: structured diagnostics, a way to pull
  the bug information out, and lint gates that run clean. A module that
  cannot say what it did and why it refused is not done.

## Check the shelf before building
- Before building anything, check the take-shelf
  (~/workspace/google-org-harvest/USE-IT.md) and the wider harvest: if
  Google or another trusted source already built it under a usable
  license, take it or retrofit it. Rebuilding what exists is waste.
- For any new market or feature area, research the top 10 competitors
  (the customer's list and the platform's list) before designing.

## Security
- No secrets in the repo, ever: keys, tokens, and passwords live in
  secure storage only. Secret scanning (gitleaks or betterleaks) runs
  in the gates.
- Every refusal, overage, and verdict emits through the project's
  diagnostic channel. Nothing fails silently.
- Untrusted input gets a bounded parser and a fuzz target, not trust.

## Verification
- Done means proven: a test with asserted values, a log line, a bench
  note, or a live check. "It compiles" is not proof.
- Never blindly trust delegated work; re-run the gates before merging.
- Emulation-proven is not bench-proven; label which one a claim is.

## Writing
- Plain, direct, technical. No em dashes in prose, docs, or code.
- Brand casing is exact: AlphaVision, AlphaFlux, iTaK, Mimir,
  Patriot Pest Control (full name wherever there are words).
- Project docs: TODO.md tracks the work, PLAN.md holds the plan,
  IDEA.md holds David's ideas in his words.

## Project specifics

- **No garbage-collected language inside the token loop.** The engine
  crates allocate once and pool. A change that allocates per token is a
  design defect, not a style issue, and the allocation counter test in
  TODO T-2.4 is its gate.
- **Contracts are published and versioned.** Shared types live in
  itak-core (Go) and itak-types (Rust), both versioned artifacts with
  version fields in the data. Relative-path module references are
  forbidden; that is the failure that killed the original engine.
- **Benchmarks are claimed only from real hardware.** Emulation, other
  people's cards, and vendor numbers are labelled as such wherever they
  appear. This project's numbers come from the RX 7900 XTX record in
  docs/research/ or they do not appear.
- **Experiments behind flags, off by default.** Mamba-3 mode,
  cache-aware MoE routing, and linear-attention swaps enter as config
  flags and earn defaults only with proof filed in this repo
  (DECISIONS.md records the promotion).
- **Weights never on the root disk.** Model files live on the large
  array on the home server; see DECISIONS.md 2026-10-27.
- **The server never reboots for this project.** Any step that appears
  to need one stops and is reported instead (see docs/RUNBOOK.md).
