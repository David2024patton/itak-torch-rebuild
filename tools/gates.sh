#!/usr/bin/env bash
# PURPOSE: Commit gates for this repo (TODO T-1.5): one command runs
# the three tree-wide scans every commit must pass: gitleaks secret
# scan, shellcheck over every shell script (bench scripts included),
# and markdown lint over every doc. Each gate reports PASS or FAIL by
# name; any FAIL makes the whole run fail.
# MODIFY PER PROJECT: markdown rule selection lives in
# .markdownlint-cli2.jsonc and gitleaks allowlists in .gitleaks.toml,
# never in this script. A missing tool is a loud failure (exit 2),
# never a silent skip: a gate that did not run did not pass.
# USAGE: tools/gates.sh
# EXIT: 0 all gates green; 1 a gate found findings; 2 a required
# tool is missing.
set -u
cd "$(dirname "$0")/.." || exit 2
[ -x "$HOME/go/bin/gitleaks" ] && PATH="$HOME/go/bin:$PATH"
[ -x "$HOME/goroot/go/bin/go" ] && PATH="$HOME/goroot/go/bin:$PATH"

fail=0
missing=0
run_gate() {
  name="$1"; shift
  echo "== gate: $name"
  if "$@"; then
    echo "PASS: $name"
  else
    echo "FAIL: $name" >&2
    fail=1
  fi
}

# --- gate 1: secrets -------------------------------------------------
if command -v gitleaks >/dev/null 2>&1; then
  run_gate "gitleaks secret scan" gitleaks dir . --no-banner
else
  echo "FAIL: gitleaks secret scan: gitleaks not installed" >&2
  missing=1
fi

# --- gate 2: shell ---------------------------------------------------
mapfile -t sh_files < <(find . -name '*.sh' -not -path './.git/*' -not -path './target/*')
if [ "${#sh_files[@]}" = 0 ]; then
  echo "== gate: shellcheck (no shell scripts found)"
  echo "PASS: shellcheck"
elif command -v shellcheck >/dev/null 2>&1; then
  run_gate "shellcheck (${#sh_files[@]} scripts)" shellcheck "${sh_files[@]}"
else
  echo "FAIL: shellcheck: shellcheck not installed" >&2
  missing=1
fi

# --- gate 3: markdown ------------------------------------------------
if command -v markdownlint-cli2 >/dev/null 2>&1; then
  run_gate "markdown lint" markdownlint-cli2 "**/*.md"
elif command -v npx >/dev/null 2>&1; then
  run_gate "markdown lint" npx -y markdownlint-cli2 "**/*.md"
else
  echo "FAIL: markdown lint: neither markdownlint-cli2 nor npx installed" >&2
  missing=1
fi

if [ "$missing" = 1 ]; then
  echo "GATES INCOMPLETE: a required tool is missing (see FAIL lines above)" >&2
  exit 2
fi
if [ "$fail" = 1 ]; then
  echo "GATES FAILED" >&2
  exit 1
fi
echo "ALL GATES GREEN"
