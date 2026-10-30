#!/usr/bin/env bash
# PURPOSE: Test-skip enforcement (TODO T-1.7): run the Go platform
# tests and the Rust workspace tests, collect every skipped test,
# and fail if a skipped test is not listed in
# tools/test-skip-exceptions.txt. A test that fails still fails here
# too: this script never turns a red suite green.
# MODIFY PER PROJECT: exceptions live only in
# tools/test-skip-exceptions.txt, never in this script.
# USAGE: tools/check-test-skips.sh [go|rust|all]   (default: all;
# CI passes its own language so each job checks only its suite)
# EXIT: 0 suites green and every skip excepted; 1 a suite failed or
# an unlisted skip was found; 2 a required tool is missing.
set -u
cd "$(dirname "$0")/.." || exit 2
ROOT="$PWD"
EXCEPTIONS="$ROOT/tools/test-skip-exceptions.txt"
LANG_SEL="${1:-all}"
case "$LANG_SEL" in go|rust|all) ;; *) echo "usage: $0 [go|rust|all]" >&2; exit 2 ;; esac
[ -x "$HOME/goroot/go/bin/go" ] && PATH="$HOME/goroot/go/bin:$PATH"
[ -d "$HOME/.cargo/bin" ] && PATH="$HOME/.cargo/bin:$PATH"
if [ "$LANG_SEL" != "rust" ]; then
  command -v go >/dev/null 2>&1 || { echo "go not found" >&2; exit 2; }
fi
if [ "$LANG_SEL" != "go" ]; then
  command -v cargo >/dev/null 2>&1 || { echo "cargo not found" >&2; exit 2; }
fi

fail=0
skips_file="$(mktemp)"
trap 'rm -f "$skips_file"' EXIT

if [ "$LANG_SEL" != "rust" ]; then
  echo "== go test (platform)"
  go_out="$(cd platform && go test -v ./... 2>&1)"
  go_status=$?
  printf '%s\n' "$go_out" | tail -n 12
  [ "$go_status" != 0 ] && { echo "FAIL: go test suite failed" >&2; fail=1; }
  printf '%s\n' "$go_out" | sed -n 's/.*--- SKIP: \([A-Za-z0-9_]*\).*/\1/p' >> "$skips_file"
fi

if [ "$LANG_SEL" != "go" ]; then
  echo "== cargo test (workspace)"
  rust_out="$(cargo test --workspace 2>&1)"
  rust_status=$?
  printf '%s\n' "$rust_out" | grep -E "test result" || true
  [ "$rust_status" != 0 ] && { echo "FAIL: cargo test suite failed" >&2; fail=1; }
  printf '%s\n' "$rust_out" | sed -n 's/^test \([^ ]*\) \.\.\. ignored$/\1/p' >> "$skips_file"
fi

sort -u -o "$skips_file" "$skips_file"
if [ -s "$skips_file" ]; then
  echo "== skipped tests found:"
  cat "$skips_file"
  while IFS= read -r name; do
    [ -z "$name" ] && continue
    if grep -qx "$name" "$EXCEPTIONS"; then
      echo "excepted skip (listed): $name"
    else
      echo "FAIL: unlisted skipped test: $name (add it to tools/test-skip-exceptions.txt with a reason, or make it run)" >&2
      fail=1
    fi
  done < "$skips_file"
else
  echo "== skipped tests found: none"
fi

[ "$fail" = 1 ] && { echo "SKIP CHECK FAILED" >&2; exit 1; }
echo "SKIP CHECK PASSED"
