# Repro: demo-fail

## Command

```sh
bash -c echo\ \"synthetic\ wrong\ data:\ top5\ mismatch\ at\ row\ 3\"\;\ echo\ \"harness\ noise\ on\ stderr\"\ \>\&2\;\ exit\ 3
```

Run from the repo root (the wrapper's working directory),
with the environment in this bundle's env.txt.

## Pass condition that failed

Expected: exit code 0 and none of the failure markers
(FAIL FAILED mismatch DIVERGES REFUSED degenerate) in the combined stdout+stderr.

Observed: exit code 3; failure markers matched: mismatch(1).

## State at the run

- git: 2f8b3ae1e20a064e6df271bd677aee1d0de3c337 (branch lane/failure-bundles, working tree dirty)
- model: none detected sha256 n/a
- full output: stdout.log and stderr.log in this bundle
