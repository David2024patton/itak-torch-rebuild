# Failure vault index

One line per wrapped run (bench/failure-bundle.sh). FAIL rows
point at a full bundle directory in this folder; PASS rows are
the whole record, because a pass deletes its bulky logs by
design. The convention is bench/failure-bundles-2026-11-06.md.

What a committed bundle contains: the two bundles below are the
synthetic demos from the tooling proof, not production failures.
By the wrapper's design (bench/failure-bundle.sh), a bundle's
env.txt holds only the resolved WGPU_COMPUTE_* and ITAK_* engine
tuning variables, never the process environment, and its
manifest.txt holds repro facts (host name, kernel string, repo
SHA and branch, timestamps, model path and hash). No bundle
carries credentials; the capture pattern cannot match a key or
token variable. The demo bundles' logs were stripped before
commit under the size rule recorded in their LOGS-STRIPPED.txt.

| UTC | slug | verdict | bundle | command |
| --- | --- | --- | --- | --- |
| 2026-11-06T22:43:48Z | demo-fail | FAIL | 20261106T224349Z-demo-fail | bash -c echo\ \"synthetic\ wrong\ data:\ top5\ mismatch\ at\ row\ 3\"\;\ echo\ \"harness\ noise\ on\ stderr\"\ \>\&2\;\ exit\ 3 |
| 2026-11-06T22:43:52Z | demo-marker | FAIL | 20261106T224352Z-demo-marker | bash -c echo\ \"run\ completed\;\ output\ is\ degenerate:\ every\ top-1\ probability\ is\ exactly\ 1.0\"\;\ exit\ 0 |
| 2026-11-06T22:43:52Z | demo-pass | PASS | - | bash -c echo\ \"golden\ check:\ all\ tokens\ equal\,\ stream\ identical\"\;\ exit\ 0 |
