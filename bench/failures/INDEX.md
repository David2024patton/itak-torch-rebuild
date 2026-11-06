# Failure vault index

One line per wrapped run (bench/failure-bundle.sh). FAIL rows
point at a full bundle directory in this folder; PASS rows are
the whole record, because a pass deletes its bulky logs by
design. The convention is bench/failure-bundles-2026-11-06.md.

| UTC | slug | verdict | bundle | command |
| --- | --- | --- | --- | --- |
| 2026-11-06T22:43:48Z | demo-fail | FAIL | 20261106T224349Z-demo-fail | bash -c echo\ \"synthetic\ wrong\ data:\ top5\ mismatch\ at\ row\ 3\"\;\ echo\ \"harness\ noise\ on\ stderr\"\ \>\&2\;\ exit\ 3 |
| 2026-11-06T22:43:52Z | demo-marker | FAIL | 20261106T224352Z-demo-marker | bash -c echo\ \"run\ completed\;\ output\ is\ degenerate:\ every\ top-1\ probability\ is\ exactly\ 1.0\"\;\ exit\ 0 |
| 2026-11-06T22:43:52Z | demo-pass | PASS | - | bash -c echo\ \"golden\ check:\ all\ tokens\ equal\,\ stream\ identical\"\;\ exit\ 0 |
