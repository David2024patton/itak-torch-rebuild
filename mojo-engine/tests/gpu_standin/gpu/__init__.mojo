# gpu package STAND-IN for Lane C2's VM tests and any
# pre-integration server proof only.
#
# This directory is NOT src/gpu/ (that package is Lane C1's).
# It lives under tests/gpu_standin/ and mirrors, with simpler
# internals, the exact public surface of C1's built package
# (context / weights / device_kv) that Lane C2's src code
# codes against, so Lane C2's modules and VM tests compile
# and run in this lane's own worktree before integration.
# The server proof of record runs the same src code against
# C1's REAL package in a merged staging tree; the stand-in's
# simplifications (no budget ledger beyond a cap sum, no
# slot-order metas beyond what accessors need) are internal
# only and are never presented as C1 verdicts.
