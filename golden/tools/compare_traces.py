#!/usr/bin/env python3
"""Stage-by-stage trace diff for muse-glimmer (FORWARD-DIFF-PLAN).

Usage: compare_traces.py <our_trace.txt> <llama_dump_dir>

Our side is a glimmer_layer_trace dump: trace v=1 lines carrying
whole-tensor and last-row statistics (mean, population variance,
max abs, first values) per stage (embed, layer.<i>.mid,
layer.<i>.out, final_norm, logits). The reference side is a
cb_dump_glimmer output directory (raw f32 .bin + manifest.json);
this script reduces its tensors to the identical statistics and
compares stage by stage in capture order.

Reading rules (from the plan): a stage matches when mean and
variance agree to roughly 1e-2 relative, last row first, whole
tensor second, with the first values tracking. The mean is judged
in units of the stream's own standard deviation (residual means
sit near zero, so a plain relative error is meaningless there);
variance is judged relatively. The first stage whose input stage
matched but which does not itself match is the defect's home.

The reference final_norm and logits tensors carry every position;
only the last position is reduced, matching our single-row
captures. The reference logits also yield the top-1 id and its
softmax probability, for the first-token residue comparison.
"""

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

FIRST_N = 8
LINE_RE = re.compile(
    r"stage=(?P<stage>\S+) rows=(?P<rows>\d+) cols=(?P<cols>\d+) n=(?P<n>\d+) "
    r"mean=(?P<mean>\S+) var=(?P<var>\S+) max_abs=(?P<max_abs>\S+) "
    r"last_mean=(?P<last_mean>\S+) last_var=(?P<last_var>\S+) "
    r"last_max_abs=(?P<last_max_abs>\S+) "
    r"first=\[(?P<first>[^\]]*)\] last_first=\[(?P<last_first>[^\]]*)\]"
)


def parse_first(s):
    return [float(v) for v in s.split(",") if v != ""]


def parse_ours(path):
    stages = {}
    order = []
    for line in Path(path).read_text().splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        d = m.groupdict()
        rec = {
            "rows": int(d["rows"]),
            "cols": int(d["cols"]),
            "mean": float(d["mean"]),
            "var": float(d["var"]),
            "max_abs": float(d["max_abs"]),
            "last_mean": float(d["last_mean"]),
            "last_var": float(d["last_var"]),
            "last_max_abs": float(d["last_max_abs"]),
            "first": parse_first(d["first"]),
            "last_first": parse_first(d["last_first"]),
        }
        stages[d["stage"]] = rec
        order.append(d["stage"])
    return stages, order


def stats_of(a, first_n=FIRST_N):
    """Same reduction as muse_glimmer::trace::stats_of (f64)."""
    a64 = a.astype(np.float64)
    mean = float(a64.mean())
    var = float(a64.var())  # population variance
    return {
        "mean": mean,
        "var": var,
        "max_abs": float(np.abs(a64).max()),
        "first": [float(v) for v in a[:first_n]],
    }


def load_ref(d, stem, manifest):
    dims = manifest[stem]
    a = np.fromfile(Path(d) / f"{stem}.bin", dtype="<f4")
    total = 1
    for v in dims:
        total *= v
    assert a.size == total, f"{stem}: {a.size} != prod({dims})"
    return a, dims


def reduce_ref(d, manifest, n_layers):
    """Reduce the cb_dump_glimmer tensors to per-stage records in
    the same shape as the parsed trace lines."""
    out = {}

    def stream_record(stem):
        a, dims = load_ref(d, stem, manifest)
        width = dims[0]
        rows = a.size // width
        full = stats_of(a)
        last = stats_of(a[-width:])
        return {
            "rows": rows,
            "cols": width,
            "mean": full["mean"],
            "var": full["var"],
            "max_abs": full["max_abs"],
            "first": full["first"],
            "last_mean": last["mean"],
            "last_var": last["var"],
            "last_max_abs": last["max_abs"],
            "last_first": last["first"],
        }

    def last_row_record(stem):
        a, dims = load_ref(d, stem, manifest)
        width = dims[0]
        last = stats_of(a[-width:])
        return {
            "rows": 1,
            "cols": width,
            "mean": last["mean"],
            "var": last["var"],
            "max_abs": last["max_abs"],
            "first": last["first"],
            "last_mean": last["mean"],
            "last_var": last["var"],
            "last_max_abs": last["max_abs"],
            "last_first": last["first"],
        }

    if "embed" in manifest:
        out["embed"] = stream_record("embed")
    for il in range(n_layers):
        mid, outp = f"L{il:02d}_mid", f"L{il:02d}_out"
        if mid in manifest:
            out[f"layer.{il}.mid"] = stream_record(mid)
        if outp in manifest:
            out[f"layer.{il}.out"] = stream_record(outp)
    if "result_norm" in manifest:
        out["final_norm"] = last_row_record("result_norm")
    logits_info = None
    if "logits" in manifest:
        out["logits"] = last_row_record("logits")
        a, dims = load_ref(d, "logits", manifest)
        width = dims[0]
        last = a[-width:].astype(np.float64)
        top = int(np.argmax(last))
        p = float(np.exp(last - last.max())[top] / np.exp(last - last.max()).sum())
        logits_info = (top, float(last[top]), p)
    extra = {}
    if "embd_raw" in manifest:
        extra["embd_raw"] = stream_record("embd_raw")
    return out, extra, logits_info


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def judge(our, ref):
    """Plan reading rule. Returns (ok, metrics dict)."""
    std = math.sqrt(max(our["last_var"], ref["last_var"]))
    mean_shift = abs(our["last_mean"] - ref["last_mean"]) / max(std, 1e-12)
    var_rel = rel(our["last_var"], ref["last_var"])
    wstd = math.sqrt(max(our["var"], ref["var"]))
    w_mean_shift = abs(our["mean"] - ref["mean"]) / max(wstd, 1e-12)
    w_var_rel = rel(our["var"], ref["var"])
    first_d = max(
        (abs(x - y) for x, y in zip(our["last_first"], ref["last_first"])),
        default=0.0,
    )
    first_rel = first_d / max(std, 1e-12)
    ok = (
        mean_shift <= 1e-2
        and var_rel <= 1e-2
        and w_mean_shift <= 1e-2
        and w_var_rel <= 1e-2
        # The reading rule's "first values tracking": the first
        # values gate at the same band as the other criteria. This
        # is the only per-element check the comparator has; without
        # it a permutation passes as a match.
        and first_rel <= 1e-2
    )
    return ok, {
        "mean_shift_std": mean_shift,
        "var_rel": var_rel,
        "w_mean_shift_std": w_mean_shift,
        "w_var_rel": w_var_rel,
        "first_max_diff": first_d,
        "first_rel_std": first_rel,
    }


def main():
    our_path, ref_dir = sys.argv[1], sys.argv[2]
    manifest = json.loads((Path(ref_dir) / "manifest.json").read_text())
    ours, order = parse_ours(our_path)
    n_layers = sum(1 for s in ours if s.endswith(".out"))
    ref, extra, logits_info = reduce_ref(ref_dir, manifest, n_layers)

    print(f"{'stage':16s} {'verdict':9s} {'last mean shift (std)':>22s} "
          f"{'last var rel':>13s} {'whole var rel':>14s} {'first diff':>11s}")
    first_div = None
    no_ref = False
    prev_ok = True
    for stage in order:
        if stage not in ref:
            print(f"{stage:16s} {'NO REF':9s}")
            no_ref = True
            continue
        ok, mt = judge(ours[stage], ref[stage])
        verdict = "match" if ok else "DIVERGENT"
        mark = ""
        if not ok and first_div is None:
            first_div = stage
            mark = "  <-- FIRST DIVERGENT (input stage matched: %s)" % prev_ok
        print(f"{stage:16s} {verdict:9s} {mt['mean_shift_std']:22.3e} "
              f"{mt['var_rel']:13.3e} {mt['w_var_rel']:14.3e} "
              f"{mt['first_max_diff']:11.3e}{mark}")
        prev_ok = ok

    print(f"\nfirst divergent stage: {first_div}")
    if "embd_raw" in extra:
        e = extra["embd_raw"]
        print(f"reference embd_raw (pre-norm, info only): mean={e['mean']:.6e} "
              f"var={e['var']:.6e} max_abs={e['max_abs']:.6e}")
    if logits_info is not None:
        top, val, p = logits_info
        print(f"reference logits: top-1 id={top} logit={val:.4f} prob={p:.4f}")
    # The verdict is also the exit status: 1 on any divergence or on
    # a stage with no reference, 0 only on a full match.
    return 1 if (first_div is not None or no_ref) else 0


if __name__ == "__main__":
    sys.exit(main())
