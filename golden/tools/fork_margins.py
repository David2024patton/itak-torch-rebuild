#!/usr/bin/env python3
"""Fork-margin probe for a golden candidate (one-off analysis tool).

For every prompt where the candidate's completion diverges from the
golden fixture, find the first divergent index k, then ask llama.cpp
itself (throwaway CPU llama-server, same flags as gen_golden.py) for
the top-5 probabilities at that exact position, feeding the golden
prefix (prompt ids + golden completion ids before k) as token ids.

The reported margin is llama.cpp's own logit gap at the fork:
ln(p_winner) - ln(p_best_other). Greedy forks on margins smaller
than the cross-engine logit noise band are coin flips, not math
errors. The band referenced is the one RESULTS-layer-diff-qwen.md
measured on Qwen2.5-1.5B (up to ~0.78 logits); it was NOT measured
on the model probed here, and this script says so in its output.

Usage:
    fork_margins.py --golden <fixture.jsonl> --candidate <cand.jsonl>
                    --model <file.gguf>
                    [--bin-dir /home/server/llama.cpp/build-vulkan2/bin]
                    [--port 18931]

Standard library only.
"""

import argparse
import json
import math
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

QWEN_BAND = 0.78  # logits, cross-engine noise max measured on Qwen2.5


def load_records(path):
    records, order = {}, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "meta" in obj and "id" not in obj:
                continue
            records[obj["id"]] = obj
            order.append(obj["id"])
    return records, order


def first_divergence(g_ids, c_ids):
    n = min(len(g_ids), len(c_ids))
    for i in range(n):
        if g_ids[i] != c_ids[i]:
            return i
    if len(g_ids) != len(c_ids):
        return n
    return None


def wait_health(port, proc, log_path):
    url = f"http://127.0.0.1:{port}/health"
    for _ in range(120):
        if proc.poll() is not None:
            tail = Path(log_path).read_text(errors="replace")[-2000:]
            raise RuntimeError(f"llama-server exited early; log tail:\n{tail}")
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("llama-server did not become healthy in 60 s")


def top_probs(port, prefix_ids, n_probs=5):
    body = json.dumps({
        "prompt": prefix_ids,
        "n_predict": 1,
        "temperature": 0,
        "seed": 42,
        "n_probs": n_probs,
        "cache_prompt": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=body, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read().decode("utf-8"))
    if resp["tokens_evaluated"] != len(prefix_ids):
        raise RuntimeError(
            f"prefix length mismatch: server evaluated "
            f"{resp['tokens_evaluated']}, probe fed {len(prefix_ids)}")
    steps = resp["completion_probabilities"]
    if len(steps) != 1:
        raise RuntimeError(f"expected exactly 1 step, got {len(steps)}")
    return steps[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--golden", required=True, type=Path)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--bin-dir", type=Path,
                    default=Path("/home/server/llama.cpp/build-vulkan2/bin"))
    ap.add_argument("--port", type=int, default=18931)
    args = ap.parse_args()

    golden, order = load_records(args.golden)
    cand, _ = load_records(args.candidate)

    forks = []
    for pid in order:
        g, c = golden[pid], cand.get(pid)
        if c is None:
            print(f"{pid}: MISSING in candidate, skipped")
            continue
        if g["prompt_token_ids"] != c.get("prompt_token_ids"):
            print(f"{pid}: prompt ids differ, fork probe not meaningful, skipped")
            continue
        k = first_divergence(g["completion_token_ids"],
                             c["completion_token_ids"])
        if k is None:
            continue
        g_id = g["completion_token_ids"][k] if k < len(g["completion_token_ids"]) else None
        c_id = c["completion_token_ids"][k] if k < len(c["completion_token_ids"]) else None
        prefix = g["prompt_token_ids"] + g["completion_token_ids"][:k]
        forks.append((pid, k, g_id, c_id, prefix))

    if not forks:
        print("no divergent prompts to probe")
        return 0

    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as log:
        log_path = log.name
    proc = subprocess.Popen(
        [str(args.bin_dir / "llama-server"), "-m", str(args.model),
         "-ngl", "0", "-c", "512", "-t", "8",
         "--host", "127.0.0.1", "--port", str(args.port)],
        stdout=open(log_path, "ab"), stderr=subprocess.STDOUT,
    )
    try:
        wait_health(args.port, proc, log_path)
        print(f"probing {len(forks)} fork(s); band reference: {QWEN_BAND} "
              f"logits (measured on Qwen2.5-1.5B, not this model)")
        for pid, k, g_id, c_id, prefix in forks:
            step = top_probs(args.port, prefix)
            # This llama.cpp build returns per-step "top_logprobs":
            # [{id, token, bytes, logprob}], already in logit units.
            tops = step.get("top_logprobs") or []
            by_id = {p["id"]: p["logprob"] for p in tops}
            win = tops[0] if tops else None
            others = [p for p in tops if p["id"] != g_id]
            lp_win = by_id.get(g_id)
            lp_other = others[0]["logprob"] if others else None
            line = f"{pid}: fork at {k}, golden {g_id} vs candidate {c_id}"
            if win is not None and win["id"] != g_id:
                line += (f"; NOTE llama.cpp top-1 here is {win['id']} "
                         f"(logprob={win['logprob']:.3f}), "
                         f"not the fixture winner")
            if lp_win is not None and lp_other is not None:
                margin = lp_win - lp_other
                verdict = ("within" if margin < QWEN_BAND else "EXCEEDS")
                line += (f"; llama.cpp winner margin = {margin:.3f} logits "
                         f"-> {verdict} the {QWEN_BAND} band")
            else:
                line += "; winner not in returned top-5, margin not measurable"
            if c_id in by_id and lp_win is not None:
                gap = lp_win - by_id[c_id]
                line += f"; winner-vs-candidate gap = {gap:.3f} logits"
            elif c_id is not None:
                line += "; candidate token outside llama.cpp top-5"
            print(line)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
