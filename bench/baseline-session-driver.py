#!/usr/bin/env python3
"""Baseline session driver (2026-11-06 consolidated card session).

Cross-backend agreement collection for Leg 1(c): the overnight
baseline pattern (5 golden prompts x 16 tokens, temperature 0,
token-id input) run against multiple stacks and compared
pairwise. The parsing and alignment mirror
bench/overnight_baseline.py exactly: llama.cpp generated ids are
[s["id"] for s in completion_probabilities]; our per-position
tops are 1-based with prefill = position 1, aligned against
llama step pos-1; a step's top-1 probability is exp(logprob) of
the sampled (greedy) token.

Subcommands:
  llama  --bin-dir D --ngl N --port P --tag T --out-dir O
         [--repo R] [--model M]
         Start llama-server from D, tokenize with D's
         llama-tokenize, complete all five prompts, write
         O/stack-llama-<tag>.json, stop the server.
  ours   --tag T --out-dir O --log pid=path [--log pid=path...]
         Parse glimmer_generate logs (CPU or --gpu path; same
         output contract) into O/stack-ours-<tag>.json.
  compare --out-dir O
         Load every stack-*.json in O, compare all pairs per
         prompt (prompt ids, generated ids position by
         position, top-1 probability deltas over the shared
         prefix), write O/compare-matrix.json and print the
         matrix.
"""

import argparse
import json
import math
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROMPT_IDS = [
    "factual_capital",
    "factual_science",
    "arithmetic_add",
    "continuation_classic",
    "code_python",
]
N_TOKENS = 16
MODEL_DEFAULT = "/mnt/models/Muse-Glimmer-30B/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"
CTX = 512
THREADS = 8


def load_prompts(repo: Path):
    by_id = {}
    for line in (repo / "golden" / "prompts.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            by_id[rec["id"]] = rec["prompt"]
    return [(pid, by_id[pid]) for pid in PROMPT_IDS]


def step_prob(step):
    if "prob" in step:
        return step["prob"]
    return math.exp(step["logprob"])


def step_top(step):
    cands = step.get("top_probs") or step.get("top_logprobs") or []
    out = []
    for t in cands:
        p = t["prob"] if "prob" in t else math.exp(t["logprob"])
        out.append({"id": t["id"], "prob": p})
    return out


def wait_health(proc, port, log_path: Path):
    url = f"http://127.0.0.1:{port}/health"
    for _ in range(240):
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace")[-2000:]
            raise RuntimeError(f"llama-server exited early; log tail:\n{tail}")
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("llama-server did not become healthy in 120 s")


def cmd_llama(a):
    bin_dir = Path(a.bin_dir)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(Path(a.repo))
    version = subprocess.run(
        [str(bin_dir / "llama-server"), "--version"],
        capture_output=True, text=True).stderr.strip().splitlines()
    log_path = out_dir / f"llama-server-{a.tag}.log"
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(
            [str(bin_dir / "llama-server"), "-m", a.model,
             "-ngl", str(a.ngl), "-c", str(CTX), "-t", str(THREADS),
             "--host", "127.0.0.1", "--port", str(a.port)],
            stdout=lf, stderr=subprocess.STDOUT)
    result = {"stack": f"llama-{a.tag}", "bin_dir": str(bin_dir),
              "ngl": a.ngl, "version": version[-1] if version else "?",
              "prompts": {}}
    try:
        wait_health(proc, a.port, log_path)
        for pid, prompt in prompts:
            ids = json.loads(subprocess.run(
                [str(bin_dir / "llama-tokenize"), "-m", a.model, "-p", prompt,
                 "--ids", "--no-escape"],
                capture_output=True, text=True, check=True).stdout.strip())
            body = json.dumps({
                "prompt": ids, "n_predict": N_TOKENS, "temperature": 0,
                "seed": 42, "n_probs": 5, "ignore_eos": True,
                "cache_prompt": False}).encode("utf-8")
            req = urllib.request.Request(
                f"http://127.0.0.1:{a.port}/completion", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1200) as r:
                resp = json.loads(r.read().decode("utf-8"))
            steps = resp["completion_probabilities"]
            result["prompts"][pid] = {
                "prompt_ids": ids,
                "generated_ids": [s["id"] for s in steps],
                "top1_probs": [step_prob(s) for s in steps],
                "first_top5": step_top(steps[0]) if steps else [],
                "raw": resp,
            }
            print(f"llama-{a.tag} {pid}: {result['prompts'][pid]['generated_ids']}",
                  flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    (out_dir / f"stack-llama-{a.tag}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1))
    print(f"wrote stack-llama-{a.tag}.json; server stopped", flush=True)
    return 0


TOP_LINE = re.compile(
    r"^(?P<label>\w+)_top rank=(?P<rank>\d+) id=(?P<id>\d+) "
    r"tok=(?P<tok>.*) logit=(?P<logit>[-\d.]+) prob=(?P<prob>[-\d.e]+)$"
)


def parse_ours(log_text: str):
    prompt_ids = None
    generated = None
    tops = {}
    for line in log_text.splitlines():
        if line.startswith("prompt_ids="):
            prompt_ids = json.loads(line.split("=", 1)[1])
        elif line.startswith("generated_ids="):
            generated = json.loads(line.split("=", 1)[1])
        else:
            m = TOP_LINE.match(line)
            if m:
                label = m.group("label")
                pos = 1 if label == "prefill" else int(label[4:])
                tops.setdefault(pos, []).append({
                    "rank": int(m.group("rank")), "id": int(m.group("id")),
                    "logit": float(m.group("logit")),
                    "prob": float(m.group("prob"))})
    for pos in tops:
        tops[pos].sort(key=lambda r: r["rank"])
    return prompt_ids, generated, tops


def cmd_ours(a):
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"stack": f"ours-{a.tag}", "prompts": {}}
    for item in a.log:
        pid, path = item.split("=", 1)
        p_ids, generated, tops = parse_ours(Path(path).read_text(errors="replace"))
        result["prompts"][pid] = {
            "prompt_ids": p_ids,
            "generated_ids": generated,
            "top1_probs": [(tops[p][0]["prob"] if tops.get(p) else None)
                           for p in range(1, N_TOKENS + 1)],
            "first_top5": [{"id": t["id"], "prob": t["prob"]}
                           for t in tops.get(1, [])],
            "tops": {str(k): v for k, v in sorted(tops.items())},
        }
        print(f"ours-{a.tag} {pid}: {generated}", flush=True)
    (out_dir / f"stack-ours-{a.tag}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1))
    print(f"wrote stack-ours-{a.tag}.json", flush=True)
    return 0


def cmd_compare(a):
    out_dir = Path(a.out_dir)
    stacks = {}
    for f in sorted(out_dir.glob("stack-*.json")):
        d = json.loads(f.read_text())
        stacks[d["stack"]] = d
    names = sorted(stacks)
    matrix = {"stacks": names, "pairs": {}}
    print(f"stacks: {names}")
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            pair = {"prompts": {}, "any_token_divergence": False,
                    "max_abs_top1_delta": 0.0}
            for pid in PROMPT_IDS:
                px = stacks[x]["prompts"].get(pid, {})
                py = stacks[y]["prompts"].get(pid, {})
                gx = px.get("generated_ids") or []
                gy = py.get("generated_ids") or []
                prefix = 0
                for u, v in zip(gx, gy):
                    if u != v:
                        break
                    prefix += 1
                first_div = None
                if prefix < min(len(gx), len(gy)):
                    first_div = prefix + 1
                deltas = []
                for pos in range(prefix):
                    ax = (px.get("top1_probs") or [None] * 16)[pos]
                    ay = (py.get("top1_probs") or [None] * 16)[pos]
                    if ax is not None and ay is not None:
                        deltas.append(ax - ay)
                md = max((abs(d) for d in deltas), default=0.0)
                pair["max_abs_top1_delta"] = max(pair["max_abs_top1_delta"], md)
                if first_div is not None:
                    pair["any_token_divergence"] = True
                pair["prompts"][pid] = {
                    "prompt_ids_match": px.get("prompt_ids") == py.get("prompt_ids"),
                    "generated_x": gx, "generated_y": gy,
                    "prefix_agreement": prefix,
                    "first_divergence_pos": first_div,
                    "top1_deltas_shared_prefix": deltas,
                    "max_abs_top1_delta": md,
                }
                print(f"{x} vs {y} {pid}: prefix={prefix} "
                      f"first_div={first_div} max|delta|={md:.4f}")
            matrix["pairs"][f"{x} vs {y}"] = pair
    (out_dir / "compare-matrix.json").write_text(
        json.dumps(matrix, ensure_ascii=False, indent=1))
    print("wrote compare-matrix.json", flush=True)
    return 0


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("llama")
    q.add_argument("--bin-dir", required=True)
    q.add_argument("--ngl", type=int, required=True)
    q.add_argument("--port", type=int, required=True)
    q.add_argument("--tag", required=True)
    q.add_argument("--out-dir", required=True)
    q.add_argument("--repo", default=".")
    q.add_argument("--model", default=MODEL_DEFAULT)
    q.set_defaults(fn=cmd_llama)
    q = sub.add_parser("ours")
    q.add_argument("--tag", required=True)
    q.add_argument("--out-dir", required=True)
    q.add_argument("--log", action="append", required=True)
    q.set_defaults(fn=cmd_ours)
    q = sub.add_parser("compare")
    q.add_argument("--out-dir", required=True)
    q.set_defaults(fn=cmd_compare)
    a = p.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
