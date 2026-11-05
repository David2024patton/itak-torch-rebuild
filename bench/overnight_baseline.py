#!/usr/bin/env python3
"""Overnight baseline agreement run: our CPU engine vs llama.cpp.

For a fixed prompt set this records, per prompt, on the real
Glimmer GGUF at temperature 0:

  - llama.cpp side: CPU-only llama-server /completion with the
    prompt posted as a token-id array (llama-tokenize --ids), so
    both engines consume byte-identical input; n_predict 16,
    n_probs 5, ignore_eos true.
  - our side: crates/itak-loader/examples/glimmer_generate.rs on
    the cpu-compute reference path, 16 greedy tokens, per-step
    top-5 with probabilities parsed from its stdout.

Outputs (into --out-dir): one JSON per side per prompt, the raw
engine logs, and compare.json with token-prefix agreement, first
divergence, first-token top-5 tables and per-position top-1
probability deltas over the shared prefix.

Server discipline (see bench/overnight_baseline_run.sh): run
under the phase flock, CPU only, llama-server on port 18923,
killed at the end. Standard library only.
"""

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

N_TOKENS = 16
PORT = 18923
THREADS = 8
CTX = 512

# The fixed prompt set: the golden factual_capital plus four more
# from golden/prompts.jsonl (the T-2.9 golden fixture set).
PROMPT_IDS = [
    "factual_capital",
    "factual_science",
    "arithmetic_add",
    "continuation_classic",
    "code_python",
]


def load_prompts(repo: Path):
    by_id = {}
    for line in (repo / "golden" / "prompts.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            by_id[rec["id"]] = rec["prompt"]
    return [(pid, by_id[pid]) for pid in PROMPT_IDS]


def tokenize(bin_dir: Path, model: str, prompt: str):
    out = subprocess.run(
        [str(bin_dir / "llama-tokenize"), "-m", model, "-p", prompt,
         "--ids", "--no-escape"],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout.strip())


def wait_health(proc, log_path: Path):
    url = f"http://127.0.0.1:{PORT}/health"
    for _ in range(120):
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
    raise RuntimeError("llama-server did not become healthy in 60 s")


def step_prob(step):
    """Sampled-token probability from a /completion step. This
    llama.cpp build reports logprob / top_logprobs (no prob
    fields); older shapes report prob / top_probs. Accept both."""
    if "prob" in step:
        return step["prob"]
    return 2.718281828459045 ** step["logprob"]


def step_top(step):
    """Top candidates of a step as [{id, prob}], best first."""
    cands = step.get("top_probs") or step.get("top_logprobs") or []
    out = []
    for t in cands:
        p = t["prob"] if "prob" in t else 2.718281828459045 ** t["logprob"]
        out.append({"id": t["id"], "prob": p})
    return out


def llama_complete(prompt_ids):
    body = json.dumps({
        "prompt": prompt_ids,
        "n_predict": N_TOKENS,
        "temperature": 0,
        "seed": 42,
        "n_probs": 5,
        "ignore_eos": True,
        "cache_prompt": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/completion",
        data=body, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.loads(r.read().decode("utf-8"))


TOP_LINE = re.compile(
    r"^(?P<label>\w+)_top rank=(?P<rank>\d+) id=(?P<id>\d+) "
    r"tok=(?P<tok>.*) logit=(?P<logit>[-\d.]+) prob=(?P<prob>[-\d.e]+)$"
)


def parse_ours(log_text: str):
    prompt_ids = None
    generated = None
    tops = {}  # position (1-based) -> list of {rank,id,logit,prob}
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
                    "rank": int(m.group("rank")),
                    "id": int(m.group("id")),
                    "logit": float(m.group("logit")),
                    "prob": float(m.group("prob")),
                })
    for pos in tops:
        tops[pos].sort(key=lambda r: r["rank"])
    return prompt_ids, generated, tops


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", type=Path, required=True,
                    help="staging tree root (contains golden/, target/)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--bin-dir", type=Path,
                    default=Path("/home/server/llama.cpp/build-vulkan2/bin"))
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args.repo)
    engine_bin = args.repo / "target" / "release" / "examples" / "glimmer_generate"
    if not engine_bin.exists():
        print(f"engine example not built: {engine_bin}", file=sys.stderr)
        return 1

    ver = subprocess.run(
        [str(args.bin_dir / "llama-server"), "--version"],
        capture_output=True, text=True, check=True)
    llama_version = (ver.stdout or ver.stderr).splitlines()[0].strip()
    (args.out_dir / "llama-version.txt").write_text(llama_version + "\n")
    print(f"llama.cpp: {llama_version}", flush=True)

    # ---- llama.cpp side (fast; server up only for this phase) ----
    log_path = args.out_dir / "llama-server.log"
    proc = subprocess.Popen(
        [str(args.bin_dir / "llama-server"), "-m", args.model,
         "-ngl", "0", "-c", str(CTX), "-t", str(THREADS),
         "--host", "127.0.0.1", "--port", str(PORT)],
        stdout=open(log_path, "ab"), stderr=subprocess.STDOUT,
    )
    llama = {}
    try:
        wait_health(proc, log_path)
        for pid, prompt in prompts:
            ids = tokenize(args.bin_dir, args.model, prompt)
            resp = llama_complete(ids)
            llama[pid] = {"prompt_ids": ids, "response": resp}
            (args.out_dir / f"llama-{pid}.json").write_text(
                json.dumps(llama[pid], ensure_ascii=False, indent=1))
            steps = resp["completion_probabilities"]
            print(f"llama {pid}: prompt {len(ids)} ids, "
                  f"sampled {[s['id'] for s in steps]}", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("llama-server stopped", flush=True)

    # ---- our side (slow CPU reference; one process per prompt) ----
    ours = {}
    for pid, prompt in prompts:
        log_file = args.out_dir / f"ours-{pid}.log"
        t0 = time.time()
        with open(log_file, "w") as lf:
            rc = subprocess.run(
                [str(engine_bin), args.model, prompt, str(N_TOKENS)],
                stdout=lf, stderr=subprocess.STDOUT,
            ).returncode
        wall = time.time() - t0
        text = log_file.read_text(errors="replace")
        p_ids, generated, tops = parse_ours(text)
        ours[pid] = {
            "rc": rc, "wall_s": round(wall, 1),
            "prompt_ids": p_ids, "generated_ids": generated,
            "tops": {str(k): v for k, v in sorted(tops.items())},
        }
        (args.out_dir / f"ours-{pid}.json").write_text(
            json.dumps(ours[pid], ensure_ascii=False, indent=1))
        print(f"ours {pid}: rc={rc} wall={wall:.0f}s generated={generated}",
              flush=True)

    # ---- comparison ----
    compare = {"llama_version": llama_version, "prompts": {}}
    for pid, _prompt in prompts:
        o = ours.get(pid, {})
        l = llama.get(pid, {})
        steps = l.get("response", {}).get("completion_probabilities", [])
        llama_ids = [s["id"] for s in steps]
        our_ids = o.get("generated_ids") or []
        prefix = 0
        for a, b in zip(our_ids, llama_ids):
            if a != b:
                break
            prefix += 1
        first_div = None
        if prefix < min(len(our_ids), len(llama_ids)):
            first_div = prefix + 1  # 1-based generated position
        pos_rows = []
        for pos in range(1, prefix + 1):
            step = steps[pos - 1]
            lprob = step_prob(step)
            our_top = (o.get("tops", {}).get(str(pos)) or [{}])[0]
            pos_rows.append({
                "pos": pos,
                "id": step["id"],
                "llama_top1_prob": lprob,
                "ours_top1_prob": our_top.get("prob"),
                "delta": (our_top.get("prob") - lprob)
                if our_top.get("prob") is not None else None,
            })
        first_llama_top5 = step_top(steps[0]) if steps else []
        compare["prompts"][pid] = {
            "prompt_ids_match": o.get("prompt_ids") == l.get("prompt_ids"),
            "ours_prompt_ids": o.get("prompt_ids"),
            "llama_prompt_ids": l.get("prompt_ids"),
            "ours_generated_ids": our_ids,
            "llama_generated_ids": llama_ids,
            "prefix_agreement": prefix,
            "first_divergence_pos": first_div,
            "first_token_top5_llama": first_llama_top5,
            "first_token_top5_ours": (o.get("tops", {}).get("1") or []),
            "shared_prefix_top1_probs": pos_rows,
            "ours_wall_s": o.get("wall_s"),
            "ours_rc": o.get("rc"),
        }
        print(f"compare {pid}: prefix={prefix} first_div={first_div}",
              flush=True)
    (args.out_dir / "compare.json").write_text(
        json.dumps(compare, ensure_ascii=False, indent=1))
    print("wrote compare.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
