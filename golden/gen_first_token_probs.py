#!/usr/bin/env python3
"""First-token top-5 probability golden (TODO T-9.3 reference).

For each prompt in the golden prompt set this records the top 5
candidate first tokens with their probabilities, as reported by a
throwaway CPU llama-server (/completion, n_predict=1, n_probs=5,
temperature 0). The probability-level reference complements the
token-level fixtures: an engine's logit path (final norm, softcap,
logit scale) can be checked against these numbers within tolerance
even where its greedy trajectory would match anyway.

Same discipline as gen_golden.py: CPU only (-ngl 0), localhost,
port 18923 by default, the server is terminated in a finally block,
and the production Glimmer server is never touched. The prompt ids
come from llama-tokenize --ids and the server's evaluated prompt
count is asserted against them, exactly as in gen_golden.py.

Output: one metadata line, then one record per prompt:
{"id": ..., "prompt": ..., "prompt_token_ids": [...],
 "top5": [{"id": int, "text": str, "prob": float}, ...]}

Usage:
    gen_first_token_probs.py --model /path/to/model.gguf \
        [--bin-dir /home/server/llama.cpp/build-vulkan2/bin] \
        [--prompts prompts.jsonl] [--out fixtures/<model>.top5.jsonl] \
        [--port 18923]

Standard library only.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

TEMPERATURE = 0
SEED = 42
CTX = 512
THREADS = 8
N_PROBS = 5
PROMPT_SET_VERSION = "1.0"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prompts(path):
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def tokenize(bin_dir, model, prompt):
    out = subprocess.run(
        [str(bin_dir / "llama-tokenize"), "-m", str(model), "-p", prompt,
         "--ids", "--no-escape"],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout.strip())


def llama_version(bin_dir):
    out = subprocess.run(
        [str(bin_dir / "llama-server"), "--version"],
        capture_output=True, text=True, check=True,
    )
    first = (out.stdout or out.stderr).splitlines()[0]
    return first.removeprefix("version: ").strip()


def wait_health(port, proc, log_path):
    url = f"http://127.0.0.1:{port}/health"
    for _ in range(240):
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
    raise RuntimeError("llama-server did not become healthy in 120 s")


def complete_first(port, prompt):
    body = json.dumps({
        "prompt": prompt,
        "n_predict": 1,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "n_probs": N_PROBS,
        "post_sampling_probs": True,
        "cache_prompt": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=body, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = Path(__file__).resolve().parent
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--bin-dir", type=Path,
                    default=Path("/home/server/llama.cpp/build-vulkan2/bin"))
    ap.add_argument("--prompts", type=Path, default=here / "prompts.jsonl")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--port", type=int, default=18923)
    args = ap.parse_args()

    model = args.model
    out = args.out or here / "fixtures" / (model.stem + ".top5.jsonl")
    prompts = load_prompts(args.prompts)

    meta = {
        "kind": "llama.cpp-first-token-top5",
        "version": 1,
        "model_file": model.name,
        "model_sha256": sha256_of(model),
        "model_bytes": model.stat().st_size,
        "llama_cpp_version": llama_version(args.bin_dir),
        "generation_binary": "llama-server (/completion, n_predict=1, n_probs=5)",
        "tokenize_binary": "llama-tokenize (--ids --no-escape)",
        "flags": {
            "n_gpu_layers": 0, "ctx_size": CTX, "threads": THREADS,
            "n_predict": 1, "n_probs": N_PROBS, "temperature": TEMPERATURE,
            "seed": SEED, "post_sampling_probs": True,
            "chat_template": "none (raw completion)",
        },
        "prompts_file": args.prompts.name,
        "prompt_set_version": PROMPT_SET_VERSION,
        "prompts_sha256": sha256_of(args.prompts),
    }

    records = []
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as log:
        log_path = log.name
    proc = subprocess.Popen(
        [str(args.bin_dir / "llama-server"), "-m", str(model),
         "-ngl", "0", "-c", str(CTX), "-t", str(THREADS),
         "--host", "127.0.0.1", "--port", str(args.port)],
        stdout=open(log_path, "ab"), stderr=subprocess.STDOUT,
    )
    try:
        wait_health(args.port, proc, log_path)
        for p in prompts:
            prompt_ids = tokenize(args.bin_dir, model, p["prompt"])
            resp = complete_first(args.port, p["prompt"])
            if resp["tokens_evaluated"] != len(prompt_ids):
                raise RuntimeError(
                    f"prompt token count mismatch for {p['id']}: server "
                    f"evaluated {resp['tokens_evaluated']}, llama-tokenize "
                    f"produced {len(prompt_ids)}"
                )
            steps = resp["completion_probabilities"]
            if len(steps) != 1:
                raise RuntimeError(
                    f"{p['id']}: expected exactly 1 probability step, "
                    f"got {len(steps)}"
                )
            # This llama.cpp build reports the per-step alternatives
            # under "top_probs", each {"id", "token", "bytes", "prob"}
            # (server-task.cpp, completion_token_output::to_json).
            if "top_probs" not in steps[0]:
                raise RuntimeError(
                    f"{p['id']}: step has no top_probs; keys are "
                    f"{sorted(steps[0].keys())}"
                )
            top5 = [{"id": pr["id"], "text": pr["token"], "prob": pr["prob"]}
                    for pr in steps[0]["top_probs"][:N_PROBS]]
            records.append({
                "id": p["id"],
                "category": p["category"],
                "prompt": p["prompt"],
                "prompt_token_ids": prompt_ids,
                "sampled_token_id": steps[0]["id"],
                "top5": top5,
            })
            best = top5[0]
            print(f"  {p['id']}: top1 id={best['id']} "
                  f"prob={best['prob']:.6f} text={best['text']!r}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(records)} records)")


if __name__ == "__main__":
    sys.exit(main())
