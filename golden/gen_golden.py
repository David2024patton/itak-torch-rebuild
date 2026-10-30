#!/usr/bin/env python3
"""Generate llama.cpp golden-output fixtures for the Torch rebuild.

For each prompt in prompts.jsonl this records:
  - prompt_token_ids      from llama-tokenize --ids (the canonical
                          tokenization, BOS included, exactly what a
                          llama.cpp generation run evaluates)
  - completion_token_ids  the token ids llama.cpp actually sampled
  - completion_text       the text of that same generation

Why llama-server and not llama-completion for the generation step:
llama-completion/llama-cli cannot report the sampled token ids (no
flag exists; --verbose dumps graph internals, checked against build
a7af149). The ids are the whole point of the fixture, and they cannot
be recovered by retokenizing the completion text either: tokenization
is context-dependent (llama-tokenize on a completion text standalone
split its first token three ways where the in-context tokenization
of the same text has a single token), so text alone does not pin
down the trajectory an engine must reproduce.
llama-server's /completion with n_probs=1 returns, per step, the
sampled token's id, text piece and raw bytes; the bytes of the kept
steps must concatenate to the response content exactly, and this
script asserts that, so ids and text in a fixture provably come from
one and the same generation.

Decoding is greedy: temperature 0, seed 42 (recorded; unused by the
greedy sampler), 64 predicted tokens, raw completion (no chat
template), CPU only (-ngl 0) so the production GPU server is never
touched. Fixtures carry no timestamps: regenerating with the same
llama.cpp build and flags reproduces the file byte for byte.

Usage:
    gen_golden.py --model /path/to/model.gguf \
                  [--bin-dir /home/server/llama.cpp/build-vulkan2/bin] \
                  [--prompts prompts.jsonl] [--out fixtures/<model>.golden.jsonl] \
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

N_PREDICT = 64
TEMPERATURE = 0
SEED = 42
CTX = 512
THREADS = 8
PROMPT_SET_VERSION = "1.1"


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


def complete(port, prompt):
    body = json.dumps({
        "prompt": prompt,
        "n_predict": N_PREDICT,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "n_probs": 1,
        "cache_prompt": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=body, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode("utf-8"))


def extract_record(resp, prompt_ids):
    """Pull (completion_ids, completion_text, ended_with_eos) out of a
    /completion response, asserting the ids and the text are consistent."""
    if resp["tokens_evaluated"] != len(prompt_ids):
        raise RuntimeError(
            f"prompt token count mismatch: server evaluated "
            f"{resp['tokens_evaluated']}, llama-tokenize produced {len(prompt_ids)}"
        )
    steps = resp["completion_probabilities"]
    if len(steps) != resp["tokens_predicted"]:
        raise RuntimeError("probability steps do not match tokens_predicted")
    content = resp["content"]
    # This build reports the stop reason in stop_type ("eos" / "limit");
    # stopped_eos is null. When generation stops on EOS, the EOS token
    # is part of the sampled trajectory and stays in the record (its
    # text piece is empty, so the byte check below still holds).
    ended_with_eos = resp.get("stop_type") == "eos" or bool(resp.get("stopped_eos"))

    joined = b"".join(bytes(s["bytes"]) for s in steps)
    if joined != content.encode("utf-8"):
        # A sampled EOS is reported as a step but contributes no text.
        if ended_with_eos and steps and \
                b"".join(bytes(s["bytes"]) for s in steps[:-1]) == content.encode("utf-8"):
            steps = steps[:-1]
        else:
            raise RuntimeError("step bytes do not concatenate to the completion text")
    return [s["id"] for s in steps], content, ended_with_eos


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
    out = args.out or here / "fixtures" / (model.stem + ".golden.jsonl")
    prompts = load_prompts(args.prompts)

    meta = {
        "kind": "llama.cpp-golden-fixture",
        "version": 1,
        "model_file": model.name,
        "model_sha256": sha256_of(model),
        "model_bytes": model.stat().st_size,
        "llama_cpp_version": llama_version(args.bin_dir),
        "generation_binary": "llama-server (/completion, n_probs=1)",
        "tokenize_binary": "llama-tokenize (--ids --no-escape)",
        "flags": {
            "n_gpu_layers": 0, "ctx_size": CTX, "threads": THREADS,
            "n_predict": N_PREDICT, "temperature": TEMPERATURE, "seed": SEED,
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
            resp = complete(args.port, p["prompt"])
            comp_ids, comp_text, eos = extract_record(resp, prompt_ids)
            records.append({
                "id": p["id"],
                "category": p["category"],
                "prompt": p["prompt"],
                "prompt_token_ids": prompt_ids,
                "completion_token_ids": comp_ids,
                "completion_text": comp_text,
                "ended_with_eos": eos,
            })
            print(f"  {p['id']}: prompt {len(prompt_ids)} tokens, "
                  f"completion {len(comp_ids)} tokens, eos={eos}")
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
