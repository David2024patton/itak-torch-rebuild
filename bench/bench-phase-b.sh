#!/bin/bash
# Phase B: llama-server accuracy + reasoning_strength + 2-user concurrency + KV log capture
# Usage: bench-phase-b.sh <backend: vulkan|rocm> <model gguf path> <tag>
# Server on port 18090, -c 131072 -np 4 (shared profile from research). Logs to /tmp/bench-results/
set -u
BACKEND="${1:?backend}"; MODEL="${2:?model}"; TAG="${3:?tag}"
OUT=/tmp/bench-results; mkdir -p "$OUT"
BIN=/home/server/llama.cpp/build-${BACKEND}/bin/llama-server
[ "$BACKEND" = "rocm" ] && BIN=/home/server/llama.cpp/build-rocm/bin/llama-server
SLOG="$OUT/server_${BACKEND}_${TAG}.log"
"$BIN" -m "$MODEL" -ngl 99 -c 131072 -np 4 --jinja --host 127.0.0.1 --port 18090 > "$SLOG" 2>&1 &
SRV=$!
for _ in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:18090/health && break; sleep 5; done
grep -iE "kv|buffer" "$SLOG" | head -20 > "$OUT/kvlog_${BACKEND}_${TAG}.txt" || true
python3 - "$OUT" "$TAG" "$BACKEND" <<'PYEOF'
import json, sys, time, urllib.request, concurrent.futures
out, tag, backend = sys.argv[1], sys.argv[2], sys.argv[3]
URL = "http://127.0.0.1:18090/v1/chat/completions"
PROMPTS = [
 ("math", "A store sells pencils in packs of 6 for $4.50 per pack. Maria needs 84 pencils and has a coupon for 15% off her total. How much does she pay? Show your work step by step, then give the final amount."),
 ("go_code", "Write a Go function `MedianOfTwoSortedArrays(a, b []int) float64` that returns the median of two sorted slices in O(log(min(n,m))) time. Code only, no explanation."),
 ("logic", "Three boxes are labeled 'Apples', 'Oranges', and 'Apples & Oranges', but every label is wrong. You may draw one fruit from one box. Which box do you draw from, and how do you then correctly label all three? Answer concisely."),
 ("json", "Extract the data and reply with ONLY a JSON object, no other text: 'Jane Cooper, age 34, ordered 3 units of SKU-2214 on March 3rd, 2026, total $129.99, shipping to Austin, TX.' Keys: name, age, units, sku, date (ISO format), total, city, state."),
 ("factual", "In Ursula K. Le Guin's novel The Left Hand of Darkness, what is the name of the planet where most of the story takes place, what are its people called, and what is unusual about their biology? Two or three sentences."),
 ("summary", "Summarize in exactly 3 bullet points: Continuous batching lets an inference server swap finished requests out and new ones in at every token step instead of waiting for a whole batch. This raises GPU utilization because short requests no longer idle behind long ones. The cost is scheduler complexity and the need for paged KV cache management so sequences can start and stop without fragmenting memory. Operators also gain per-request controls like max tokens and stop sequences, which batch-at-once systems handle poorly."),
]
def ask(prompt, extra=None):
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 512}
    if extra: body.update(extra)
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    dt = time.time() - t0
    return d["choices"][0]["message"]["content"], d.get("usage", {}), dt
res = {"backend": backend, "tag": tag, "accuracy": {}, "reasoning": {}, "concurrency": {}}
for name, p in PROMPTS:
    c, u, dt = ask(p)
    res["accuracy"][name] = {"content": c, "usage": u, "seconds": round(dt, 2)}
# reasoning_strength low vs unset(default) on the math prompt
c, u, dt = ask(PROMPTS[0][1], {"chat_template_kwargs": {"reasoning_strength": "low"}})
res["reasoning"]["low"] = {"usage": u, "seconds": round(dt, 2), "content": c}
res["reasoning"]["default_usage"] = res["accuracy"]["math"]["usage"]
# 2-user concurrency: long generation from both at once
LONG = "Write a detailed explanation of how a sourdough starter works, covering the microbiology, feeding schedule, troubleshooting, and how to use it in a basic loaf recipe."
def worker(i):
    return ask(LONG)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    t0 = time.time(); futs = [ex.submit(worker, i) for i in range(2)]
    outs = [f.result() for f in futs]; wall = time.time() - t0
per = []
for c, u, dt in outs:
    ct = u.get("completion_tokens", 0)
    per.append({"completion_tokens": ct, "seconds": round(dt, 2), "tok_per_s": round(ct / dt, 2) if dt else 0})
res["concurrency"] = {"wall_seconds": round(wall, 2), "per_user": per,
                      "aggregate_tok_per_s": round(sum(u.get("completion_tokens", 0) for _, u, _ in outs) / wall, 2)}
with open(f"{out}/phaseb_{backend}_{tag}.json", "w") as f:
    json.dump(res, f, indent=1)
print("PHASE_B_DONE", backend, tag)
PYEOF
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
echo "PHASE_B_${BACKEND}_${TAG}_DONE"
