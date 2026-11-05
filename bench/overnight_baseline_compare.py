#!/usr/bin/env python3
"""Recompute compare.json from saved per-prompt JSONs.

Recovery companion to overnight_baseline.py: the 2026-11-05 run's
driver crashed in its compare phase because this llama.cpp build
reports logprob/top_logprobs while the compare code first assumed
prob/top_probs. All per-prompt JSONs were already saved, so the
comparison is recomputed here from them with the shape-tolerant
helpers. Same outputs as the driver's compare phase.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overnight_baseline import PROMPT_IDS, step_prob, step_top  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    llama_version = (args.out_dir / "llama-version.txt").read_text().strip()
    compare = {"llama_version": llama_version, "prompts": {}}
    for pid in PROMPT_IDS:
        o = json.loads((args.out_dir / f"ours-{pid}.json").read_text())
        l = json.loads((args.out_dir / f"llama-{pid}.json").read_text())
        steps = l["response"]["completion_probabilities"]
        llama_ids = [s["id"] for s in steps]
        our_ids = o.get("generated_ids") or []
        prefix = 0
        for a, b in zip(our_ids, llama_ids):
            if a != b:
                break
            prefix += 1
        first_div = None
        if prefix < min(len(our_ids), len(llama_ids)):
            first_div = prefix + 1
        pos_rows = []
        for pos in range(1, prefix + 1):
            lprob = step_prob(steps[pos - 1])
            our_top = (o.get("tops", {}).get(str(pos)) or [{}])[0]
            pos_rows.append({
                "pos": pos,
                "id": steps[pos - 1]["id"],
                "llama_top1_prob": lprob,
                "ours_top1_prob": our_top.get("prob"),
                "delta": (our_top.get("prob") - lprob)
                if our_top.get("prob") is not None else None,
            })
        compare["prompts"][pid] = {
            "prompt_ids_match": o.get("prompt_ids") == l.get("prompt_ids"),
            "ours_prompt_ids": o.get("prompt_ids"),
            "llama_prompt_ids": l.get("prompt_ids"),
            "ours_generated_ids": our_ids,
            "llama_generated_ids": llama_ids,
            "prefix_agreement": prefix,
            "first_divergence_pos": first_div,
            "first_token_top5_llama": step_top(steps[0]) if steps else [],
            "first_token_top5_ours": (o.get("tops", {}).get("1") or []),
            "shared_prefix_top1_probs": pos_rows,
            "ours_wall_s": o.get("wall_s"),
            "ours_rc": o.get("rc"),
        }
        print(f"compare {pid}: prefix={prefix} first_div={first_div}")
    (args.out_dir / "compare.json").write_text(
        json.dumps(compare, ensure_ascii=False, indent=1))
    print("wrote compare.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
