#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""TTFT comparison harness for CXL-P2P vs NIXL-P2P.

Usage:
    bench_ttft.py \
        --node-a-url http://NODE_A:8100 \
        --node-b-url http://NODE_B:8200 \
        --prompt-tokens 1000 2000 4000 8000 \
        --repeat 5 \
        --label cxl-static \
        --out results-cxl.csv

Run once per arm (CXL, NIXL), pointing at the respective vLLM ports.
Pass `--label` to tag the rows so you can concat the CSVs later.

Methodology assumed:
  - Both nodes serve the same model, same dtype, same chunk size.
  - prefix-caching disabled in vLLM so the only cache is LMCache.
  - Operator manually restarts the stack between condition groups
    and drops page cache. This script does NOT manage lifecycle.
  - For each (prompt_len, repeat), generates a fresh random prompt
    so the stack hasn't seen it before. Then:
      1. send to A — warmup, discard timing
      2. send to A — measure TTFT_A_local
      3. send to B — measure TTFT_B_cold (this triggers cross-node fetch)
      4. send to B — measure TTFT_B_warm (post-fetch hit)
    All four hit the same cache key, so steps 2/3/4 should hit cache.

Reports CSV with columns:
    label, prompt_tokens, repeat, ttft_a_local, ttft_b_cold, ttft_b_warm

Median + p99 are computed in analysis. Don't report means.
"""

# Standard
from __future__ import annotations
import argparse
import csv
import json
import random
import statistics
import string
import sys
import time
import urllib.request


def make_prompt(num_tokens: int, seed: int) -> str:
    """Generate a deterministic random-ish prompt of approximately
    `num_tokens` tokens (rough heuristic: 4 chars/token).
    """
    rng = random.Random(seed)
    words = [
        "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 8)))
        for _ in range(num_tokens)
    ]
    return " ".join(words)


def send_completion(url: str, model: str, prompt: str, timeout_s: float) -> float:
    """POST a single-token completion request. Returns elapsed wall-clock
    seconds (TTFT proxy: max_tokens=1, so end-to-end ≈ TTFT)."""
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        # Drain the response so timing covers the full first-token path.
        resp.read()
    return time.perf_counter() - t0


def print_median_summary(rows: list[dict]) -> None:
    """Print a per-prompt-length median table for each TTFT metric.

    Medians (not means) are reported because TTFT distributions are
    right-skewed: a few slow requests pull the mean up and misrepresent
    typical latency. Grouped by ``prompt_tokens`` across all repeats.

    Args:
        rows: Per-trial result dicts produced by the benchmark loop.
    """
    metrics = ("ttft_a_local", "ttft_b_cold", "ttft_b_warm")
    by_len: dict[int, list[dict]] = {}
    for row in rows:
        by_len.setdefault(row["prompt_tokens"], []).append(row)

    print("\n=== Median TTFT (seconds) by prompt length ===", file=sys.stderr)
    header = f"{'prompt_tokens':>14} {'n':>4} " + " ".join(
        f"{m:>14}" for m in metrics
    )
    print(header, file=sys.stderr)
    for n_tok in sorted(by_len):
        group = by_len[n_tok]
        medians = " ".join(
            f"{statistics.median(r[m] for r in group):>14.4f}" for m in metrics
        )
        print(f"{n_tok:>14} {len(group):>4} {medians}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--node-a-url", required=True,
                   help="vLLM URL on Node A, e.g. http://10.0.0.1:8100")
    p.add_argument("--node-b-url", required=True,
                   help="vLLM URL on Node B, e.g. http://10.0.0.2:8200")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--prompt-tokens", type=int, nargs="+",
                   default=[1000, 2000, 4000, 8000],
                   help="Approx prompt lengths to sweep.")
    p.add_argument("--repeat", type=int, default=5,
                   help="Trials per prompt length.")
    p.add_argument("--label", required=True,
                   help="Tag for the rows (e.g. cxl-static, nixl-controller).")
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--timeout-s", type=float, default=120.0)
    p.add_argument("--out", required=True, help="CSV output path.")
    args = p.parse_args()

    rows = []
    for n_tok in args.prompt_tokens:
        for trial in range(args.repeat):
            seed = args.seed_base + n_tok * 1000 + trial
            prompt = make_prompt(n_tok, seed)

            # Warmup on A — discard.
            send_completion(args.node_a_url, args.model, prompt, args.timeout_s)
            # Measured request on A — local cache (warm).
            t_a_local = send_completion(
                args.node_a_url, args.model, prompt, args.timeout_s)
            # First request on B — triggers cross-node fetch.
            t_b_cold = send_completion(
                args.node_b_url, args.model, prompt, args.timeout_s)
            # Second request on B — post-fetch hit.
            t_b_warm = send_completion(
                args.node_b_url, args.model, prompt, args.timeout_s)

            row = {
                "label": args.label,
                "prompt_tokens": n_tok,
                "repeat": trial,
                "ttft_a_local": t_a_local,
                "ttft_b_cold": t_b_cold,
                "ttft_b_warm": t_b_warm,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}", file=sys.stderr)

    print_median_summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
