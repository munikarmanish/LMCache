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
        --out results/results-cxl.csv

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


def _basic_prompt(seed: int, num_tokens: int) -> str:
    """Build a prompt without a tokenizer: ``seed`` then a repeated word.

    Mirrors ``scripts/p2p/long_doc_qa.py``: the prompt starts with the seed
    number (so prompts with different seeds don't share a prefix) followed by
    ``"hi"`` repeated ``num_tokens`` times. ``"hi"`` is a single token for
    common tokenizers, so the length is approximate (off by the few tokens
    the leading seed contributes).

    Args:
        seed: Reproducibility seed; also the unique prompt prefix.
        num_tokens: Approximate target prompt length in tokens.

    Returns:
        The generated prompt string.
    """
    return f"{seed} " + " ".join(["hi"] * num_tokens)


def load_tokenizer(model: str):
    """Load ``model``'s tokenizer via the fast ``tokenizers`` library.

    Returns an ``(encode, decode)`` pair: ``encode(text) -> list[int]`` and
    ``decode(ids) -> str``. Returns ``(None, None)`` if the tokenizer can't be
    loaded, signalling callers to fall back to :func:`_basic_prompt`.

    Only the lightweight ``tokenizers`` library is used (import ~0.01s); the
    heavy ``transformers`` library is intentionally avoided. Load this ONCE
    and reuse it across all prompts.

    Args:
        model: Model name / repo id whose tokenizer to load.

    Returns:
        ``(encode, decode)`` callables, or ``(None, None)`` on failure.
    """
    try:
        # Third Party
        from tokenizers import Tokenizer

        tok = Tokenizer.from_pretrained(model)
        return (lambda text: tok.encode(text).ids), (lambda ids: tok.decode(ids))
    except Exception as exc:  # tokenizer unavailable (offline/gated/no deps)
        print(
            f"[bench_ttft] tokenizers for {model!r} unavailable ({exc}); "
            "falling back to a basic repeated-word prompt (length approximate).",
            file=sys.stderr,
        )
        return None, None


def make_prompt(num_tokens: int, seed: int, encode=None, decode=None) -> str:
    """Generate a deterministic prompt of ``num_tokens`` tokens.

    When ``encode``/``decode`` (from :func:`load_tokenizer`) are provided the
    prompt is trimmed to *exactly* ``num_tokens`` tokens. Otherwise it falls
    back to :func:`_basic_prompt` (seed prefix + repeated ``"hi"``), whose
    length is approximate. Deterministic for a given ``seed``.

    Args:
        num_tokens: Target prompt length in tokens.
        seed: RNG seed for reproducible prompts.
        encode: ``text -> list[int]`` tokenizer callable, or ``None``.
        decode: ``list[int] -> str`` tokenizer callable, or ``None``.

    Returns:
        A prompt string of (exactly or approximately) ``num_tokens`` tokens.
    """
    if encode is None or decode is None:
        return _basic_prompt(seed, num_tokens)

    # Build random-ish text from the seed, then trim to exactly num_tokens
    # token ids and decode. Over-generate so the first encode usually suffices.
    rng = random.Random(seed)
    text = " ".join(
        "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 8)))
        for _ in range(num_tokens * 2)
    )
    token_ids = encode(text)
    while len(token_ids) < num_tokens:
        text += " " + "".join(rng.choices(string.ascii_lowercase, k=5))
        token_ids = encode(text)
    return decode(token_ids[:num_tokens])


def send_completion(url: str, model: str, prompt: str, timeout_s: float) -> float:
    """POST a single-token completion request. Returns elapsed wall-clock
    seconds (TTFT proxy: max_tokens=1, so end-to-end ≈ TTFT)."""
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
        }
    ).encode("utf-8")
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
    header = f"{'prompt_tokens':>14} {'n':>4} " + " ".join(f"{m:>14}" for m in metrics)
    print(header, file=sys.stderr)
    for n_tok in sorted(by_len):
        group = by_len[n_tok]
        medians = " ".join(
            f"{statistics.median(r[m] for r in group):>14.4f}" for m in metrics
        )
        print(f"{n_tok:>14} {len(group):>4} {medians}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--node-a-url",
        required=True,
        help="vLLM URL on Node A, e.g. http://10.0.0.1:8100",
    )
    p.add_argument(
        "--node-b-url",
        required=True,
        help="vLLM URL on Node B, e.g. http://10.0.0.2:8200",
    )
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument(
        "--prompt-tokens",
        type=int,
        nargs="+",
        default=[1000, 2000, 4000, 8000],
        help="Approx prompt lengths to sweep.",
    )
    p.add_argument("--repeat", type=int, default=5, help="Trials per prompt length.")
    p.add_argument(
        "--label",
        required=True,
        help="Tag for the rows (e.g. cxl-static, nixl-controller).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed for prompt generation; per-trial seeds "
        "derive from it. Default: current timestamp (printed "
        "to stderr so a run can be reproduced with --seed).",
    )
    p.add_argument("--timeout-s", type=float, default=120.0)
    p.add_argument("--out", required=True, help="CSV output path.")
    args = p.parse_args()

    seed_base = args.seed if args.seed is not None else int(time.time())
    print(f"[bench_ttft] seed={seed_base}", file=sys.stderr)

    # Load the model tokenizer once so prompts hit an exact token length.
    encode, decode = load_tokenizer(args.model)

    rows = []
    for n_tok in args.prompt_tokens:
        for trial in range(args.repeat):
            seed = seed_base + n_tok * 1000 + trial
            prompt = make_prompt(n_tok, seed, encode, decode)

            # Warmup on A — discard.
            send_completion(args.node_a_url, args.model, prompt, args.timeout_s)
            # Measured request on A — local cache (warm).
            t_a_local = send_completion(
                args.node_a_url, args.model, prompt, args.timeout_s
            )
            # First request on B — triggers cross-node fetch.
            t_b_cold = send_completion(
                args.node_b_url, args.model, prompt, args.timeout_s
            )
            # Second request on B — post-fetch hit.
            t_b_warm = send_completion(
                args.node_b_url, args.model, prompt, args.timeout_s
            )

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
