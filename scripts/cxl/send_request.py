#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Send a single completion request to an OpenAI-compatible LLM server.

Generates a prompt of a specified token length and POSTs it to the
server's ``/v1/completions`` endpoint, printing the raw HTTP response.

The prompt is built from a seeded RNG so a given ``--seed`` always yields
the same prompt — pass a fixed seed to reproduce an experiment, or omit it
to use the current timestamp (printed to stderr so the run can be replayed).

When the model's tokenizer is importable (via ``transformers``) the prompt
is trimmed to *exactly* ``--prompt-length`` tokens; otherwise it falls back
to a ~4-chars/token heuristic (count is then approximate).

Usage:
    send_request.py HOST:PORT \
        [-m MODEL] [-n PROMPT_LEN] [-t MAX_TOKENS] [-s SEED]

Example:
    scripts/send_request.py localhost:8010 -n 2000 -s 42
"""

# Standard
from __future__ import annotations
import argparse
import json
import random
import string
import sys
import time
import urllib.error
import urllib.request

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def _random_words(rng: random.Random, count: int) -> str:
    """Return ``count`` space-joined random lowercase words."""
    words = [
        "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 8)))
        for _ in range(count)
    ]
    return " ".join(words)


def make_prompt(num_tokens: int, seed: int, model: str) -> str:
    """Generate a prompt of ``num_tokens`` tokens.

    Uses the model's real tokenizer (so the token count is exact) when one
    can be loaded; otherwise falls back to a ~4-chars/token word heuristic
    (approximate count). Deterministic for a given ``seed``.

    The tokenizer is loaded via the lightweight ``tokenizers`` library when
    available — its import is ~0.01s vs ~4s for ``transformers`` — falling
    back to ``transformers`` and finally the heuristic.

    Args:
        num_tokens: Target prompt length in tokens.
        seed: RNG seed for reproducible prompts.
        model: Model name used to load the matching tokenizer.

    Returns:
        A prompt string of (exactly or approximately) ``num_tokens`` tokens.
    """
    rng = random.Random(seed)

    encode, decode = _load_tokenizer(model)
    if encode is None:
        return _random_words(rng, num_tokens)

    # Over-generate, then trim to exactly num_tokens token ids and decode.
    text = _random_words(rng, num_tokens * 2)
    token_ids = encode(text)
    while len(token_ids) < num_tokens:
        # Rare for word-based text, but pad if the heuristic under-shot.
        text += " " + _random_words(rng, num_tokens)
        token_ids = encode(text)
    return decode(token_ids[:num_tokens])


def _load_tokenizer(model: str):
    """Load ``model``'s tokenizer, preferring the fast ``tokenizers`` lib.

    Returns an ``(encode, decode)`` pair of callables:
    ``encode(text) -> list[int]`` (no special tokens) and
    ``decode(ids) -> str``. Returns ``(None, None)`` if no tokenizer can be
    loaded, signalling the caller to use the char-count heuristic.

    Tries the lightweight ``tokenizers`` library first (import ~0.01s),
    then ``transformers`` (~4s import), then gives up.

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
    except Exception:
        pass

    try:
        # Third Party
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model)
        return (
            lambda text: tok.encode(text, add_special_tokens=False),
            lambda ids: tok.decode(ids),
        )
    except Exception as exc:  # no tokenizer available (offline/gated/no deps)
        print(
            f"[send_request] tokenizer for {model!r} unavailable ({exc}); "
            "falling back to ~4-chars/token heuristic (length approximate).",
            file=sys.stderr,
        )
        return None, None


def send_completion(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> tuple[str, float]:
    """POST a completion request; return the output text and latency.

    Args:
        endpoint: Server address as ``HOST:PORT``.
        model: Model name for the request.
        prompt: The prompt text.
        max_tokens: Max output tokens to generate.
        temperature: Sampling temperature.
        timeout_s: Socket timeout in seconds.

    Returns:
        ``(output_text, latency_s)`` where ``output_text`` is the generated
        completion (``choices[0].text``) and ``latency_s`` is the wall-clock
        time from sending the request to receiving the full response.

    Raises:
        urllib.error.URLError: On connection failure.
        urllib.error.HTTPError: On a non-2xx response (its body is returned
            by the caller's error handling).
    """
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"http://{endpoint}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
    latency_s = time.perf_counter() - t0

    output_text = json.loads(raw)["choices"][0]["text"]
    return output_text, latency_s


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send one completion request to an LLM server.",
    )
    parser.add_argument(
        "endpoint",
        help="LLM server address as HOST:PORT, e.g. localhost:8010",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "-n",
        "--prompt-length",
        type=int,
        default=100,
        help="Prompt length in tokens (default: 100).",
    )
    parser.add_argument(
        "-t",
        "--max-tokens",
        type=int,
        default=1,
        help="Max output tokens to generate (default: 1).",
    )
    parser.add_argument(
        "-T",
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0).",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Seed for prompt generation (default: current timestamp).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds (default: 120).",
    )
    args = parser.parse_args()

    if args.prompt_length < 1:
        parser.error("--prompt-length must be >= 1")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be >= 1")
    if args.temperature < 0:
        parser.error("--temperature must be >= 0")

    seed = args.seed if args.seed is not None else int(time.time())

    prompt = make_prompt(args.prompt_length, seed, args.model)

    try:
        output_text, latency_s = send_completion(
            args.endpoint,
            args.model,
            prompt,
            args.max_tokens,
            args.temperature,
            args.timeout,
        )
    except urllib.error.HTTPError as e:
        # Print the server's error body (often a useful JSON error message).
        print(e.read().decode("utf-8", errors="replace"), file=sys.stderr)
        print(f"HTTP {e.code} {e.reason}", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"request failed: {e.reason}", file=sys.stderr)
        return 1

    print(f"output:  {output_text!r}")
    print(f"latency: {latency_s:.3f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
