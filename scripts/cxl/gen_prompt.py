#!/usr/bin/env python
"""Generate a random prompt of an exact token length for /lookup_hits testing.

Samples random token IDs from the model's vocabulary (skipping special/added
tokens), then decodes to text and RE-ENCODES so the printed text and IDs are
mutually consistent — feeding the printed text back through the tokenizer
yields the printed IDs. This matters for the /lookup_hits probe: the IDs must
match what vLLM actually tokenizes the prompt into.

Usage:
    python gen_prompt.py <num_tokens> [--model NAME] [--seed N]
    python gen_prompt.py 512 --seed 0

Output (two lines on stdout):
    TEXT:<the prompt text>
    IDS:<comma-separated token ids>

Pipe the IDS line into the endpoint, e.g.:
    IDS=$(python gen_prompt.py 512 --seed 0 | sed -n 's/^IDS://p')
    http POST http://c1:8090/lookup_hits \
        model_name=meta-llama/Llama-3.1-8B-Instruct \
        token_ids:="[$IDS]"
"""
# Standard
import argparse
import random

# Third Party
from transformers import AutoTokenizer

_DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def generate_prompt(
    num_tokens: int,
    model_name: str,
    seed: int,
) -> tuple[str, list[int]]:
    """Generate a random prompt of exactly ``num_tokens`` tokens.

    Args:
        num_tokens: Target token length (the re-encoded prompt is trimmed to
            exactly this many IDs).
        model_name: HuggingFace model/tokenizer name.
        seed: RNG seed for reproducibility across nodes.

    Returns:
        A ``(text, token_ids)`` pair where ``token_ids`` has length
        ``num_tokens`` and ``tokenizer(text)`` reproduces ``token_ids``.

    Raises:
        ValueError: If ``num_tokens`` is not positive.
    """
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive (got {num_tokens})")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    rng = random.Random(seed)

    # Sample only "ordinary" vocab IDs: exclude special tokens (BOS/EOS/etc.)
    # and added tokens so decode->encode is stable and nothing injects extra
    # structure into the sequence.
    special_ids = set(tokenizer.all_special_ids)
    added_ids = set(getattr(tokenizer, "added_tokens_decoder", {}).keys())
    bad_ids = special_ids | added_ids
    candidate_ids = [i for i in range(tokenizer.vocab_size) if i not in bad_ids]

    # Oversample, decode, then re-encode (without auto special tokens) and trim
    # to the exact length. Re-encoding is what guarantees text<->IDs agree;
    # decode/encode is not always identity, so oversample to absorb shrinkage
    # and loop until we have enough.
    ids: list[int] = []
    text = ""
    oversample = max(num_tokens * 2, num_tokens + 32)
    while len(ids) < num_tokens:
        sampled = rng.choices(candidate_ids, k=oversample)
        text = tokenizer.decode(sampled, skip_special_tokens=True)
        ids = tokenizer.encode(text, add_special_tokens=False)
        oversample *= 2

    ids = ids[:num_tokens]
    text = tokenizer.decode(ids, skip_special_tokens=True)
    # Final re-encode so the emitted text and IDs are exactly consistent.
    ids = tokenizer.encode(text, add_special_tokens=False)[:num_tokens]
    return text, ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("num_tokens", type=int, help="target token length")
    parser.add_argument("--model", default=_DEFAULT_MODEL, help="tokenizer/model name")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    args = parser.parse_args()

    text, ids = generate_prompt(args.num_tokens, args.model, args.seed)
    print("\nTEXT:")
    print(text)
    print("\nIDS:")
    print(",".join(str(i) for i in ids))


if __name__ == "__main__":
    main()
