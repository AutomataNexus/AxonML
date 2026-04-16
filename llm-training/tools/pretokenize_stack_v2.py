#!/usr/bin/env python3
"""Pre-tokenize The Stack v2 into u32-LE shards for Trident-Coder training.

Streams The Stack v2 from HuggingFace datasets, filters to a code-language
allowlist, tokenizes with our trident-coder-bpe (32k vocab), and writes the
token IDs to disk as flat little-endian u32 shards consumable by the
`train_trident_code` Rust binary.

Run this on a machine with good I/O (Colab with a mounted drive beats the
Windows-on-WSL dev box). Do NOT run it from inside AxonML's build path —
The Stack v2 is hundreds of GB and the tokenized output is still many GB.

Requirements (pip install):
    datasets tokenizers huggingface_hub

Environment:
    HF_TOKEN=<token>   # required, from /opt/RESOURCES.md or vault

Usage:
    python3 pretokenize_stack_v2.py \\
        --tokenizer /opt/AxonML/tokenizers/trident-coder-bpe/tokenizer.json \\
        --out /mnt/stack-v2-trident \\
        --languages python,rust,javascript,typescript,go,cpp,c \\
        --shard-size-tokens 500_000_000 \\
        --max-shards 40 \\
        --seed 1337

    # Then in Rust:
    # cargo run --release --bin train_trident_code --features cuda -- \\
    #     --config 1b --dataset /mnt/stack-v2-trident/shard_0000.bin ...
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Pre-tokenize The Stack v2 for Trident-Coder training."
    )
    ap.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("/opt/AxonML/tokenizers/trident-coder-bpe/tokenizer.json"),
        help="Path to trident-coder-bpe tokenizer.json",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory. Shards will be written as shard_NNNN.bin (u32 LE).",
    )
    ap.add_argument(
        "--languages",
        type=str,
        default="python,rust,javascript,typescript,go,cpp,c",
        help="Comma-separated list of Stack v2 language names to include.",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default="bigcode/the-stack-v2-dedup",
        help="HuggingFace dataset name. Defaults to the dedup subset.",
    )
    ap.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to stream.",
    )
    ap.add_argument(
        "--shard-size-tokens",
        type=int,
        default=500_000_000,
        help="Target tokens per output shard before rolling to a new file.",
    )
    ap.add_argument(
        "--max-shards",
        type=int,
        default=40,
        help="Stop after this many shards (0 = unlimited).",
    )
    ap.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="If > 0, stop after this many documents total (testing).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Dataset shuffling seed (applied to the streaming loader).",
    )
    ap.add_argument(
        "--min-doc-chars",
        type=int,
        default=64,
        help="Skip documents shorter than this many chars.",
    )
    ap.add_argument(
        "--max-doc-chars",
        type=int,
        default=100_000,
        help="Truncate documents longer than this many chars pre-tokenize.",
    )
    return ap.parse_args()


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main() -> int:
    args = parse_args()

    # Lazy imports so --help works without the heavy deps.
    try:
        from datasets import load_dataset
        from tokenizers import Tokenizer
    except ImportError as e:
        log(
            f"Missing dependency: {e}\n"
            f"pip install datasets tokenizers huggingface_hub"
        )
        return 2

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        log(
            "HF_TOKEN environment variable is not set.\n"
            "See /opt/RESOURCES.md for the vault path, e.g.:\n"
            "  export HF_TOKEN=$(vault kv get -field=api_token secret/api/huggingface)"
        )
        return 2

    if not args.tokenizer.exists():
        log(f"Tokenizer not found at {args.tokenizer}")
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(args.tokenizer))
    vocab_size = tok.get_vocab_size()
    log(f"Loaded tokenizer ({vocab_size} tokens) from {args.tokenizer}")

    languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    log(f"Languages: {', '.join(languages)}")
    log(f"Dataset  : {args.dataset} (split={args.split})")

    # Stack v2 has one subset per language. Load each and interleave.
    from itertools import cycle

    streams = []
    for lang in languages:
        ds = load_dataset(
            args.dataset,
            lang,
            split=args.split,
            streaming=True,
            token=hf_token,
        ).shuffle(seed=args.seed + hash(lang) % 1000, buffer_size=10_000)
        streams.append(iter(ds))

    def interleave(streams):
        """Round-robin across language streams, skipping exhausted ones."""
        active = list(streams)
        while active:
            next_active = []
            for it in active:
                try:
                    yield next(it)
                    next_active.append(it)
                except StopIteration:
                    continue
            active = next_active

    iterator = interleave(streams)

    shard_idx = 0
    shard_path = args.out / f"shard_{shard_idx:04d}.bin"
    shard_fp = open(shard_path, "wb")
    shard_tokens = 0
    total_tokens = 0
    total_docs = 0

    log(f"Writing to {shard_path}...")

    try:
        for doc in iterator:
            text = doc.get("content") or doc.get("text") or ""
            if not text:
                continue
            if len(text) < args.min_doc_chars:
                continue
            if len(text) > args.max_doc_chars:
                text = text[: args.max_doc_chars]

            enc = tok.encode(text)
            ids = enc.ids
            if not ids:
                continue

            shard_fp.write(struct.pack(f"<{len(ids)}I", *ids))
            shard_tokens += len(ids)
            total_tokens += len(ids)
            total_docs += 1

            if total_docs % 1000 == 0:
                log(
                    f"  docs={total_docs:,} tokens={total_tokens:,} "
                    f"shard={shard_idx} shard_tokens={shard_tokens:,}"
                )

            if shard_tokens >= args.shard_size_tokens:
                shard_fp.close()
                log(
                    f"  rolled shard_{shard_idx:04d}.bin @ {shard_tokens:,} tokens"
                )
                shard_idx += 1
                if args.max_shards and shard_idx >= args.max_shards:
                    log(f"Hit --max-shards={args.max_shards}; stopping.")
                    shard_fp = None
                    break
                shard_path = args.out / f"shard_{shard_idx:04d}.bin"
                shard_fp = open(shard_path, "wb")
                shard_tokens = 0
                log(f"Writing to {shard_path}...")

            if args.max_docs and total_docs >= args.max_docs:
                log(f"Hit --max-docs={args.max_docs}; stopping.")
                break
    finally:
        if shard_fp is not None:
            shard_fp.close()

    log(
        f"Done. shards={shard_idx + 1} docs={total_docs:,} tokens={total_tokens:,}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
