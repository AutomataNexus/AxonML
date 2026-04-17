# Trident-Coder BPE Tokenizer

32,000-vocab byte-level BPE tokenizer for the 1B-parameter 1.58-bit Trident-Coder LLM. GPT-2-style byte-level encoding: lossless round-trip on any UTF-8 input, 256-glyph base alphabet, no `<unk>`.

**Version:** 0.6.1 — updated 2026-04-16. Tokenizer artifacts themselves were built 2026-04-13 23:58:51 EDT.

## Special tokens (stable IDs)

From `eval_metrics.json` → `special_token_ids` (source of truth):

| id | token |
|----|-------|
| 0  | `<|endoftext|>` |
| 1  | `<|pad|>` |
| 2  | `<|user|>` |
| 3  | `<|assistant|>` |
| 4  | `<|system|>` |
| 5  | `<|tool_use|>` |
| 6  | `<|tool_result|>` |
| 7  | `<|tool_end|>` |

`tokenizer_config.json` maps these into the HF `PreTrainedTokenizerFast` slots: `eos_token = <|endoftext|>`, `pad_token = <|pad|>`, and the remaining six listed under `additional_special_tokens`. `bos_token` and `unk_token` are null by design — the byte-level alphabet handles any UTF-8 input losslessly so an `<unk>` slot is unnecessary.

The last four tokens back Anthropic Messages API tool-call delivery:

- `<|tool_use|>` … `<|tool_end|>` wraps an assistant-emitted tool-use block
- `<|tool_result|>` … `<|tool_end|>` wraps the user-turn tool_result block

The plan is for Trident-Coder to be fine-tuned to emit these sequences verbatim so that nexus-serve's `/v1/messages` endpoint can parse the stream into Anthropic-format `content[*].type == "tool_use"` blocks natively (replacing the `<tool_use>` / `</tool_use>` prompt-template parser used for BitNet b1.58 today — see `nexus-serve/src/api/messages.rs`).

## Training corpus

Sources:

- `codeparrot/github-code-clean` (public, ungated) — parquet shards streamed one at a time, filtered to 7 target languages, then deleted from disk.
- Local `.rs` sources under `/opt/` to supplement Rust, because `the-stack-smol` and `the-stack-v2` were gated behind an access-request flow that HF fine-grained tokens cannot satisfy automatically.

Filter rules (applied to every source file before it hits the BPE trainer):

- File size ≤ 1 MB
- Non-ASCII byte ratio ≤ 50%
- No line longer than 500 characters (minified / generated heuristic)
- No `@generated` / `DO NOT EDIT` marker in the first 10 lines
- Non-empty, non-whitespace-only

Per-language cap: 700 MB post-filter. Rust hits its natural ceiling well below the cap because only ~2.9 MB of Rust appears per 360 MB github-code-clean shard. The filesystem scrape tops Rust up with source from `/opt/AxonML`, `/opt/trident-blog`, `/opt/Prometheus`, `/opt/NexusOracle`, and `/opt/FerumMail`.

Note: the shipped `training_corpus_stats.json` is currently an empty JSON object (`{}`) — per-language file counts and filter-rejection counts were not persisted at build time. Only the eval metrics below are authoritative.

## Evaluation

Held-out samples drawn from the local filesystem using a shuffle-ordering deliberately different from the build-time walk. The evaluator reports bytes-per-token (B/tok) and chars-per-token (C/tok) per language — lower is better compression.

| language | tokens | B/tok | C/tok |
|----------|-------:|------:|------:|
| python | 1,037,666 | **3.3876** | 3.3845 |
| rust | 2,132,964 | **3.4822** | 3.4775 |
| typescript | 1,412,654 | **3.7114** | 3.7057 |

Target: ≤ 3.6 B/tok on Rust (good), ideal ≤ 3.3 — Rust hits 3.48 at this build.

**Round-trip caveat:** `eval_metrics.json` currently reports `round_trip_ok_on_samples: false`. The byte-level BPE is in principle lossless on arbitrary UTF-8, so this is an open item to investigate (sample-level round-trip check failing on at least one eval sample — possibly an evaluator bug rather than a tokenizer bug). Treat the encoder as compression-verified but not yet round-trip-verified at sample level until this flips to `true`.

## Model max length

`tokenizer_config.json` sets `model_max_length = 8192`. The Trident-Coder-1B training config uses `seq_len = 4096` at 1B scale (see `llm-training/src/bin/train_trident_code.rs`).

## Loading

### Raw `tokenizers` crate / python

```python
from tokenizers import Tokenizer
tk = Tokenizer.from_file("/opt/AxonML/tokenizers/trident-coder-bpe/tokenizer.json")
ids = tk.encode("fn main() { println!(\"hi\"); }").ids
text = tk.decode(ids)
```

### Transformers wrapper

```python
from transformers import PreTrainedTokenizerFast
tk = PreTrainedTokenizerFast.from_pretrained("/opt/AxonML/tokenizers/trident-coder-bpe")
```

### Rust (AxonML)

```rust
use tokenizers::Tokenizer;
let tk = Tokenizer::from_file(
    "/opt/AxonML/tokenizers/trident-coder-bpe/tokenizer.json"
).unwrap();
let enc = tk.encode("fn main() {}", false).unwrap();
```

The Trident-Coder trainer (`llm-training/src/bin/train_trident_code.rs`) and its Python pre-tokenizer (`llm-training/tools/pretokenize_stack_v2.py`) both load this exact file.

## Files

- `tokenizer.json` — the tokenizer model (load with `Tokenizer::from_file`)
- `tokenizer_config.json` — HF `PreTrainedTokenizerFast` config (eos / pad / additional special tokens, `model_max_length = 8192`)
- `eval_metrics.json` — vocab size, special-token IDs, per-language B/tok + C/tok, round-trip flag
- `training_corpus_stats.json` — placeholder (currently `{}`)
- `README.md` — this file

## Build reproduction

The full BPE-training scripts are not checked into this directory — only the materialized tokenizer artifacts. Tokenizer-build provenance lives in the git history of this directory. The `_workdir/` scratch space used during the build was deleted at end of run.

Author: Andrew Jewell Sr. (AutomataNexus LLC), 2026.
