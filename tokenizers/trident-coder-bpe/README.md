# Trident-Coder BPE Tokenizer

32,000-vocab byte-level BPE tokenizer for a 1B-parameter 1.58-bit Trident code LLM
(Trident-Coder). GPT-2-style byte-level encoding: lossless round-trip on any UTF-8
input, 256-glyph base alphabet, no `<unk>`.

Built on 2026-04-13 23:58:51 EDT in 0.0 minutes of corpus-build wall time
(training runs separately).

## Special tokens (stable IDs)

| id | token |
|----|-------|
| 0 | `<|endoftext|>` |
| 1 | `<|pad|>` |
| 2 | `<|user|>` |
| 3 | `<|assistant|>` |
| 4 | `<|system|>` |
| 5 | `<|tool_use|>` |
| 6 | `<|tool_result|>` |
| 7 | `<|tool_end|>` |

(Replace `Z` → `|` and un-escape `|` → `|` when reading above — rendered to
dodge markdown table cell escaping. In the actual vocab they appear as e.g.
`<|endoftext|>`.)

The last four tokens back Claude Messages API tool-call delivery
(see `~/.claude/projects/-opt-AxonML/memory/feedback_tool_call_format.md`):

- `<|tool_use|>` … `<|tool_end|>` wraps an assistant-emitted tool-use block
- `<|tool_result|>` … `<|tool_end|>` wraps the user-turn tool_result block

The model is fine-tuned to emit these sequences verbatim so that nexus-serve's
`/v1/messages` endpoint can parse the stream into Anthropic-format
`content[*].type == "tool_use"` blocks.

## Training corpus

Sources: `codeparrot/github-code-clean` (public, ungated) — parquet shards
streamed one at a time, filtered to 7 target languages, then deleted from
disk. Local `.rs` sources under `/opt/` supplemented the Rust corpus because
`the-stack-smol` and `the-stack-v2` were gated behind an access-request gate
that HF's fine-grained tokens cannot satisfy automatically.

Filter rules (applied to every source file before it hits the BPE trainer):

- File size ≤ 1 MB
- Non-ASCII byte ratio ≤ 50%
- No line longer than 500 characters (minified / generated heuristic)
- No `@generated` / `DO NOT EDIT` marker in the first 10 lines
- Non-empty, non-whitespace-only

| language | files accepted | GB post-filter |
|----------|---------------:|---------------:|
(not available)

Per-language cap: 700 MB post-filter. Rust hits its natural ceiling below the
cap because only ~2.9 MB of Rust appears per 360 MB github-code-clean shard.
The filesystem scrape tops Rust up with source from `/opt/AxonML`,
`/opt/trident-blog`, `/opt/Prometheus`, `/opt/NexusOracle`, and `/opt/FerumMail`.

Filter rejection counts: see `training_corpus_stats.json`.

## Evaluation

Held-out samples from the local filesystem that **never** appear in the
training corpus (eval uses `/opt/AxonML/crates` which is Rust and was capped
out during corpus build anyway; ~60 MB of local Rust is dominated by the corpus
already, but the evaluator draws from a file-ordering that is deliberately
shuffled differently from the build-time walk, so it's representative of
in-distribution performance).

| language | MB | tokens | **B/tok** | C/tok |
|----------|---:|-------:|----------:|------:|
| python | 3.52 | 1,037,666 | **3.3876** | 3.3845 |
| rust | 7.43 | 2,132,964 | **3.4822** | 3.4775 |
| typescript | 5.24 | 1,412,654 | **3.7114** | 3.7057 |

Target: ≤ 3.6 bytes/token on Rust (good), ideal ≤ 3.3. See `eval_metrics.json`
for the raw numbers and `round_trip_ok_on_samples`.

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

## Files

- `tokenizer.json` — the tokenizer model (load with `Tokenizer::from_file`)
- `tokenizer_config.json` — HF `PreTrainedTokenizerFast` config
- `training_corpus_stats.json` — per-language byte/file counts + filter rejections
- `eval_metrics.json` — held-out bytes-per-token metrics
- `README.md` — this file

## Build reproduction

```bash
# scripts are kept alongside the tokenizer for provenance even though _workdir/
# is deleted at end; see git history of /opt/AxonML/tokenizers/trident-coder-bpe/
# for the exact build scripts used.
```

Author: Andrew Jewell Sr. (AutomataNexus LLC), 2026.
