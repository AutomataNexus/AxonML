# nexus-serve

Pure-Rust LLM inference server. Replaces ollama with native AxonML inference over an OpenAI-compatible REST API.

**Status:** Working on Qwen2.5 Coder 1.5B (CUDA + CPU). Tested end-to-end: `"What is 2+2?"` → `"2 + 2 equals 4."`

---

## Features

- **GGUF model loading** — reads ollama's quantized model blobs directly (Q4_K, Q6_K, Q8_0, Q4_0, F16, F32)
- **LLaMA-family architectures** — LLaMA, Qwen2, Mistral (split-halves RoPE, GQA, optional Q/K/V biases)
- **KV cache** for fast incremental decoding
- **HuggingFace tokenizer** (`tokenizer.json`) + GGUF-embedded BPE fallback
- **CUDA GPU acceleration** via `--features cuda`
- **Multi-model registry** — load multiple GGUF files, serve them from one endpoint
- **OpenAI-compatible API** — drop-in replacement for OpenAI/ollama clients

## Quick Start

```bash
# Build with GPU
cargo build --release --features cuda

# Load with friendly aliases (sage/oracle match nexus-agent configs)
target/release/nexus-serve \
  --alias sage   /usr/share/ollama/.ollama/models/blobs/sha256-29d8c98fa6b098e200069bfb88b9508dc3e85586d20cba59f8dda9a808165104 \
  --alias oracle /usr/share/ollama/.ollama/models/blobs/sha256-4c27e0f5b5adf02ac956c7322bd2ee7636fe3f45a8512c9aba5385242cb6e09a \
  --port 11435

# Query using the alias
curl http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sage",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 50,
    "temperature": 0.0
  }'
```

## Model Aliases

Use `--alias NAME PATH` to load a model AND register a friendly name for it. Requests for `NAME` will route to the canonical GGUF model name.

```bash
nexus-serve --alias sage /path/to/qwen.gguf --alias oracle /path/to/gemma.gguf
```

Aliases appear in `/v1/models` with `object: "model-alias"`. Clients can use either the alias or the canonical model name.

## Configuration

Persistent settings live in a TOML file. CLI flags always win over config values, and config values always win over built-in defaults.

**Precedence:** `CLI flag  >  config file  >  default`

**Default path:** `~/.config/nexus-serve/config.toml`
**Override path:** `nexus-serve --config /some/other/path.toml`

A fully-commented starter config ships at [`config.example.toml`](config.example.toml) in this repo. Copy it into place:

```bash
mkdir -p ~/.config/nexus-serve
cp config.example.toml ~/.config/nexus-serve/config.toml
```

Supported top-level keys:

| Key | Type | Default | Notes |
|---|---|---|---|
| `threads` | integer | all cores | Parallel block dequant + CPU matmul pool. Matches `--threads`. |
| `port` | integer | `11435` | Listen port. Matches `--port`. |
| `host` | string | `"0.0.0.0"` | Listen host. Matches `--host`. |
| `quantized` | bool | `false` | Lazy per-matmul dequant for memory savings. Matches `--quantized`. |

A `[hardware]` section is accepted for documentation and validation:

```toml
[hardware]
cpu    = "Intel Core Ultra 9 275HX"
cores  = 24
ram_gb = 64
```

At startup nexus-serve compares `hardware.cores` against `std::thread::available_parallelism()` and warns if they disagree — useful to catch "config copied to the wrong machine" situations. It also warns if `threads` (from CLI or config) exceeds detected cores.

Startup prints a `Resolved config` block so you can see which setting came from CLI, config, or default:

```
Resolved config:
  host      = 0.0.0.0               [default]
  port      = 11435                 [default]
  threads   = 24                    [config]
  quantized = true                  [cli]
  hardware  = Intel Core Ultra 9 275HX
```

## Endpoints

All OpenAI-compatible:

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Chat with ChatML template auto-applied |
| `POST` | `/v1/completions` | Raw text completion |
| `GET` | `/v1/models` | List loaded models |
| `GET` | `/health` | Server health check |

The server always applies the ChatML template (`<|im_start|>role\n...<|im_end|>`) for `/v1/chat/completions`. For models using a different template, use `/v1/completions` and format the prompt yourself.

## Architecture

```
src/
├── main.rs              CLI + server startup
├── lib.rs               Module exports
├── api/
│   ├── types.rs         OpenAI request/response types
│   └── routes.rs        HTTP handlers + ChatML formatter
├── model/
│   ├── gguf.rs          GGUF parser + quantized dequantization
│   │                      (Q4_K, Q6_K, Q8_0, Q4_0, F16 — ported from ggml-quants.c)
│   ├── inference.rs     Transformer forward pass + KV cache
│   │                      (LLaMA-family: RMSNorm → QKV → split-halves RoPE →
│   │                       attention → FFN(SiLU))
│   └── registry.rs      Multi-model registry
└── tokenizer/
    └── mod.rs           HuggingFace / GGUF BPE / char-level fallback
```

### Inference pipeline

For each layer:

1. RMS Norm (CPU — element-wise, fast)
2. Move normed input to weight device (CUDA)
3. QKV projections via `Tensor::matmul` on GPU
4. Add Q/K/V biases (Qwen2 has all three; LLaMA has none)
5. Split-halves RoPE (`(x[i], x[i+d/2])` pairs, not interleaved)
6. Append K/V to KV cache
7. Attention (GQA: `n_heads / n_kv_heads` queries share each KV head)
8. Output projection + residual
9. FFN (gate/up SiLU + down) + residual

## Quantization Support

Dequantization is ported verbatim from `ggml-quants.c` to guarantee compatibility with any GGUF file produced by llama.cpp / ollama.

| Type | Block size | Block bytes | Supported |
|---|---|---|---|
| F32 | 1 | 4 | ✓ |
| F16 | 1 | 2 | ✓ |
| BF16 | 1 | 2 | ✓ |
| Q8_0 | 32 | 34 | ✓ |
| Q4_0 | 32 | 18 | ✓ |
| Q4_K | 256 | 144 | ✓ (with 6-bit scales+mins, `get_scale_min_k4`) |
| Q6_K | 256 | 210 | ✓ (4-way split per 128 elements) |

Verified with standalone block tests against Python reference:

```bash
cargo test --release --test q4k_block_test
cargo test --release --test q6k_block_test
```

## Known Limitations

- **Streaming SSE** not implemented — responses are blocking
- **Gemma architecture** — not yet supported (requires different attention + rotary)
- **Phi, Mamba, MoE** — not yet supported
- **Concurrent requests** — correctness is fine (each `generate_stream()` call allocates its own `KvCache` on the stack; two requests return correct, prompt-specific answers in parallel). The shared `Arc<InferenceEngine>` only holds read-only weights. **Throughput, however, is not additive**: concurrent requests share the rayon CPU pool (and, for GPU, the CUDA context), so two simultaneous generations each run at roughly half the speed of a solo one. If you need real concurrent throughput, horizontal-scale by running multiple `nexus-serve` processes behind a load balancer.
- **Quantized-inference accuracy** — the lazy-dequant path (`--quantized`) is implemented (rayon-parallel block dequant, per-matmul scratch) but hasn't been accuracy-tested side-by-side with the eager-dequant f32 path yet. Prefer the eager path for now unless you're RAM-constrained.

## Historical Bugs (for reference)

Six distinct bugs had to be fixed for Qwen2 to produce coherent output. All are documented in `/home/devops/.claude/projects/-opt-AxonML/memory/reference_gguf_inference_gotchas.md`:

1. **F16 subnormal dequantization** — initial exponent was `-1` instead of `-14` (1000x scale error)
2. **Q4_K dequantization** — wrong sub-block loop structure; each 64-element chunk uses two different scales
3. **Q6_K dequantization** — wrong per-element approach; ggml uses 128-element 4-way split
4. **Split-halves RoPE** — our RoPE was interleaved (GPT-NeoX style); LLaMA/Qwen2/Mistral use split-halves
5. **V bias missing** — Qwen2 has Q/K/V biases; only Q/K were being applied
6. **ChatML template** — was passing raw `"role: content\n"` instead of proper `<|im_start|>` markers

## Port

Default is **11435** (ollama uses 11434, we're +1 so both can run side-by-side).
