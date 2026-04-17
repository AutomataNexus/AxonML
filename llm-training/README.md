# llm-training

End-to-end training binaries for the AxonML LLM architectures, each trained on a real text corpus (default: the complete works of Shakespeare) with GPU acceleration, a live browser monitor, rotating checkpointing, resume support, and operator pause/resume/stop controls over a Unix socket.

**Version:** 0.6.1 — updated 2026-04-16.

**Crates used from AxonML:** `axonml-core`, `axonml-tensor`, `axonml-autograd`, `axonml-nn`, `axonml-optim`, `axonml-llm`, `axonml-serialize`, `axonml` (for `TrainingMonitor`).

This is a **standalone crate** — the `Cargo.toml` declares its own empty `[workspace]` so `cargo` doesn't pull in the full AxonML workspace. Framework crates are referenced by `path = "/opt/AxonML/crates/..."`.

---

## The Training Binaries

Nine architecture-specific training binaries plus the `train_ctl` operator CLI:

| Model | Novel Features | Binary | Status |
|-------|----------------|--------|--------|
| GPT-2 | Decoder-only transformer (golden-path reference) | `train_gpt2` | Works |
| LLaMA | RoPE, GQA, SwiGLU | `train_llama` | Works |
| Mistral | Sliding-window attention, GQA | `train_mistral` | Works |
| Phi | Partial RoPE, compact design | `train_phi` | Works |
| BERT | Bidirectional encoder, masked-LM (15% masking) | `train_bert` | Works |
| SSM / Mamba | Selective S6 scan, depthwise conv | `train_ssm` | Works |
| Hydra | Hybrid SSM + windowed attention (`forward_with_loss`) | `train_hydra` | Works |
| Chimera | Sparse MoE + Differential Attention (`forward_with_loss`) | `train_chimera` | Works |
| Trident-Coder | 1.58-bit ternary SLM (byte-level BPE, linear-warmup + cosine LR) | `train_trident_code` | Works |
| — | Operator CLI over training Unix socket | `train_ctl` | Works |

The Trident paper code (FP32 vs 1.58-bit comparison, dense baseline) lives separately in `/opt/AxonML/papers/trident-blog/`.

## Corpora

Default for every trainer except `train_trident_code`: `/opt/datasets/text/shakespeare.txt` — Project Gutenberg Shakespeare, tokenized character-level (~98 vocab). Every binary accepts `--corpus PATH` to swap in a different text file.

`train_trident_code` uses the 32k-vocab byte-level BPE at `/opt/AxonML/tokenizers/trident-coder-bpe/tokenizer.json` and expects pre-tokenized u32 little-endian `.bin` shards. A smoke config auto-tokenizes Shakespeare to `/tmp/shakespeare.trident-bpe.bin` and caches it. The Python pre-tokenizer `tools/pretokenize_stack_v2.py` emits u32-LE shards from The Stack v2 for real runs.

## Usage

```bash
# Train GPT-2 on Shakespeare with GPU + monitor + resume
cargo run --release --bin train_gpt2 -p llm-training --features cuda

# Custom configuration
cargo run --release --bin train_gpt2 -p llm-training --features cuda -- \
    --epochs 10 --bs 32 --seq-len 128 --d-model 256 --layers 6

# Resume from the latest checkpoint
cargo run --release --bin train_gpt2 -p llm-training --features cuda -- \
    --resume latest

# Start fresh (ignore any existing checkpoints)
cargo run --release --bin train_gpt2 -p llm-training --features cuda -- --fresh

# Resume from a specific file
cargo run --release --bin train_gpt2 -p llm-training --features cuda -- \
    --resume /path/to/checkpoint.axonml

# Trident-Coder smoke run (CPU, ~30M params, Shakespeare-tokenized)
cargo run --release --bin train_trident_code -- \
    --config smoke --steps 100 --seq-len 64 --batch-size 8 --lr 3e-4

# Trident-Coder 1B run on H100 / Colab
cargo run --release --bin train_trident_code --features cuda -- \
    --config 1b --dataset /mnt/stack-v2.bin --steps 100000 \
    --seq-len 4096 --batch-size 4 --lr 3e-4 \
    --checkpoint-every-steps 1000 --keep-last-k 10
```

## Features Every Trainer Has

All nine `train_*` binaries share the `TrainingLifecycle` subsystem (`src/lifecycle.rs`), which is hard-wired — there is no opt-out flag.

| Feature | Description |
|---------|-------------|
| **GPU acceleration** | `--features cuda` moves params + inputs to `Device::Cuda(0)` |
| **Live browser monitor** | `axonml::TrainingMonitor` launches automatically at `http://127.0.0.1:<auto>` with real-time loss/PPL charts |
| **Best model tracking** | Saves `best_model.axonml` + `checkpoint_best.axonml` whenever the tracked metric improves |
| **Latest checkpoint** | Writes `checkpoint_latest.axonml` at the end of each epoch (used by `--resume latest`) |
| **Periodic epoch checkpoints** | Writes `checkpoint_epoch_NNNN.axonml` each epoch |
| **Rotating step-level checkpoints** | `--checkpoint-every-steps N` writes `checkpoint_step_<global_step>.axonml` every N steps; `--keep-last-k K` prunes older files. `N=0` disables the feature. |
| **Final checkpoint on exit** | Graceful stop flushes `checkpoint_final.axonml` so weeks-long runs never lose progress |
| **Resume support** | `--resume latest\|best\|<path>` with name-based then shape-based parameter matching; `--fresh` to ignore checkpoints |
| **Text sampling** | Periodic greedy generation mid-training so you can watch the model learn |
| **Final sample** | Full generation dump at end of training |

## Lifecycle Controls (`train_ctl`)

Every training process binds a Unix socket at `/tmp/axonml-train-<pid>.sock` with a convenience symlink at `/tmp/axonml-train-latest.sock`. The `train_ctl` binary speaks its plaintext protocol so operators don't have to `nc -U` by hand.

| Command | Effect |
|---------|--------|
| `train_ctl status` (default) | Prints a JSON status blob: model, pid, output dir, epoch, global step, last loss, paused/stopping flags, uptime, param count, monitor URL |
| `train_ctl pause` | Pauses after the current step (also `SIGUSR1` on the training process) |
| `train_ctl resume` | Resumes (also `SIGUSR2`) |
| `train_ctl stop` | Graceful stop + `checkpoint_final.axonml` flush, then exit (also `SIGINT`/`SIGTERM`) |
| `train_ctl checkpoint` | Flushes an ad-hoc step checkpoint on the next poll |
| `train_ctl list` | Scans `/tmp` for `axonml-train-*.sock` files (skipping the `latest` symlink) and prints a per-PID status block |
| `--socket PATH` / `--pid N` | Target a specific training run instead of `latest` |

Signals are also honored directly on the training PID: `SIGINT` / `SIGTERM` → graceful stop with final checkpoint, `SIGUSR1` → pause, `SIGUSR2` → resume. A dedicated `axonml-signal-dispatch` thread polls the raw `signal_hook` flags and translates them into the shared `ControlFlags`, avoiding non-async-signal-safe work inside the handlers.

## Default Hyperparameters (GPT-2, illustrative)

| Parameter | Default | Flag |
|-----------|---------|------|
| Context window | 128 | `--seq-len` |
| Hidden dim | 192 | `--d-model` |
| Transformer blocks | 4 | `--layers` |
| Attention heads | 6 | `--heads` |
| Batch size | 16 | `--bs` |
| Epochs | 5 | `--epochs` |
| Learning rate | 3e-4 | `--lr` |
| Steps per epoch | 500 | `--steps` |
| Log cadence | every 50 steps | `--log-every` |
| Generation cadence | every 100 steps | `--generate-every` |
| RNG seed | 1337 | `--seed` |
| Resume mode | latest | `--resume` |
| Step-checkpoint cadence | 0 (disabled) | `--checkpoint-every-steps` |
| Step-checkpoints kept | 5 | `--keep-last-k` |

Other binaries override defaults for their architecture (BERT uses 3 epochs and 150 steps/epoch with `--mlm-prob 0.15`; `train_trident_code` switches defaults on `--config smoke | 1b | 3b`). Run any binary with `--help` for its full flag set.

## Output Layout

Every trainer writes into `checkpoints/<model>/` by default (under `/opt/AxonML/llm-training/checkpoints/...`):

```
llm-training/checkpoints/gpt2/
├── best_model.axonml                # weights only — for inference
├── checkpoint_best.axonml           # full checkpoint (model + optim state)
├── checkpoint_latest.axonml         # for --resume latest
├── checkpoint_epoch_NNNN.axonml     # end-of-epoch snapshot
├── checkpoint_step_<global>.axonml  # rotating step-level, pruned to keep-last-k
└── checkpoint_final.axonml          # flushed on graceful stop
```

Active per-model subdirs today: `bert/`, `chimera/`, `gpt2/`, `hydra/`, `llama/`, `mistral/`, `phi/`, `ssm/`. Trident checkpoints land in `trident-smoke/`, `trident-1b/`, or `trident-3b/` depending on `--config`.

## Shared Utilities (`src/lib.rs`)

Every `train_*` binary pulls these from the crate root:

- `CharTokenizer::from_corpus` — deterministic character-level tokenizer with token 0 reserved for unknown / padding (`'\0'`).
- `TextDataset` — sliding-window next-token-prediction dataset with `sample_batch` returning a flat `Vec<u32>` of shape `[batch_size * seq_len]`.
- `lcg_range` — seedable linear congruential generator (Numerical Recipes constants) so batch sampling doesn't depend on an external RNG crate.
- `format_count` — thousands-separator formatter for param counts / dataset sizes.
- `ResumeMode` + `find_checkpoint` + `load_model_from_checkpoint` — checkpoint resume with name-based matching then shape-based in-order fallback, working for any AxonML `Module`.
- `shifted_cross_entropy` — causal-LM loss that shifts logits/labels by one position, flattens to `[N, V]`, and moves the f32 target tensor onto the logits' device so the fused GPU cross-entropy kernel triggers. Out-of-range labels are defensively clamped to 0. (Hydra and Chimera expose their own `forward_with_loss` and don't use this helper.)
- `read_corpus` — opinionated corpus loader that prints a friendly Shakespeare-path hint on failure.
- `lifecycle` — pause/resume/stop/checkpoint subsystem re-exported as `TrainingLifecycle` / `TrainingLifecycleBuilder` / `LoopAction`.

## Why Not Modify `crates/axonml-llm/examples/`?

The `examples/` directory in the locked `axonml-llm` crate contains reference training scripts that users may already depend on. This crate adds complete training pipelines (GPU, lifecycle controls, rotating checkpoints, resume, sampling, operator socket) alongside those examples — without modifying them — in a dedicated companion crate that lives outside the framework proper.
