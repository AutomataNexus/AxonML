# llm-training

End-to-end training pipelines for the nine AxonML LLM architectures, each trained on a real text corpus (default: the complete works of Shakespeare) with GPU acceleration, live browser monitoring, checkpointing, and resume support.

**Crates used from AxonML:** `axonml-core`, `axonml-tensor`, `axonml-autograd`, `axonml-nn`, `axonml-optim`, `axonml-llm`, `axonml-serialize`, `axonml` (for `TrainingMonitor`).

This is a **standalone crate** — `[workspace]` is empty so `cargo` doesn't pull in the full AxonML workspace. It depends on the framework via `path = "/opt/AxonML/crates/..."`.

---

## The Nine LLMs

| Model | Novel Features | Binary | Status |
|-------|----------------|--------|--------|
| GPT-2 | Decoder-only transformer | `train_gpt2` | ✓ |
| LLaMA | RoPE, GQA, SwiGLU | `train_llama` | ✓ |
| Mistral | Sliding-window attention, GQA | `train_mistral` | ✓ |
| Phi | Partial RoPE, compact design | `train_phi` | ✓ |
| BERT | Bidirectional classifier (binary classification) | `train_bert` | ✓ |
| SSM / Mamba | Selective S6 scan, depthwise conv | `train_ssm` | ✓ |
| Hydra | Hybrid SSM + windowed attention | `train_hydra` | ✓ |
| Chimera | Sparse MoE + Differential Attention | `train_chimera` | ✓ |
| Trident | 1.58-bit ternary weights | (see `papers/trident-blog/`) | ✓ (paper) |

## Dataset

Default corpus: `/opt/datasets/text/shakespeare.txt` — 5.4 MB of Project Gutenberg Shakespeare, tokenized character-level (~98 vocab). Every binary accepts `--corpus PATH` to use a different text file.

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
cargo run --release --bin train_gpt2 -p llm-training --features cuda --fresh

# Resume from a specific file
cargo run --release --bin train_gpt2 -p llm-training --features cuda -- \
    --resume /path/to/checkpoint.axonml
```

## Features Every Trainer Has

| Feature | Description |
|---------|-------------|
| **GPU acceleration** | `--features cuda` moves params + inputs to `Device::Cuda(0)` |
| **Live browser monitor** | `axonml::TrainingMonitor` — opens Chromium with real-time loss/PPL charts |
| **Best model tracking** | Saves `best_model.axonml` + `checkpoint_best.axonml` whenever validation loss improves |
| **Latest checkpoint** | Always saves `checkpoint_latest.axonml` at end of each epoch (for resume) |
| **Periodic epoch checkpoints** | Saves `checkpoint_epoch_NNNN.axonml` each epoch |
| **Resume support** | `--resume latest\|best\|<path>` or `--fresh` to ignore checkpoints |
| **Text sampling** | Periodic greedy generation mid-training so you can watch the model learn |
| **Final sample** | Full generation dump at end of training |

## Default Hyperparameters (GPT-2)

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

## Output

All checkpoints land in `checkpoints/<model>/`:

```
llm-training/checkpoints/gpt2/
├── best_model.axonml            # weights only — for inference
├── checkpoint_best.axonml       # full checkpoint (model + optim + state)
├── checkpoint_latest.axonml     # for --resume latest
└── checkpoint_epoch_NNNN.axonml # periodic
```

## Why Not Modify `crates/axonml-llm/examples/`?

The `examples/` directory in the locked `axonml-llm` crate contains reference training scripts that users may already depend on. This project adds complete training pipelines (with GPU, checkpointing, resume, sampling) alongside those examples — without modifying them — in a dedicated companion crate that lives outside the framework proper.
