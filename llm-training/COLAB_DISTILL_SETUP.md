# Colab A100 — Draft-Distill Training Setup

Run sequence for a fresh Colab A100 session. Pipeline smoke already passed locally at `/opt/AxonML/llm-training/checkpoints/draft_distill_smoke/` (50 steps, tiny student, Qwen3-0.6B teacher, CE/KL both decreasing). The real 50k-step run goes here.

## 1. Boot (~12 min)

```bash
# Verify A100
!nvidia-smi | head -5

# Install Rust
!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.85
!source $HOME/.cargo/env && rustc --version

# CUDA toolkit already provisioned on Colab A100 runtimes (CUDA 12.x)
!nvcc --version | head -4
```

## 2. Pull AxonML

```bash
# Option A: clone a snapshot you've pushed to a private remote
!git clone --depth 1 git@github.com:AutomataNexus/AxonML.git /content/AxonML

# Option B (simpler): upload a tarball
# From local: tar cz -C /opt AxonML | split -b 1900m - /mnt/d/axonml-src.tgz.
# Upload the .aa / .ab parts via Colab file panel; reassemble on Colab:
# !cat /content/axonml-src.tgz.* | tar xz -C /content
```

## 3. Build with CUDA

```bash
%cd /content/AxonML
!source $HOME/.cargo/env && cargo build --release -p llm-training --features cuda --bin train_draft_distill
!ls -lh llm-training/target/release/train_draft_distill
```

A100 has more cores but AxonML's crates hit a serial bottleneck in the build; expect ~15-20 min first time.

## 4. Upload inputs

```bash
# Teacher GGUFs (pick one: 0.6B or 1.7B; 4B if you want more headroom)
# Easiest: download fresh on Colab from HF hub (no upload cost):
!mkdir -p /content/models/qwen3-1.7b
!wget -q -O /content/models/qwen3-1.7b/Qwen3-1.7B-Q4_K_M.gguf \
  "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen_Qwen3-1.7B-Q4_K_M.gguf"

# Or upload your local copy if HF is rate-limiting you.

# Tokenized corpus (pre-tokenized bin):
# Upload from local /opt/datasets/fineweb-qwen/tokens.bin
# (produced by: tokenize_corpus --gguf <qwen3-gguf> --input fineweb.txt --output tokens.bin)
```

## 5. Launch

```bash
!source $HOME/.cargo/env && cd /content/AxonML && \
  ./llm-training/target/release/train_draft_distill \
    --teacher-gguf /content/models/qwen3-1.7b/Qwen3-1.7B-Q4_K_M.gguf \
    --tokens-bin /content/datasets/fineweb-qwen/tokens.bin \
    --arch 0.6b \
    --seq-len 2048 \
    --batch-size 16 \
    --epochs 1 \
    --steps 50000 \
    --lr 3e-4 \
    --warmup 500 \
    --temperature 3.0 \
    --ce-weight 0.1 \
    --weight-decay 0.1 \
    --grad-clip 1.0 \
    --log-every 50 \
    --generate-every 0 \
    --checkpoint-every-steps 1000 \
    --keep-last-k 5 \
    --output-dir /content/checkpoints/draft_distill \
    2>&1 | tee /content/distill.log &
```

`--batch-size 16` assumes A100-40GB. For A100-80GB or H100 bump to 32. If OOM, drop seq-len to 1024.

## 6. Handle Colab disconnects

Session caps at 12h (free) / 24h (Pro). Use the built-in resume:

```bash
# Resume from most-recent checkpoint after reconnect
!source $HOME/.cargo/env && cd /content/AxonML && \
  ./llm-training/target/release/train_draft_distill \
    ... same flags as above ... \
    --resume latest
```

`ResumeMode::Latest` finds the most recent checkpoint in `--output-dir` automatically. Steps, optimizer state, LR schedule all restored.

## 7. Monitor + intervene from local

While Colab runs, tunnel the training monitor back to your laptop:

```bash
# On Colab:
!pip install pyngrok -q
!ngrok authtoken <YOUR_TOKEN>
from pyngrok import ngrok
# The training monitor binds to a random port printed in distill.log line `training monitor: http://127.0.0.1:XXXX`
!grep -oE 'monitor: http://[^ ]+' /content/distill.log | head -1
public_url = ngrok.connect(<PORT>).public_url
print(public_url)
```

Open `public_url` in your local browser for live loss curves. Or run `nexus-training-ticker` locally pointed at the ngrok URL — you'd need to add a `--monitor-url` flag (current ticker hardcodes the local unix socket; Colab adaptation is out of scope for this run).

## 8. Post-training — pull weights + export GGUF

```bash
# Pull the final checkpoint back to local
# From Colab: !zip -r /content/final.zip /content/checkpoints/draft_distill/best_model.axonml
# Download via Colab file panel to /opt/AxonML/checkpoints/draft_distill/

# Convert to GGUF locally (axonml_to_gguf in llm-training/tools/)
./llm-training/target/release/axonml_to_gguf \
  --input /opt/AxonML/checkpoints/draft_distill/best_model.axonml \
  --arch qwen3 \
  --quant f16 \
  --output /opt/AxonML/models/nexus-mlm-0.6b/nexus-mlm-0.6b.gguf

# Smoke-test in nexus-serve
/opt/AxonML/nexus-serve/target/release/nexus-serve \
  --model /opt/AxonML/models/nexus-mlm-0.6b/nexus-mlm-0.6b.gguf \
  --port 11435
```

## 9. Use as spec-decode draft

```bash
/opt/AxonML/nexus-serve/target/release/spec_bench \
  --target /opt/AxonML/models/deepseek-r1-distill-qwen-7b/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf \
  --draft  /opt/AxonML/models/nexus-mlm-0.6b/nexus-mlm-0.6b.gguf \
  --prompt "Write a paragraph about the history of computing." \
  -n 128 -g 3 -q
```

Target acceptance α ≥ 0.75 at γ=3 → projected 45-50 tok/s end-to-end.

## Budget

| Phase | A100 wall-time |
|---|---|
| Build Rust + CUDA | 15-20 min |
| Load teacher + init student | 30 s |
| Warmup (500 steps) | 10-15 min |
| Main run (49.5k steps) | 12-18 h |
| Total | **~13-19 h** (fits in one 24h Pro session) |

If it runs out mid-flight, the `--resume latest` flow restarts cleanly from the last 1000-step checkpoint.
