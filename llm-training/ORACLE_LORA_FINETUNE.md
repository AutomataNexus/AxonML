# Oracle LoRA fine-tune — DeepSeek-R1-Distill-Qwen-7B on Claude Code traces

End-to-end runbook for fine-tuning `DeepSeek-R1-Distill-Qwen-7B` into an
agentic Oracle model using the user's own Claude Code session traces as the
SFT corpus. The trained adapter is merged, converted to GGUF, and loaded
into `nexus-serve` on port 11436; the NexusOracle desktop app's header
toggle then flips the reasoning backend from cloud Claude to local Oracle
with no daemon restart.

**Target:** 7B parameter LoRA, rank 16, ≈ 430 training sessions
(9.7 K tool_use examples, ≈ 29 MB corpus), A100 80GB, ~3–5 hours.

**Upstream of:** NexusOracle task #53 (local-reasoning toggle in the
desktop app — the runtime infrastructure is already landed; this runbook
produces the model it will point at).

---

## 0. Prereqs

- Claude Code traces present at `~/.claude/projects/*/*.jsonl`.
- `cargo` (nightly 2024 edition) to build the export binary.
- GCP project with `roles/notebooks.admin`, `roles/compute.admin`, and
  `roles/storage.admin` on the service account you'll SSH as.
- Claude Code Workbench template `nexus-a100-80gb` already registered
  (a2-ultragpu-1g + 200 GB standard disk + Python 3.12 + idle-shutdown
  30 min).

---

## 1. Build the SFT corpus

Rust binary that walks `~/.claude/projects`, reconstructs sessions from
the event log, renders each into DeepSeek-R1 chat-template form with
inline `<tool_use>{json}</tool_use>` blocks matching nexus-serve's
Messages-API parser, and writes 95/5 train/val JSONL.

```bash
cd /opt/AxonML/llm-training
cargo build --release --bin claude_trace_export

./target/release/claude_trace_export \
  --projects-dir /home/devops/.claude/projects \
  --output-dir   /opt/datasets/oracle-lora \
  --val-split    0.05 \
  --seed         42
```

Expected output (as of 2026-04-19):

```
done: 778 sessions → 428 train + 23 val (rejected 327)
  reject[bad_opener] = 5
  reject[interrupted] = 39
  reject[no_tool_use] = 7
  reject[too_long] = 265
  reject[too_short] = 11
```

Artifacts in `/opt/datasets/oracle-lora`:

- `corpus_train.jsonl` (~ 28 MB, 428 sessions)
- `corpus_val.jsonl`   (~ 1.8 MB, 23 sessions)
- `corpus_stats.json` — turn / tool_use / char counts + rejection reasons

### Why the filters are what they are

| Filter | Default | Rationale |
|---|---|---|
| `--max-chars` | 131072 (≈ 32K tokens) | R1-Distill's native context is 32K; anything larger can't be trained without chunking |
| `--max-tool-result-chars` | 8192 | a single cat of a giant log would otherwise dominate one training sample |
| `--drop-interrupted` | true | "[Request interrupted]" traces teach incomplete behaviors |
| no tool_use | reject | goal is agentic tool-calling, not pure chat |

---

## 2. Spin up the A100 Workbench

Run once from your local shell (has `gcloud` + the service-account creds
for the GCP project that owns the `nexus-a100-80gb` template):

```bash
# Replace with your actual values; saved as vars so subsequent commands
# reuse the same names.
export PROJECT=automatanexus
export ZONE=us-central1-a
export VM=oracle-lora-sft

gcloud workbench instances create "$VM" \
  --project="$PROJECT" \
  --location="$ZONE" \
  --from-template=nexus-a100-80gb \
  --idle-shutdown-timeout=1800s
```

Workbench exposes JupyterLab on port 8080 + SSH via IAP. Prefer SSH:

```bash
gcloud compute ssh "$VM" --project="$PROJECT" --zone="$ZONE" --tunnel-through-iap
```

First-time-on-this-VM setup (Python 3.12 is pre-installed; CUDA 12 drivers
+ PyTorch 2.4 + transformers are in the default DL image):

```bash
pip install --user -U unsloth bitsandbytes peft trl accelerate datasets
python -c "import unsloth, peft, bitsandbytes; print('ok')"
```

---

## 3. Upload the corpus

From the local host:

```bash
gcloud compute scp \
  --recurse /opt/datasets/oracle-lora \
  "$VM":~/oracle-lora \
  --project="$PROJECT" --zone="$ZONE" --tunnel-through-iap
```

---

## 4. Training script — `finetune_oracle.py`

Create this **on the A100 VM** at `~/finetune_oracle.py`:

```python
"""LoRA SFT of DeepSeek-R1-Distill-Qwen-7B on Claude Code traces.

Uses Unsloth for 2–3× throughput over bare transformers + PEFT,
bitsandbytes 4-bit base weights, gradient checkpointing, bf16 mixed
precision. Target: ≈ 4 GB LoRA adapter weights at rank 16, ≈ 14 GB
4-bit base weights, ≈ 50 GB activations at bs=2 + 4K seq — fits A100
80GB with 10 GB headroom for mempool jitter.
"""

import os
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

MAX_SEQ_LEN = 4096  # per-sample truncation; sessions longer than this
                    # are chunked by the SFT trainer's packing.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,      # auto: bf16 on A100
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

train_ds = load_dataset("json", data_files="/home/jupyter/oracle-lora/corpus_train.jsonl", split="train")
val_ds   = load_dataset("json", data_files="/home/jupyter/oracle-lora/corpus_val.jsonl",   split="train")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    dataset_text_field="text",   # matches claude_trace_export output
    max_seq_length=MAX_SEQ_LEN,
    packing=True,                # pack multiple short samples per seq
    args=SFTConfig(
        output_dir="/home/jupyter/oracle-lora-out",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,   # effective bs = 16
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        report_to="none",
        seed=42,
    ),
)

trainer.train()
model.save_pretrained("/home/jupyter/oracle-lora-out/final")
tokenizer.save_pretrained("/home/jupyter/oracle-lora-out/final")
```

Run:

```bash
python ~/finetune_oracle.py 2>&1 | tee ~/train.log
```

Expected wall-clock: ~3–5 hours for 3 epochs on 428 sessions at
bs=2/grad_accum=8, seq 4K, packed.

### Hyperparameter rationale

- **rank=16** — the Oracle task is a style shift + tool-convention shift,
  not a new domain; rank 8 underfits the `<tool_use>` format, rank 32+
  overfits to user-specific tool-use quirks.
- **target_modules = all 7 linear projections** — standard for
  Qwen/DeepSeek; skipping `gate/up/down_proj` loses ~30% of the quality
  gain per the Unsloth benchmark suite.
- **lr=2e-4 + cosine + 3% warmup** — canonical Unsloth recipe; R1-Distill
  is stable at this rate, no need for the lower 5e-5 rate used for
  larger-scale SFT.
- **3 epochs** — 428 sessions × 3 = 1284 passes ≈ 30 K–40 K gradient
  steps at effective bs=16; enough for the adapter to converge without
  overfitting the small corpus.
- **bf16 not fp16** — A100 native, no loss-scaling needed, and no NaN
  risk with large tool_result token sequences.

---

## 5. Merge LoRA → base and export GGUF

On the A100 VM:

```bash
python - <<'PY'
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained(
    model_name="/home/jupyter/oracle-lora-out/final",
    max_seq_length=4096,
    load_in_4bit=False,  # merge requires full-precision base
)
# Saves a full merged model in fp16 directly.
model.save_pretrained_merged(
    "/home/jupyter/oracle-merged",
    tok,
    save_method="merged_16bit",
)
PY
```

Convert the merged fp16 weights to GGUF Q4_K_M (the quantization
nexus-serve is tuned for on 7B models):

```bash
# Pre-installed inside the nexus-a100-80gb template image; otherwise:
pip install --user llama-cpp-python

python -m llama_cpp.llama_convert_hf_to_gguf \
  /home/jupyter/oracle-merged \
  --outfile /home/jupyter/oracle-q4km.gguf \
  --outtype q4_k_m
```

(Alternative: use `llama.cpp`'s `convert_hf_to_gguf.py` +
`quantize Q4_K_M` two-step if the combined script lags behind the
model release.)

Sanity-check the GGUF on-VM with llama.cpp before shipping:

```bash
./llama-cli -m /home/jupyter/oracle-q4km.gguf \
  -p "<｜begin▁of▁sentence｜>You are Oracle.<｜User｜>What is 2+2?<｜Assistant｜>" \
  -n 32 -temp 0
```

Must return coherent text (e.g., "4" or a short reasoning trace). Gibberish
means the chat-template or tokenizer config didn't round-trip — revisit
step 4's `tokenizer.save_pretrained` and re-export.

---

## 6. Ship the GGUF back

From the local host:

```bash
mkdir -p /opt/AxonML/models/oracle-distill
gcloud compute scp \
  "$VM":~/oracle-q4km.gguf \
  /opt/AxonML/models/oracle-distill/oracle-r1-distill-q4km.gguf \
  --project="$PROJECT" --zone="$ZONE" --tunnel-through-iap

# Tear down the Workbench so the clock stops running.
gcloud workbench instances delete "$VM" \
  --project="$PROJECT" --location="$ZONE" --quiet
```

---

## 7. Load into nexus-serve + flip the Oracle toggle

Launch nexus-serve against the new GGUF on the local-reasoning port:

```bash
pkill -f nexus-serve 2>/dev/null; sleep 2
nohup /opt/AxonML/nexus-serve/target/release/nexus-serve \
  --model /opt/AxonML/models/oracle-distill/oracle-r1-distill-q4km.gguf \
  --port  11436 --quantized \
  > /tmp/nexus_oracle.log 2>&1 &

# Wait for /health to return 200
until curl -s http://127.0.0.1:11436/health -o /dev/null \
        -w "%{http_code}" 2>/dev/null | grep -q 200; do sleep 1; done
```

Point NexusOracle's daemon at port 11436 — edit `~/.nexusoracle/config.toml`:

```toml
[model]
claude_local_base_url = "http://127.0.0.1:11436"
claude_local_model    = "oracle-r1-distill-q4km"
reasoning_mode        = "local"    # or "auto" to fall back to cloud on errors
```

Restart the NexusOracle daemon (or hit `POST /api/v1/config/reasoning-mode`
directly), then open the desktop app. The header segment control
(Local / Auto / Cloud) will now show **Local** enabled. Clicking it flips
the router atomically — no restart.

### Smoke tests (after cut-over)

```bash
# 1. Coherent baseline — must produce real language, not "Okay00000..."
curl -s -X POST http://127.0.0.1:11436/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"oracle-r1-distill-q4km",
       "max_tokens":64,"temperature":0.0,
       "messages":[{"role":"user","content":"Reply with exactly: hello world"}]}'

# 2. Tool-call reliability — must emit a tool_use block
curl -s -X POST http://127.0.0.1:11436/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"oracle-r1-distill-q4km",
       "max_tokens":512,"temperature":0.1,
       "tools":[{"name":"read_file",
                 "description":"Read a file.",
                 "input_schema":{"type":"object",
                                 "properties":{"path":{"type":"string"}},
                                 "required":["path"]}}],
       "messages":[{"role":"user",
                    "content":"What is in /etc/hostname?"}]}'
```

Expected on test 2: `stop_reason="tool_use"`, `content` contains a
`tool_use` block with `name="read_file"` and `input.path="/etc/hostname"`.
The LoRA should make this MORE reliable than the base R1-Distill (task
#52's template fix already made base-model tool calls work; the LoRA is
what makes the model actually *choose* to call them agentically, matching
the user's own Claude Code style).

---

## 8. What if it's worse than base R1-Distill?

Fallback paths, ordered by cost:

1. **Flip to `reasoning_mode = "auto"`** — router prefers local but falls
   back to cloud on errors/timeouts. Zero-cost recovery; user never sees
   a regression.
2. **Flip to `reasoning_mode = "cloud"`** — hard revert to cloud Claude.
   Oracle LoRA stays in the file tree, no harm done.
3. **Re-SFT with larger corpus** — wait a month for more Claude Code
   usage, rerun `claude_trace_export`, retrain.
4. **Escalate to the TPU v5litepod-8 path** — full SFT (not LoRA) on the
   base model using PyTorch/XLA. Documented out of scope for this run;
   use if step 3 plateaus.

---

## 9. Handoff / what lives where

| Asset | Path |
|---|---|
| Corpus builder source | `/opt/AxonML/llm-training/src/bin/claude_trace_export.rs` |
| Built corpus | `/opt/datasets/oracle-lora/corpus_{train,val}.jsonl` |
| Corpus stats | `/opt/datasets/oracle-lora/corpus_stats.json` |
| LoRA adapter | `~/oracle-lora-out/final/` (on the VM) |
| Merged fp16 | `~/oracle-merged/` (on the VM) |
| Deployed GGUF | `/opt/AxonML/models/oracle-distill/oracle-r1-distill-q4km.gguf` |
| nexus-serve config | launch flags above; no persistent config file |
| NexusOracle config | `~/.nexusoracle/config.toml` — `reasoning_mode`, `claude_local_base_url`, `claude_local_model` |
| UI toggle wiring | `apps/oracle-app/src/components/reasoning_toggle.rs` + `src-tauri/src/commands.rs::{get,set}_reasoning_mode` |
| Runtime router | `crates/oracle-daemon/src/reasoning_router.rs` (task #53) |

---

## Author

Andrew Jewell Sr. — AutomataNexus LLC
ORCID: 0009-0005-2158-7060
Created: 2026-04-19
