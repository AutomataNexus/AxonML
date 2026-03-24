# Show HN: AxonML -- A PyTorch-equivalent ML framework written in Rust

**Title:** Show HN: AxonML -- A PyTorch-equivalent ML framework written in Rust

**URL:** https://github.com/AutomataNexus/AxonML

---

## First Comment (by andrewjewell)

Hi HN. I've been building AxonML for a bit now, testing often, and it's at v0.3.3 now -- 22 crates, 336 Rust source files, 1,076+ passing tests. It's a from-scratch ML framework in pure Rust aiming for PyTorch parity, dual licensed MIT/Apache-2.0.

I'm sharing it because I think the "Rust for ML" space is still underexplored relative to its potential, and I wanted to show what one person building full-time can produce.

### What's built

The full stack, bottom to top:

**Core compute:** N-dimensional tensors with broadcasting (NumPy rules), arbitrary shapes, views, slicing. Reverse-mode automatic differentiation with a tape-based computational graph. GPU backends for CUDA (GPU-resident tensors, cuBLAS GEMM, 20+ element-wise kernels with automatic dispatch), Vulkan, Metal, and WebGPU.

**Neural networks:** Linear, Conv1d/2d, MaxPool, AvgPool, AdaptiveAvgPool, BatchNorm1d/2d, LayerNorm, GroupNorm, InstanceNorm2d, Dropout, RNN/LSTM/GRU (with cell variants), MultiHeadAttention, CrossAttention, full Transformer encoder/decoder, Seq2SeqTransformer, Embedding. Loss functions: MSE, CrossEntropy, BCE, BCEWithLogits, L1, SmoothL1, NLL. Initialization: Xavier, Kaiming, Orthogonal.

**Optimizers:** SGD (with momentum/Nesterov), Adam, AdamW, RMSprop, Adagrad, LBFGS, LAMB. GradScaler for mixed precision. LR schedulers: Step, Cosine, OneCycle, Warmup, ReduceLROnPlateau, MultiStep, Exponential.

**Distributed training:** DDP, Fully Sharded Data Parallel (ZeRO-2/ZeRO-3), Pipeline Parallelism with microbatching, Tensor Parallelism.

**LLM architectures:** BERT (encoder, sequence classification, masked LM), GPT-2 (decoder, LM head), LLaMA (RMSNorm, RotaryEmbedding, GroupedQueryAttention), Mistral, Phi. Text generation with top-k, top-p, temperature sampling. Pretrained model hub configs.

**Ecosystem tooling:** ONNX import/export (40+ operators, opset 17), model quantization (INT4/INT5/INT8/F16, block-based with calibration, ~8x size reduction at Q4), kernel fusion (automatic pattern detection, FusedLinear, up to 2x on memory-bound ops), JIT compilation (graph optimization, Cranelift foundation), profiling (timeline with Chrome trace export, bottleneck analyzer).

**Vision/Audio/NLP:** ResNet, VGG, ViT architectures, image transforms, MFCC/spectrogram, BPE tokenizer, vocabulary management.

**Full application stack:** CLI with 50+ commands, terminal UI (ratatui-based dashboard), web dashboard (Leptos/WASM with WebSocket), Axum REST API server with JWT auth, MFA (TOTP + WebAuthn), model registry, inference endpoint deployment, in-browser terminal via WebSocket PTY, Prometheus metrics, Weights & Biases integration, Kaggle integration.

I estimate PyTorch parity at roughly 92-95% for the core training loop and standard layer types.

### Production deployment -- this is the part I'm most proud of

AxonML is running live production inference right now. 12 HVAC predictive maintenance models (LSTM autoencoders for anomaly detection + GRU failure predictors) are deployed across 6 Raspberry Pi edge controllers, monitoring commercial building equipment across 5 facilities. Each model is cross-compiled to `armv7-unknown-linux-musleabihf` (static musl), runs as a PM2-managed daemon at ~2-3 MB RSS, and exposes predictions via REST API at 1 Hz.

Beyond those initial 6 controllers, I've built out models for 35 HVAC areas across 7 facilities (FCOG, Warren, Huntington, Akron, Hopebridge, NE Realty, and a unified NexusBMS system with 22 trained models covering air handlers, boilers, chillers, VAVs, fan coils, make-up air units, DOAS units, pumps, and steam systems). 69 `.axonml` model files total.

The deployment pipeline: AxonML training on CPU --> `.axonml` serialized weights --> cross-compiled ARM inference binary (pure tensor ops, no autograd overhead) --> PM2 process management on the Pi --> HTTP endpoints for integration with the building management system.

This is the use case that drove most of the framework's development. The models needed to be small, fast, and run on constrained hardware without Python.

### Kaggle competition usage

I'm also using AxonML for the Deep Past Initiative Kaggle competition -- machine translation from Akkadian cuneiform to English. Full seq2seq Transformer (encoder-decoder with multi-head attention, sinusoidal positional encoding, BPE tokenization) trained on ~1,561 parallel sentence pairs. It compiles and trains end-to-end through AxonML. Evaluated on BLEU + chrF++.

### Honest limitations

- **Ecosystem maturity.** PyTorch has thousands of contributors, Hugging Face, torchvision's pretrained zoo, a decade of Stack Overflow answers. AxonML has one developer and a growing but small set of pretrained weights. If you need a specific pretrained model, you'll probably need to convert it yourself via ONNX
- **GPU kernel coverage.** CUDA support works -- cuBLAS GEMM, 20+ element-wise kernels, GPU-resident tensors -- but the coverage is nowhere near cuDNN-backed PyTorch. Some operations will fall back to CPU. Vulkan/Metal/WebGPU backends are implemented but less battle-tested than CUDA
- **Python interop doesn't exist.** If your workflow depends on pandas, scikit-learn preprocessing, or Jupyter notebooks, you'll need to handle data prep separately. This is a Rust-native framework

### Why Rust for ML?

Three reasons from practical experience:

1. **Single-binary deployment.** `cargo build --release --target armv7-unknown-linux-musleabihf` gives you a statically-linked inference binary. No Python runtime, no pip, no conda, no Docker. Copy it to a Raspberry Pi and it runs. This is why my HVAC models actually work in production
2. **Compile-time safety.** Dimension mismatches, type errors, and lifetime issues are caught before you start a training run, not 3 hours into one
3. **Memory predictability.** No GC pauses, no reference counting overhead on the hot path, deterministic memory layout. On a Raspberry Pi with 1 GB RAM running at 2-3 MB RSS, this matters

GitHub: https://github.com/AutomataNexus/AxonML

Happy to answer questions about the architecture, the borrow-checker-vs-autograd challenges, the edge deployment pipeline, or the Kaggle experience.
