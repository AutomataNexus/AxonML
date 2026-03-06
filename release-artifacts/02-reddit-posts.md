# AxonML Reddit Posts

---

## Post 1: r/rust

### Title: AxonML v0.3.3 -- A 22-crate ML framework with autograd, CUDA, and production edge deployments

I've been building a PyTorch-equivalent ML framework in Rust and wanted to share both the project and some of the Rust-specific design decisions that came out of it.

**GitHub:** https://github.com/AutomataNexus/AxonML
**License:** MIT/Apache-2.0
**Stats:** 22 crates, 336 source files, 1,076+ tests, Rust 1.75+

#### Crate architecture

The framework is layered, bottom-up:

```
axonml-core         -- Device abstraction (CPU/CUDA/Vulkan/Metal/WebGPU), Storage<T>, DType
axonml-tensor       -- N-dim Tensor<T: Scalar>, views, broadcasting, 60+ ops
axonml-autograd     -- Variable, tape-based reverse-mode autodiff, AMP, gradient checkpointing
axonml-nn           -- Module trait, Linear/Conv/LSTM/GRU/Transformer/Attention layers
axonml-optim        -- Optimizer trait, SGD/Adam/AdamW/RMSprop/LBFGS/LAMB, LR schedulers
axonml-data         -- Dataset trait, DataLoader, samplers
axonml-serialize    -- .axonml format, StateDict, checkpoint management
axonml-onnx         -- ONNX import/export, 40+ operators, opset 17
axonml-quant        -- INT4/INT5/INT8/F16 quantization, block-based with calibration
axonml-fusion       -- Automatic kernel fusion (FusedLinear, FusedElementwise)
axonml-jit          -- IR, graph optimization (constant folding, DCE, CSE), Cranelift foundation
axonml-profile      -- Profiler, MemoryProfiler, TimelineProfiler (Chrome trace), BottleneckAnalyzer
axonml-distributed  -- DDP, FSDP (ZeRO-2/3), Pipeline Parallelism, Tensor Parallelism
axonml-vision       -- ResNet, VGG, ViT, image transforms, pretrained hub
axonml-audio        -- MelSpectrogram, MFCC, resample, augmentation
axonml-text         -- BPE/Char/Whitespace tokenizers, vocabulary
axonml-llm          -- BERT, GPT-2, LLaMA, Mistral, Phi architectures, text generation
axonml-cli          -- 50+ commands, W&B integration, Kaggle integration
axonml-tui          -- ratatui-based terminal dashboard
axonml-dashboard    -- Leptos/WASM web UI with WebSocket
axonml-server       -- Axum REST API, JWT auth, MFA, model registry, PTY terminal
axonml              -- Umbrella re-export crate
```

Each crate has its own test suite and can be pulled in independently via feature flags.

#### Rust-specific design decisions

**Tensor type system.** `Tensor<T: Scalar>` where `Scalar` is bounded by `Copy + Clone + Debug + Send + Sync + Pod + 'static`. This means tensors are generic over their scalar type at compile time. The tradeoff: you get type safety (can't accidentally add an `f32` tensor to an `i64` tensor), but you lose the ability to dynamically switch dtypes without enum dispatch. I chose compile-time safety over runtime flexibility here.

**Autograd without GC.** This was the hardest part. PyTorch's autograd relies on Python's reference counting + GC to manage the computational graph. In Rust, the tape-based approach uses `Arc<Mutex<...>>` for shared ownership of graph nodes. `Variable` wraps a `Tensor` and holds a reference to its gradient function. Backward traversal is topological sort over the graph. The borrow checker forces you to think carefully about when gradients are computed vs. when they're consumed -- which actually prevents a class of bugs where PyTorch would silently accumulate stale gradients.

The main pain point: intermediate operations need to create new `Variable`s that participate in the graph, which means `Variable::from_operation()` has to be public, and you end up with `Arc` allocations on every forward pass. I haven't found a way to avoid this without unsafe graph manipulation.

**`unsafe` usage.** Concentrated in a few places:
- CUDA backend (FFI to cuBLAS, raw pointer management for GPU memory)
- Tensor views (zero-copy slicing requires asserting that the underlying storage outlives the view -- enforced by `Arc<Storage>`)
- WebSocket PTY in the server (libc FFI for pseudo-terminal allocation)

The rest of the codebase is safe Rust. I use `clippy` with a strict configuration.

**Module trait design.** Neural network layers implement a `Module` trait with `forward()`, `parameters()`, `train()`, `eval()`. This is close to PyTorch's `nn.Module` but without inheritance -- composition via `Sequential` and manual struct nesting. Rust's lack of inheritance actually makes model architecture more explicit, which I've come to prefer.

#### Production deployment

The project's main production use case: 69 trained models for HVAC predictive maintenance across 7 commercial building facilities. 12 of those are running live inference on Raspberry Pi edge controllers (ARM, static musl binary, PM2-managed, ~2-3 MB RSS each). Cross-compiled with `cargo build --release --target armv7-unknown-linux-musleabihf`.

The deployment story is where Rust really shines. A single statically-linked binary that runs on a Pi with no runtime dependencies. Try doing that with PyTorch.

#### What I'd do differently

- The `Arc<Mutex<>>` autograd graph adds overhead. If I started over, I'd explore arena-based allocation for graph nodes
- Some trait bounds are more restrictive than necessary because I designed them before I fully understood the usage patterns
- I'd integrate `rayon` earlier -- parallelism was bolted on after initial design rather than being foundational

Questions, criticism, and PRs welcome.

---

## Post 2: r/MachineLearning

### Title: [P] AxonML: Pure-Rust ML framework at ~92-95% PyTorch parity, running production inference on edge hardware and Kaggle competitions

I built a machine learning framework in Rust that covers most of what you'd use PyTorch for. Sharing it because I think it's reached a point where it's genuinely useful, not just a proof of concept.

**GitHub:** https://github.com/AutomataNexus/AxonML
**License:** MIT/Apache-2.0 | **v0.3.3** | 22 crates | 1,076+ tests

#### What's implemented

**Layers:** Linear, Conv1d/2d, MaxPool, AvgPool, AdaptiveAvgPool, BatchNorm1d/2d, LayerNorm, GroupNorm, InstanceNorm2d, Dropout, RNN, LSTM, GRU (with cell variants), MultiHeadAttention, CrossAttention, TransformerEncoder, TransformerDecoder, Seq2SeqTransformer, Embedding.

**Optimizers:** SGD (momentum + Nesterov), Adam, AdamW, RMSprop, Adagrad, LBFGS, LAMB. GradScaler for AMP. Schedulers: StepLR, CosineAnnealing, OneCycle, Warmup, ReduceLROnPlateau, MultiStep, Exponential.

**Loss functions:** MSE, CrossEntropy, BCE, BCEWithLogits, L1, SmoothL1, NLL.

**LLM architectures:** Full implementations of BERT (encoder, sequence classification, masked LM), GPT-2 (decoder, LM head with generation), LLaMA (RMSNorm, RotaryEmbedding, GroupedQueryAttention), Mistral, Phi. Text generation with top-k, top-p, and temperature sampling.

**Distributed:** DDP, FSDP (ZeRO-2/ZeRO-3), Pipeline Parallelism, Tensor Parallelism.

**Infrastructure:** ONNX import/export (40+ ops), INT4/INT5/INT8/F16 quantization (~8x model size reduction at Q4), automatic kernel fusion, JIT compilation with graph optimization, model serialization with checkpoint management, CUDA GPU support (cuBLAS GEMM + element-wise kernels).

**Vision:** ResNet, VGG, ViT with pretrained weight hub. **Audio:** MelSpectrogram, MFCC. **NLP:** BPE tokenizer, vocabulary management.

#### Real-world usage

**Production HVAC monitoring.** 69 trained models across 7 commercial building facilities. LSTM autoencoders for anomaly detection, GRU networks for failure prediction. 12 models running live on Raspberry Pi edge controllers at ~2-3 MB RSS each, processing sensor data at 1 Hz. Models cover air handlers, boilers, chillers, VAVs, fan coils, make-up air units, DOAS, pumps, and steam systems. Cross-compiled to ARM, deployed as static binaries. These have been running in production.

**Kaggle: Deep Past Initiative.** Machine translation from Akkadian cuneiform to English. Full seq2seq Transformer trained on ~1,561 parallel sentence pairs with 5,571 unique source tokens. BPE tokenization, sinusoidal positional encoding, encoder-decoder with multi-head attention. Evaluated on BLEU + chrF++. The entire training pipeline -- data loading, tokenization, model definition, training loop, checkpoint management -- runs through AxonML.

#### Why not just use PyTorch?

For most people, you should use PyTorch. The ecosystem is enormous and mature.

AxonML fills a specific niche: when you need ML models that compile to a single binary, run on constrained hardware without Python, and benefit from Rust's compile-time safety guarantees. My HVAC deployment is the canonical example -- I needed sub-5MB inference processes on Raspberry Pis with no runtime dependencies.

If your workflow is Jupyter notebooks + Hugging Face + A100s, PyTorch is the right choice. If you're deploying to edge hardware, embedding inference in a Rust application, or want compile-time dimension checking, AxonML is worth looking at.

#### Limitations

- Pretrained model availability is limited compared to Hugging Face's ecosystem. You can import via ONNX, but native pretrained weights are currently ResNet, VGG, MobileNet, EfficientNet, BERT, GPT-2
- GPU kernel coverage is growing but not at cuDNN parity
- No Python bindings (this is by design -- it's Rust-native -- but it means no Jupyter workflow)
- One developer, so development velocity is inherently limited compared to PyTorch's team

#### Full application stack

Beyond the ML framework itself, AxonML includes a complete CLI (50+ commands), a terminal UI dashboard, a Leptos/WASM web dashboard with real-time training monitoring, and an Axum-based API server with JWT auth, model registry, and inference endpoint deployment. Weights & Biases integration for experiment tracking.

---

## Post 3: r/programming

### Title: AxonML: What happens when you try to rebuild PyTorch in Rust (22 crates, 1076+ tests, running in production)

Over the past year+ I've been building AxonML, a machine learning framework in Rust that aims for PyTorch-equivalent functionality. It's now at v0.3.3 with 22 crates, 336 source files, and 1,076+ passing tests.

**GitHub:** https://github.com/AutomataNexus/AxonML
**License:** MIT/Apache-2.0

#### Why Rust for ML?

The conventional wisdom is that ML frameworks should be Python-first with C++/CUDA backends. PyTorch, TensorFlow, JAX -- they all follow this pattern. I went a different direction for practical reasons:

1. **Deployment.** I needed ML inference running on Raspberry Pi edge controllers for HVAC monitoring. Python on a Pi with 1 GB RAM is painful. A statically-linked Rust binary at 2-3 MB RSS is not
2. **Compile-time correctness.** A type error in PyTorch surfaces as a runtime exception, sometimes hours into training. In Rust, the compiler catches it before you start
3. **No runtime dependencies.** `cargo build --release` produces one binary. No virtualenv, no pip, no system Python version conflicts, no GLIBC compatibility issues

#### Architecture

22 crates, layered from low-level compute to high-level application:

- **Compute layer:** `axonml-core` (device abstraction for CPU/CUDA/Vulkan/Metal/WebGPU), `axonml-tensor` (N-dim generic tensors), `axonml-autograd` (reverse-mode autodiff with AMP and gradient checkpointing)
- **ML layer:** `axonml-nn` (layers + losses), `axonml-optim` (optimizers + schedulers), `axonml-data` (DataLoader + samplers)
- **Infrastructure:** `axonml-serialize`, `axonml-onnx` (40+ operators), `axonml-quant` (INT4-INT8, F16), `axonml-fusion` (automatic kernel fusion), `axonml-jit` (Cranelift-based), `axonml-profile` (Chrome trace export)
- **Domain:** `axonml-vision` (ResNet, VGG, ViT), `axonml-audio` (MFCC, spectrogram), `axonml-text` (BPE tokenizer), `axonml-llm` (BERT, GPT-2, LLaMA, Mistral, Phi), `axonml-distributed` (DDP, FSDP, pipeline/tensor parallelism)
- **Application:** `axonml-cli` (50+ commands, W&B + Kaggle integration), `axonml-tui` (terminal dashboard), `axonml-dashboard` (Leptos/WASM web UI), `axonml-server` (Axum REST API with JWT, MFA, model registry, WebSocket PTY)

#### Interesting technical challenges

**Autograd in Rust.** PyTorch's computational graph relies on Python's reference counting. In Rust, you don't have a GC. The solution is a tape-based approach with `Arc<Mutex<>>` for shared graph node ownership. Every forward operation creates a new `Variable` that holds a reference to its gradient function. Backward pass does a topological sort. The Rust ownership model actually helps here -- it's impossible to accidentally read a gradient that's been freed, which is a real class of bugs in PyTorch.

**CUDA integration.** The CUDA backend uses FFI to cuBLAS for matrix multiplication and custom PTX kernels for element-wise operations. GPU memory is managed through a `CudaStorage` type that handles allocation/deallocation in `Drop`. The tricky part is ensuring tensors don't outlive their GPU context -- Rust's lifetime system handles this, but getting the API ergonomics right took several iterations.

**Generic tensors.** `Tensor<T: Scalar>` is generic over the scalar type. This means `Tensor<f32>` and `Tensor<f64>` are different types at compile time, so you can't accidentally mix precision. The downside is that dynamic dtype selection requires enum dispatch, which adds a level of indirection. I chose compile-time safety over runtime flexibility.

**Quantization.** Block-based quantization (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0) with calibration data. A 4-bit quantized model is ~8x smaller than f32. This was critical for edge deployment -- a 400K parameter model goes from ~1.6 MB to ~200 KB.

#### Production numbers

- 69 trained `.axonml` model files across 7 commercial building facilities
- 35 edge deployment areas covering air handlers, boilers, chillers, VAVs, fan coils, pumps, steam systems, DOAS, make-up air units
- 12 models running live inference on Raspberry Pi (ARM, static musl, PM2-managed)
- ~2-3 MB RSS per inference daemon
- 1 Hz inference cycle (poll sensor data, run model, expose predictions via HTTP)
- Also used for a Kaggle competition (Akkadian-to-English machine translation with a seq2seq Transformer)

#### What the Rust ecosystem still needs for ML

- Better BLAS story. There are Rust BLAS bindings but nothing approaching the optimization level of Intel MKL or OpenBLAS's hand-tuned assembly
- Mature autodiff. I built mine from scratch; the existing Rust autodiff crates weren't sufficient for a full framework
- GPU compute ecosystem. `wgpu` is great for graphics but the compute shader story for ML is still developing
- Pretrained model hosting. There's no Rust equivalent of Hugging Face's model hub

These aren't blockers -- clearly you can build a full framework without them -- but they represent friction that Python's ecosystem has already solved.

---

## Post 4: r/learnmachinelearning

### Title: I built a full ML framework in Rust -- here's what I learned about ML internals by building everything from scratch

If you're learning ML and want to deeply understand what happens under the hood of PyTorch, building your own framework is one of the best ways to do it. I spent the past year+ building AxonML, a PyTorch-equivalent framework in Rust, and I learned more about ML fundamentals from this project than from any course.

**GitHub:** https://github.com/AutomataNexus/AxonML (MIT/Apache-2.0, free to use and learn from)

#### What you learn by building a tensor library

When you use `torch.tensor([1, 2, 3])` in PyTorch, a lot happens that you never see. Building a tensor library teaches you:

- **Memory layout.** Tensors are contiguous blocks of memory with shape and stride metadata. A "view" (like `.reshape()` or `.transpose()`) doesn't copy data -- it just changes how you index into the same memory. Understanding strides is fundamental to understanding why some operations are fast and others aren't
- **Broadcasting.** When you add a `[3, 1]` tensor to a `[1, 4]` tensor, you get a `[3, 4]` result. The rules for this are surprisingly nuanced and implementing them forces you to truly understand them
- **Type generics.** A tensor needs to work with f32, f64, i32, i64, bool, etc. Designing a type system that handles all of these while remaining ergonomic teaches you a lot about abstraction

#### What you learn by building autograd

Automatic differentiation is the core of deep learning, and most practitioners treat it as a black box. Building it from scratch teaches you:

- **Computational graphs.** Every operation in the forward pass creates a node in a directed acyclic graph. `loss.backward()` walks this graph in reverse topological order, applying the chain rule at each node
- **Gradient functions.** Every tensor operation (add, multiply, matmul, relu, etc.) has a corresponding gradient function. When you implement `MatMulBackward`, you're directly implementing the multivariate chain rule for matrix multiplication
- **Memory management.** The graph has to stay alive until backward is called, then be freed. In Python this happens automatically via reference counting. In Rust you have to think about it explicitly, which teaches you what's actually happening under the hood

#### What you learn by building neural network layers

Implementing `nn.Linear`, `nn.LSTM`, `nn.MultiHeadAttention`, etc. from scratch:

- **Linear is just matrix multiply + bias.** `output = input @ weight.T + bias`. That's it. Once you've implemented this, you understand fully-connected layers forever
- **LSTM/GRU gates.** Implementing the gate equations (`forget_gate = sigmoid(W_f @ x + U_f @ h + b_f)`) makes you understand exactly how recurrent networks control information flow. It's not magic -- it's four matrix multiplications and some element-wise operations
- **Attention is just queries, keys, values.** `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V`. Implementing multi-head attention from raw tensor ops makes the Transformer architecture completely transparent
- **BatchNorm tracks running statistics.** It behaves differently in training vs. eval mode. This is the kind of detail that trips people up in practice, and implementing it makes the behavior obvious

#### What you learn by building optimizers

SGD, Adam, AdamW -- these are much simpler than they seem:

- **SGD** is literally `parameter -= learning_rate * gradient`. That's the whole algorithm. Momentum adds one exponential moving average
- **Adam** tracks two exponential moving averages (first and second moments of the gradient) and uses them to adapt the learning rate per-parameter. The entire algorithm is about 10 lines of math
- **Weight decay** is just `parameter -= wd * parameter` each step. AdamW decouples this from the gradient update, which matters for training stability

#### The project itself

AxonML has 22 crates covering tensors, autograd, neural networks, optimizers, data loading, vision (ResNet, VGG, ViT), audio (MFCC, spectrogram), NLP (BPE tokenizer), LLM architectures (BERT, GPT-2, LLaMA), CUDA GPU support, ONNX import/export, quantization, distributed training, and more.

It's running real production inference -- 69 trained models for HVAC predictive maintenance across 7 building facilities, with 12 models doing live inference on Raspberry Pi edge controllers. I also used it for a Kaggle competition doing Akkadian-to-English machine translation with a full seq2seq Transformer.

I'm not suggesting you should build your own ML framework (unless you want to). But the source code is MIT/Apache-2.0 and fully readable. If you're curious about how any of these components work internally, the implementations are all there.

Some starting points if you want to explore the code:
- `crates/axonml-tensor/` -- how tensors work
- `crates/axonml-autograd/` -- how backpropagation is implemented
- `crates/axonml-nn/` -- how layers are built from tensor operations
- `crates/axonml-optim/` -- how optimizers update weights
- `crates/axonml-llm/` -- how Transformer/BERT/GPT-2/LLaMA architectures are structured

Even if you don't know Rust, the logic is readable if you understand the math. Rust is closer to pseudocode than Python in many ways -- the types make the dimensions and data flow explicit.
