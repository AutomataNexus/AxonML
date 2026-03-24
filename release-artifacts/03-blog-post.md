# Building a PyTorch-Equivalent ML Framework in Rust

*Andrew Jewell Sr / AutomataNexus LLC*

---

Over the past year and a half, I've been building AxonML -- a machine learning framework in Rust that aims for feature parity with PyTorch. It's now at v0.3.2: 22 crates, 336 Rust source files, 1,095 passing tests, and it's running production inference on Raspberry Pi edge hardware in commercial buildings. This post covers why I built it, how it's architected, the hard technical problems I ran into, and where it's actually being used.

**GitHub:** [github.com/AutomataNexus/AxonML](https://github.com/AutomataNexus/AxonML)
**License:** MIT / Apache-2.0

## Motivation

I built an entire building automation ecosystem from scratch. NexusBMS is the central building management platform -- won an InfluxDB hackathon with it, runs InfluxDB 3.0 OSS alongside my own database (Aegis-DB, also open source). The edge controllers are 50+ Raspberry Pi 4/5s running my custom NexusEdge software: Rust hardware daemons for I2C, BACnet, and Modbus communications, direct HVAC equipment control via analog outputs, 24V triacs, 0-10V inputs, 10K/1K thermistor inputs, and dry contact inputs. Custom control logic per equipment type. 16+ facilities including Taylor University, Element Labs, Byrna Ammunition, St. Jude Catholic School, Heritage Point Retirement Facilities in two different cities. Over 120 pieces of equipment -- air handlers, boilers, cooling towers, pumps, DOAS units, natatorium pool units, exhaust fans, greenhouses.

The monitoring uses machine learning -- LSTM autoencoders for anomaly detection, GRU networks for failure prediction -- running on those Pi edge controllers mounted in mechanical rooms. Pi 5s have Hailo NPU chips running larger models; Pi 4s run smaller AxonML Rust inference models.

The original plan was to train models in PyTorch and deploy inference in Python on the Pis. This didn't work well. Python's memory footprint on a 1 GB RAM Pi was too high. Dependency management was fragile. PyTorch's ARM support was incomplete. And I was spending more time fighting the deployment pipeline than building models.

I wanted a framework where I could:
1. Define and train models with PyTorch-like ergonomics
2. Compile to a single static binary
3. Cross-compile to ARM
4. Run inference at 2-3 MB RSS with no runtime dependencies

Rust was the obvious choice. The question was whether one person could build enough of a framework to actually be useful.

The answer, it turns out, is yes -- with caveats I'll get into.

## Architecture: 22 Crates

AxonML is structured as a Cargo workspace with 22 crates, organized in layers. Each crate is independently testable and can be pulled in via feature flags.

### Layer 1: Compute Foundation

**`axonml-core`** provides device abstraction across CPU, CUDA, Vulkan, Metal, and WebGPU. The `Device` enum dispatches operations to the appropriate backend. `Storage<T>` is the reference-counted raw memory backing for tensors. The CUDA backend implements GPU memory allocation, cuBLAS GEMM for matrix multiply, and 20+ element-wise CUDA kernels compiled from PTX source.

**`axonml-tensor`** implements N-dimensional tensors generic over scalar type: `Tensor<T: Scalar>`. Broadcasting follows NumPy rules. Views and slicing are zero-copy where possible (backed by `Arc<Storage>`). 60+ operations including arithmetic, reductions (sum, mean, max, min, prod), sorting (sort, argsort, topk), indexing (gather, scatter, nonzero, unique), shape manipulation (flip, roll, squeeze, unsqueeze, permute), and activations (ReLU, Sigmoid, Tanh, Softmax, GELU, SiLU, ELU, LeakyReLU). Sparse tensor support in COO format.

**`axonml-autograd`** is the reverse-mode automatic differentiation engine. `Variable` wraps a tensor and connects it to the computational graph via gradient functions. The graph is tape-based with `Arc<Mutex<>>` for shared ownership. Backward pass performs topological sort over the graph and applies the chain rule. Includes Automatic Mixed Precision (autocast context for F16 training) and gradient checkpointing (trade compute for memory). Backward functions cover activations (ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, GELU), arithmetic (Add, Sub, Mul, Div, Neg, Pow, Sum, Mean), and linalg (MatMul, Transpose, Reshape, Cat, Select, Expand, SumDim).

### Layer 2: ML Primitives

**`axonml-nn`** provides the `Module` trait (with `forward()`, `parameters()`, `train()`/`eval()`) and 37+ layer types across 12 source files:

- **Core:** Linear, Conv1d, Conv2d, Embedding, ResidualBlock
- **Pooling:** MaxPool1d/2d, AvgPool1d/2d, AdaptiveAvgPool2d
- **Normalization:** BatchNorm1d/2d, LayerNorm, GroupNorm, InstanceNorm2d
- **Regularization:** Dropout, Dropout2d, AlphaDropout
- **Recurrent:** RNN, LSTM, GRU (each with cell variants: RNNCell, LSTMCell, GRUCell)
- **Attention:** MultiHeadAttention, CrossAttention
- **Transformer:** TransformerEncoderLayer, TransformerDecoderLayer, TransformerEncoder, TransformerDecoder, Seq2SeqTransformer
- **Graph neural networks:** GCNConv (Graph Convolutional Network), GATConv (Graph Attention Network)
- **Signal processing:** FFT1d, STFT (Short-Time Fourier Transform)

Loss functions: MSELoss, CrossEntropyLoss, BCELoss, BCEWithLogitsLoss, L1Loss, SmoothL1Loss, NLLLoss.

Initialization: Xavier/Glorot (uniform and normal), Kaiming/He (uniform and normal), Orthogonal, Sparse, plus uniform, normal, constant, zeros, ones, eye, diag.

Activations: ReLU, Sigmoid, Tanh, GELU, SiLU, ELU, LeakyReLU, Softmax, LogSoftmax, Identity.

**`axonml-optim`** implements five optimizers: SGD (with momentum, Nesterov, weight decay, dampening), Adam (with AMSGrad), AdamW (decoupled weight decay), RMSprop (centered, with momentum), and LAMB (layer-wise adaptive moments for large-batch training). GradScaler for mixed-precision gradient scaling. Seven learning rate schedulers: StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, OneCycleLR, WarmupLR, ReduceLROnPlateau.

**`axonml-data`** provides the `Dataset` trait, `DataLoader` with batching and shuffling, samplers (Sequential, Random, SubsetRandom, WeightedRandom, Batch), transforms (Normalize, RandomNoise, RandomCrop, RandomFlip, Scale, Clamp), and collate functions.

### Layer 3: Infrastructure

**`axonml-serialize`** handles model save/load in the `.axonml` binary format, JSON, and SafeTensors. StateDict extraction (PyTorch-compatible concept), checkpoint management with builder pattern, format auto-detection from file extensions and magic bytes, PyTorch key conversion utilities.

**`axonml-onnx`** imports and exports ONNX models with 40+ operator implementations at opset version 17.

**`axonml-quant`** provides block-based quantization (32-element blocks) in INT8 (Q8_0), INT4 (Q4_0, Q4_1), INT5 (Q5_0, Q5_1), and F16 with four calibration methods: MinMax, Percentile, Entropy, MeanStd. Q4 quantization achieves roughly 8x model size reduction. Parallel processing via Rayon. Compression stats with RMSE, max error, mean error.

**`axonml-fusion`** detects and applies kernel fusion patterns automatically: MatMul+Bias, MatMul+Bias+ReLU/GELU, Conv+BatchNorm, Conv+BatchNorm+ReLU, elementwise chains, Add+ReLU, Mul+Add FMA. FusedLinear and FusedElementwise (14 elementwise ops). Configurable optimizer with conservative/aggressive modes. Up to 2x speedup for memory-bound operations.

**`axonml-jit`** provides an intermediate representation for computation graphs (40+ ops), operation tracing, graph optimization passes (constant folding, dead code elimination, common subexpression elimination, algebraic simplification, elementwise fusion, strength reduction), LRU function caching, and shape inference with broadcasting. Built on a Cranelift foundation for native codegen.

**`axonml-profile`** includes memory profiling (allocation tracking, peak usage, leak detection), compute profiling (operation timing, FLOPS, throughput, bandwidth), timeline profiling with Chrome trace export (`chrome://tracing`), automatic bottleneck analysis (5 categories: SlowOperation, HighCallCount, MemoryHotspot, MemoryLeak, LowThroughput), and reports in Text/JSON/Markdown/HTML.

**`axonml-distributed`** implements four parallelism strategies: DistributedDataParallel (DDP) with gradient bucketing, Fully Sharded Data Parallel (FSDP) with ZeRO-2/ZeRO-3 and CPU offload, Pipeline Parallelism with microbatching and configurable schedules, and Tensor Parallelism via ColumnParallelLinear/RowParallelLinear. Collective operations: all-reduce (sum, mean, min, max, product), broadcast, all-gather, reduce-scatter, barrier, ring all-reduce, scatter, gather.

### Layer 4: Domain Libraries

**`axonml-vision`** provides image transforms (Resize, CenterCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter, Grayscale, Normalize, Pad), synthetic datasets (MNIST, CIFAR, Fashion-MNIST, CIFAR-100), and architecture implementations: LeNet, SimpleCNN, MLP, ResNet (18/34), VGG (11/13/16/19), Vision Transformer (ViT). Pretrained weight hub with local caching and ImageNet/MNIST/CIFAR normalization presets.

**`axonml-audio`** provides MelSpectrogram, MFCC, resampling, time stretching, pitch shifting, noise augmentation (SNR-based), audio normalization, and silence trimming. Synthetic datasets for command recognition, music genre classification, and speaker identification.

**`axonml-text`** provides six tokenizer types: Whitespace, Character, WordPunct, NGram (word and character n-grams), BPE (with training), and Unigram. Vocabulary management with special tokens (PAD, UNK, BOS, EOS, MASK) and frequency-based filtering. Synthetic datasets for sentiment analysis, seq2seq, and language modeling.

**`axonml-llm`** implements five LLM architectures:
- **BERT** -- encoder with BertForSequenceClassification, BertForMaskedLM, BertPooler, configs for base/large/tiny
- **GPT-2** -- decoder with GPT2LMHead, configs for small/medium/large/xl/tiny
- **LLaMA** -- LLaMAForCausalLM with RMSNorm, RotaryEmbedding, GroupedQueryAttention
- **Mistral** -- MistralForCausalLM with sliding window attention
- **Phi** -- PhiForCausalLM

Plus: FlashAttention with KVCache, MultiHeadSelfAttention, CausalSelfAttention. Text generation with greedy, top-k, top-p/nucleus, temperature, and beam search with repetition penalty. Hugging Face model loader with weight mappers for LLaMA, Mistral, and Phi. Pretrained model hub with configs for LLaMA, Mistral, Phi, and Qwen.

### Layer 5: Application Stack

**`axonml-cli`** is a comprehensive CLI with 50+ commands covering project scaffolding, training, evaluation, inference, model conversion, ONNX export/import, quantization, workspace management, analysis/reports, data management, bundling/deployment, benchmarking, GPU management, pretrained model hub, Kaggle integration, dataset management (NexusConnectBridge API), Weights & Biases experiment tracking, and dashboard/server lifecycle management.

**`axonml-tui`** is a ratatui-based terminal dashboard with model architecture visualization, dataset explorer, real-time training monitor with sparkline trends and ETA, interactive loss/accuracy/learning rate graphs, file browser with preview, and vi-style keyboard navigation.

**`axonml-dashboard`** is a Leptos/WASM web frontend (client-side rendered) with 20+ page routes: full authentication flow (login/register/session), MFA setup (TOTP + WebAuthn), training run management with real-time WebSocket metrics, model registry (browse/upload/version), inference endpoint deployment, dark mode toggle, toast notifications, slide-out terminal with WebSocket PTY, responsive design.

**`axonml-server`** is an Axum-based REST + WebSocket API backend with 50+ endpoints: JWT authentication with refresh tokens, multi-factor authentication (TOTP/RFC 6238 + WebAuthn/FIDO2 hardware keys + recovery codes), Argon2 password hashing, training run management with WebSocket streaming, model registry with file upload/download, inference endpoint management, terminal WebSocket PTY, CORS, structured tracing, and Prometheus metrics export. Uses Aegis-DB as its backing document store.

## The Hard Problems

### Autograd Without a Garbage Collector

This was the central technical challenge. PyTorch's computational graph is managed by Python's reference counting and garbage collector. When a tensor goes out of scope, its graph node is freed. When you call `loss.backward()`, Python's GC ensures the graph stays alive exactly as long as needed.

Rust doesn't have a GC. The ownership model is fundamentally different.

My approach: each `Variable` (the autograd-aware tensor wrapper) holds an `Arc` reference to its gradient function. Gradient functions hold `Arc` references to their input variables. This creates a reference-counted graph that stays alive as long as any variable referencing it is alive.

Backward traversal collects all reachable nodes, topologically sorts them, and applies gradient functions in reverse order. After backward, the graph can be dropped.

The `Arc<Mutex<>>` pattern adds allocation overhead on every forward operation. Each tensor op creates a new `Variable` with a new `Arc`-wrapped gradient function. For small models and short sequences this is negligible. For very large models with long computational graphs, it's measurable.

If I were starting over, I'd explore arena-based allocation -- allocate all graph nodes from a bump allocator that gets reset after each backward pass. This would trade some API complexity for better performance characteristics.

A critical bug I found and fixed: early versions used `Variable::new()` for intermediate results, which creates leaf variables that sever the gradient graph. The fix was `Variable::from_operation()`, which creates non-leaf variables that properly participate in backpropagation. This is the kind of bug that's obvious in hindsight but took significant debugging to identify (loss would decrease for a few epochs then plateau because gradients weren't flowing through certain layers).

### CUDA Integration

The CUDA backend was built incrementally. The core abstraction is `CudaStorage` -- GPU-resident memory managed via CUDA's `cudaMalloc`/`cudaFree`, with deallocation in `Drop` so GPU memory leaks are structurally impossible (barring panics).

Matrix multiplication dispatches to cuBLAS GEMM. Element-wise operations (add, subtract, multiply, divide, relu, sigmoid, tanh, exp, log, sqrt, abs, neg, clamp, pow, and more) use custom CUDA kernels compiled from PTX source.

The main challenge is keeping tensors on-device. In PyTorch, `.to('cuda')` moves a tensor to GPU and subsequent operations stay on GPU. In AxonML, the `Device` enum propagates through operations -- if both inputs are on CUDA, the output stays on CUDA. If there's a mismatch, you get an explicit error at compile time via the type system, rather than a runtime error.

### Generic Tensors vs. Dynamic Types

I chose `Tensor<T: Scalar>` -- tensors are generic over their scalar type. This means `Tensor<f32>` and `Tensor<f64>` are distinct types. You can't accidentally add a float tensor to an integer tensor. Dimension mismatches are caught at compile time.

The tradeoff: you can't dynamically switch dtypes without enum dispatch. PyTorch lets you call `.float()` or `.half()` and it returns the same type with different internal representation. In AxonML, changing dtype requires converting to a different concrete type. This adds some API friction but eliminates an entire class of runtime bugs.

For the LLM architectures, where mixed-precision training switches between f32 and f16, I implemented the AMP autocast context that handles the conversion explicitly.

### Cross-Compilation for ARM

The deployment target for my HVAC models is `armv7-unknown-linux-musleabihf` -- 32-bit ARM with hardware floating point, statically linked against musl libc. The build command:

```bash
cargo build --release --target armv7-unknown-linux-musleabihf
```

This produces a single binary with no dynamic library dependencies. Copy it to the Raspberry Pi, set it executable, and it runs. No cross-compilation toolchain complexity beyond having the right Rust target installed.

The inference binaries use pure tensor operations -- no autograd tape, no gradient tracking, no optimizer state. This keeps the binary small and the runtime footprint minimal. Each inference daemon runs at ~2-3 MB RSS.

## Production: HVAC Predictive Maintenance

This is where AxonML proves itself beyond benchmarks and test suites.

I have 69 trained `.axonml` model files across 7 commercial building facilities: FCOG, Warren, Huntington, Akron, Hopebridge, NE Realty, and a unified NexusBMS system. The models cover a wide range of HVAC equipment:

- **Air handlers** -- supply air temperature prediction, mixed air anomalies
- **Boilers** -- steam/comfort/domestic hot water anomaly detection
- **Chillers** -- condenser/evaporator anomaly patterns
- **VAV boxes** -- zone temperature and airflow prediction
- **Fan coils** -- heating/cooling valve anomalies
- **Make-up air units** -- outside air conditioning monitoring
- **DOAS** (Dedicated Outdoor Air Systems) -- ventilation anomalies
- **Pumps** -- flow and pressure anomaly detection
- **Steam systems** -- bundle condition and trap monitoring

The model architectures are:
- **Anomaly detectors:** LSTM autoencoders that learn normal operating patterns and flag deviations. An input sequence of sensor readings goes through an LSTM encoder, gets compressed to a latent representation, then reconstructed by an LSTM decoder. Reconstruction error above a threshold signals anomalous behavior
- **Failure predictors:** GRU networks that take recent sensor history and predict probability of equipment failure in the near future

12 of these models are running live inference on Raspberry Pi edge controllers. The deployment pipeline:

1. Train on server (CPU) using AxonML
2. Save model weights as `.axonml` files (or quantize to INT4/INT8 for smaller footprint)
3. Cross-compile inference daemon to ARM static binary
4. Deploy to Pi via the building management system's OTA update pipeline
5. PM2 manages the process (auto-restart, log management)
6. Daemon polls local NexusEdge controller for sensor data at 1 Hz
7. Runs inference, maintains rolling time-series buffers
8. Exposes anomaly scores and failure predictions via REST API (`/api/inference/latest`)
9. NexusBMS building management dashboard consumes the API

The NexusBMS system alone has 22 trained models covering every major equipment type in a commercial building. Each model trains in minutes on CPU, serializes to a few hundred KB (or less with quantization), and runs inference in microseconds.

## Kaggle: Akkadian-to-English Machine Translation

To exercise the seq2seq and NLP capabilities, I entered the Deep Past Initiative Kaggle competition. The task: translate Akkadian cuneiform text to English. The dataset has ~1,561 parallel sentence pairs with 5,571 unique source tokens.

The AxonML model:
- BPE tokenizer for both source and target languages
- Sinusoidal positional encoding
- Transformer encoder-decoder with multi-head attention
- Trained end-to-end through AxonML's training pipeline
- Evaluated on BLEU + chrF++

The entire pipeline -- data loading, tokenization, vocabulary building, model definition, training loop, checkpoint management, generation with beam search -- runs through AxonML. No Python anywhere in the pipeline.

This was a good stress test for the framework's NLP capabilities. seq2seq translation exercises: embedding layers, positional encoding, encoder with self-attention, decoder with masked self-attention and cross-attention, output projection, autoregressive generation.

## What's Next

- **Real-time model serving with batched inference.** The inference server works but doesn't batch across concurrent requests yet
- **Expanded CUDA kernel coverage.** More operations need GPU implementations to reduce CPU fallbacks
- **Self-hosted pretrained weight hosting.** Currently using a hub config system; want to host weights directly
- **More pretrained weights and ONNX import improvements.** Making it easier to convert models from Hugging Face

## Should You Use AxonML?

If you're doing standard ML research with Jupyter notebooks, Hugging Face, and cloud GPUs: probably not. PyTorch's ecosystem is vast and mature, and fighting a smaller ecosystem isn't worth it for research iteration speed.

If you're in one of these situations, it might be worth evaluating:

- **Edge deployment.** You need ML inference on constrained hardware without Python
- **Rust applications.** You're building a Rust application that needs embedded ML inference
- **Single-binary deployment.** You want a model that compiles to one file with no dependencies
- **Graph neural networks in Rust.** GCNConv and GATConv layers are implemented and ready to use
- **Learning ML internals.** The codebase is MIT/Apache-2.0 and every layer is implemented from scratch in readable Rust. If you want to understand how autograd, attention, LSTM gates, or graph convolutions actually work, the source is there
- **HVAC/IoT/industrial.** You're in a similar domain where models need to run on real hardware in real buildings

AxonML is one developer's work. It's not going to outpace PyTorch's development velocity. But it solves a real problem -- production ML on constrained hardware with compile-time safety -- and it's been doing that in production for real buildings with real equipment.

**GitHub:** [github.com/AutomataNexus/AxonML](https://github.com/AutomataNexus/AxonML)

---

*Andrew Jewell Sr is the founder of AutomataNexus LLC. AxonML is open source under MIT/Apache-2.0 dual license.*

---

## Publishing Notes

**Target platforms:** dev.to, Medium, personal blog

**Tags:** #rust #machinelearning #opensource #deeplearning #edgecomputing

**Suggested title image:** Terminal screenshot showing AxonML training output, or a photo of a Raspberry Pi running inference in a mechanical room
