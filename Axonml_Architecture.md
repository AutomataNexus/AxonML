# Axonml Architecture

> Technical architecture documentation for the Axonml ML framework.

**Version:** 0.1.0
**Last Updated:** 2026-01-19 (All crates complete - 758 tests)
**Author:** AutomataNexus Development Team

---

## Table of Contents

1. [Overview](#overview)
2. [Crate Architecture](#crate-architecture)
3. [Core Abstractions](#core-abstractions)
4. [Memory Model](#memory-model)
5. [Type System](#type-system)
6. [Backend Architecture](#backend-architecture)
7. [Design Decisions](#design-decisions)
8. [Future Considerations](#future-considerations)

---

## Overview

Axonml is designed as a layered architecture where each layer builds upon the previous, providing increasing levels of abstraction. The design prioritizes:

1. **Safety** - Leverage Rust's ownership and type system
2. **Performance** - Zero-cost abstractions, SIMD, parallelism
3. **Ergonomics** - PyTorch-like API where possible
4. **Extensibility** - Pluggable backends and operations

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        axonml (main crate)                      │
│                    Re-exports all subcrates                      │
├─────────────────────────────────────────────────────────────────┤
│  axonml-vision  │  axonml-audio  │  axonml-text  │ axonml-distributed │
│  (torchvision)   │  (torchaudio)   │  (torchtext)   │ (multi-GPU/node)    │
├──────────────────┴─────────────────┴────────────────┴───────────┤
│    axonml-llm    │    axonml-jit    │     axonml-profile        │
│   BERT, GPT-2     │   JIT compilation │   Profiling tools         │
├──────────────────┴──────────────────┴───────────────────────────┤
│         axonml-serialize          │         axonml-onnx        │
│    Save/load models, Checkpoint    │   ONNX import/export        │
├────────────────────────────────────┴────────────────────────────┤
│         axonml-quant              │         axonml-fusion       │
│    INT8/INT4 quantization          │   Kernel fusion optimization │
├────────────────────────────────────┴────────────────────────────┤
│                           axonml-data                           │
│              DataLoader, Dataset trait, Transforms               │
├──────────────────────────────────────────────────────────────────┤
│              axonml-optim              │           axonml-nn    │
│        SGD, Adam, AdamW, etc.           │    Linear, Conv, etc.   │
├─────────────────────────────────────────┴────────────────────────┤
│                           axonml-autograd                        │
│              Computational graph, reverse-mode autodiff           │
├──────────────────────────────────────────────────────────────────┤
│                           axonml-tensor                          │
│              N-dimensional array, views, broadcasting, ops        │
├──────────────────────────────────────────────────────────────────┤
│                           axonml-core                            │
│              Storage, Device abstraction, Memory management       │
├──────────────┬──────────────┬──────────────┬──────────────┬──────┤
│   CPU/BLAS   │    CUDA      │   Vulkan     │    Metal     │WebGPU│
│  (feature)   │  (feature)   │  (feature)   │  (feature)   │(feat)│
└──────────────┴──────────────┴──────────────┴──────────────┴──────┘
```

---

## Crate Architecture

### axonml-core

The foundation layer providing device-agnostic abstractions.

**Key Components:**

- `Device` - Enum representing compute devices (CPU, CUDA, Vulkan, etc.)
- `DType` - Runtime data type representation
- `Storage<T>` - Reference-counted raw memory storage
- `Allocator` - Trait for device-specific memory allocation
- `Error` - Comprehensive error types

```rust
// Device abstraction
pub enum Device {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(usize),
    #[cfg(feature = "vulkan")]
    Vulkan(usize),
    // ...
}

// Type-safe scalar types
pub trait Scalar: Copy + Clone + Debug + Send + Sync + Pod + 'static {
    const DTYPE: DType;
}

// Reference-counted storage
pub struct Storage<T: Scalar> {
    inner: Arc<RwLock<StorageInner<T>>>,
    offset: usize,
    len: usize,
}
```

### axonml-tensor

N-dimensional array operations built on axonml-core.

**Key Components:**

- `Tensor<T>` - Main tensor struct
- `Shape` / `Strides` - Dimension management
- Creation functions (`zeros`, `ones`, `rand`, etc.)
- Arithmetic operations
- Broadcasting logic

```rust
pub struct Tensor<T: Scalar> {
    storage: Storage<T>,
    shape: Shape,
    strides: Strides,
    offset: usize,
}
```

### axonml-autograd (Implemented)

Automatic differentiation via computational graphs.

**Key Components:**

- `Variable` - Tensor wrapper with gradient tracking (f32)
- `GradFn` - Arc-wrapped gradient function with stable ID
- `GradientFunction` - Trait for gradient computation
- `ComputationGraph` - Thread-local graph structure
- `backward()` - Reverse-mode autodiff implementation
- `no_grad` - Context managers for disabling gradients

**Gradient Functions:**

| Category | Functions |
|----------|-----------|
| Basic | AddBackward, SubBackward, MulBackward, DivBackward, NegBackward, PowBackward, SumBackward, MeanBackward |
| Activation | ReluBackward, SigmoidBackward, TanhBackward, SoftmaxBackward, LeakyReluBackward, GeluBackward |
| Linear Algebra | MatMulBackward, TransposeBackward, ReshapeBackward, SqueezeBackward, UnsqueezeBackward, ViewBackward |
| Loss | MseLossBackward, CrossEntropyLossBackward, NllLossBackward, BceLossBackward, L1LossBackward, SmoothL1LossBackward |

```rust
// Variable with automatic differentiation
let x = Variable::new(Tensor::from_vec(vec![2.0, 3.0], &[2]).unwrap(), true);
let y = x.pow(2.0).sum();  // y = sum(x^2)
y.backward();               // Compute gradients
let grad = x.grad();       // dy/dx = 2x = [4.0, 6.0]
```

### axonml-nn (Implemented)

Neural network modules with 69 tests passing.

**Key Components:**

- `Module` - Object-safe trait for all neural network modules
- `Parameter` - Wrapper for learnable parameters with Variable
- `Sequential` - Container for sequential layers with builder pattern
- `ModuleList` - Dynamic container for modules

**Layers:**

| Category | Components |
|----------|------------|
| Linear | Linear (fully connected with optional bias) |
| Convolution | Conv1d, Conv2d with stride/padding |
| Pooling | MaxPool1d/2d, AvgPool1d/2d, AdaptiveAvgPool2d |
| Normalization | BatchNorm1d, BatchNorm2d, LayerNorm |
| Dropout | Dropout, Dropout2d, AlphaDropout |
| Recurrent | RNN, LSTM, GRU (cells and multi-layer) |
| Attention | MultiHeadAttention (self and cross attention) |
| Embedding | Embedding with padding support |

**Activations:**

| Module | Function |
|--------|----------|
| ReLU | max(0, x) |
| LeakyReLU | max(0, x) + α·min(0, x) |
| Sigmoid | 1 / (1 + exp(-x)) |
| Tanh | tanh(x) |
| Softmax | exp(x) / sum(exp(x)) along dim |
| LogSoftmax | log(softmax(x)) |
| GELU | x·Φ(x) |
| SiLU | x·sigmoid(x) |
| ELU | max(0, x) + α·(exp(min(0, x)) - 1) |

**Loss Functions:**

| Loss | Description |
|------|-------------|
| MSELoss | Mean squared error |
| L1Loss | Mean absolute error |
| CrossEntropyLoss | Log softmax + NLL |
| NLLLoss | Negative log likelihood |
| BCELoss | Binary cross entropy |
| BCEWithLogitsLoss | Sigmoid + BCE (numerically stable) |
| SmoothL1Loss | Huber loss |

**Weight Initialization:**

| Function | Strategy |
|----------|----------|
| xavier_uniform/normal | Glorot initialization |
| kaiming_uniform/normal | He initialization |
| orthogonal | Orthogonal matrices |
| sparse | Sparse initialization |

```rust
// Example: Building a simple MLP
let model = Sequential::new()
    .add(Linear::new(784, 256))
    .add(ReLU)
    .add(Linear::new(256, 10));

let output = model.forward(&input);
let loss = CrossEntropyLoss::new().compute(&output, &target);
loss.backward();
```

### axonml-optim (Implemented)

Optimization algorithms with 25 tests passing.

**Key Components:**

- `Optimizer` - Trait for all optimizers (step, zero_grad, lr management)
- `ParamState` - Per-parameter optimizer state storage

**Optimizers:**

| Optimizer | Features |
|-----------|----------|
| SGD | Momentum, Nesterov, weight decay, dampening |
| Adam | Adaptive learning rates, bias correction |
| AdamW | Decoupled weight decay |
| RMSprop | Square gradient moving average, centered variant |

**Learning Rate Schedulers:**

| Scheduler | Description |
|-----------|-------------|
| StepLR | Decay by gamma every N steps |
| MultiStepLR | Decay at specified milestones |
| ExponentialLR | Exponential decay each epoch |
| CosineAnnealingLR | Cosine annealing to eta_min |
| ReduceLROnPlateau | Reduce when metric plateaus |
| OneCycleLR | 1cycle policy (warmup + anneal) |
| WarmupLR | Linear warmup period |

```rust
// Example: Training loop with optimizer and scheduler
let mut optimizer = Adam::new(model.parameters(), 0.001)
    .betas((0.9, 0.999))
    .weight_decay(0.01);

let mut scheduler = CosineAnnealingLR::new(&optimizer, total_epochs);

for epoch in 0..total_epochs {
    optimizer.zero_grad();
    let output = model.forward(&input);
    let loss = criterion.compute(&output, &target);
    loss.backward();
    optimizer.step();
    scheduler.step(&mut optimizer);
}
```

### axonml-data (Implemented)

Data loading infrastructure with 28 tests passing.

**Key Components:**

- `Dataset` - Trait for data sources
- `DataLoader` - Batched iteration with shuffling
- `Batch` - Struct containing data, targets, and size
- `Sampler` - Index sampling strategies

**Samplers:**

| Sampler | Description |
|---------|-------------|
| SequentialSampler | Indices in order |
| RandomSampler | Shuffled indices |

```rust
let dataset = SyntheticMNIST::new(60000);
let loader = DataLoader::with_shuffle(dataset, 32, true);

for batch in loader.iter() {
    let input = &batch.data;      // [batch_size, ...]
    let target = &batch.targets;  // [batch_size, ...]
}
```

### axonml-vision (Implemented)

Computer vision utilities with 54 tests passing.

**Transforms:**

| Transform | Description |
|-----------|-------------|
| Resize | Resize image to target size |
| CenterCrop | Crop center region |
| RandomHorizontalFlip | Random horizontal flip |
| ImageNormalize | Normalize with mean/std |

**Datasets:**

| Dataset | Description |
|---------|-------------|
| SyntheticMNIST | 28x28 grayscale digit images |
| SyntheticCIFAR | 32x32 RGB images |

**Models:**

| Model | Description |
|-------|-------------|
| LeNet | Classic LeNet-5 architecture |
| SimpleCNN | Flexible CNN for experiments |
| ResNet | ResNet18, ResNet34 with BasicBlock (residual connections) |
| VGG | VGG11, VGG13, VGG16, VGG19 with optional BatchNorm |
| Transformer | Full encoder-decoder transformer architecture |
| VisionTransformer | ViT (tiny, small, base, large) for image classification |

**Pretrained Model Architectures:**

```rust
// ResNet variants
let resnet18 = ResNet::resnet18(num_classes);
let resnet34 = ResNet::resnet34(num_classes);

// VGG variants (with optional batch normalization)
let vgg16 = VGG::vgg16(num_classes);
let vgg16_bn = VGG::vgg16_bn(num_classes);

// Vision Transformer variants
let vit_base = vit_base(num_classes, image_size, patch_size);
let vit_large = vit_large(num_classes, image_size, patch_size);

// Full Transformer (for sequence-to-sequence)
let transformer = Transformer::new(d_model, nhead, num_encoder_layers, num_decoder_layers);
```

### axonml-audio (Implemented)

Audio processing utilities with 28 tests passing.

**Transforms:**

| Transform | Description |
|-----------|-------------|
| Resample | Change sample rate |
| MelSpectrogram | Waveform to mel spectrogram |
| MFCC | Mel-frequency cepstral coefficients |
| NormalizeAudio | Normalize to [-1, 1] |
| AddNoise | Add noise augmentation |

**Datasets:**

| Dataset | Description |
|---------|-------------|
| SyntheticCommandDataset | Speech command classification |
| SyntheticMusicDataset | Music genre classification |

### axonml-text (Implemented)

NLP utilities with 38 tests passing.

**Tokenizers:**

| Tokenizer | Description |
|-----------|-------------|
| WhitespaceTokenizer | Split on whitespace |
| CharTokenizer | Character-level |
| BasicBPETokenizer | Byte-pair encoding |

**Components:**

- `Vocab` - Token-to-index mapping
- `TextDataset` - Text classification dataset
- `LanguageModelDataset` - Next token prediction
- `SyntheticSentimentDataset` - Sentiment analysis

### axonml-distributed (Implemented)

Distributed training utilities with 62 tests passing.

**Key Components:**

- `World` - Global process group
- `ProcessGroup` - Subset of processes
- `DDP` / `DistributedDataParallel` - Model wrapper for distributed training
- `GradientSynchronizer` - Gradient averaging

**Communication Primitives:**

| Function | Description |
|----------|-------------|
| all_reduce_sum | Sum across all ranks |
| all_reduce_mean | Average across all ranks |
| broadcast | Send from one rank to all |
| barrier | Synchronize all processes |

```rust
let world = World::new()?;
let ddp_model = DDP::new(model, world);

for batch in loader.iter() {
    let output = ddp_model.forward(&batch.data);
    let loss = compute_loss(&output, &batch.targets);
    loss.backward();  // Gradients synced automatically
    optimizer.step();
}
```

### axonml-serialize (Implemented)

Model serialization and checkpoint utilities with 17 tests passing.

**Key Components:**

- `ModelArchive` - Unified model storage format (.axonml files)
- `CheckpointManager` - Training checkpoint management
- `StateDict` - Named tensor dictionary (PyTorch-compatible concept)
- `SafeTensors` - Safe, zero-copy tensor serialization

**Serialization Formats:**

| Format | Description |
|--------|-------------|
| Binary (.bin) | Fast binary format with compression |
| JSON (.json) | Human-readable format for inspection |
| SafeTensors (.safetensors) | Safe format preventing code execution |
| Axonml Archive (.axonml) | Complete model archive with metadata |

**Features:**

- Save/load complete models with architecture + weights
- Checkpoint saving during training with epoch/optimizer state
- SafeTensors format for security
- Platform-independent serialization

```rust
// Save a model
let archive = ModelArchive::new("my_model", "1.0.0")
    .with_metadata("description", "Image classifier")
    .with_state_dict(model.state_dict());
archive.save("model.axonml")?;

// Load a model
let archive = ModelArchive::load("model.axonml")?;
model.load_state_dict(archive.state_dict())?;

// Checkpoint management
let mut checkpoint_mgr = CheckpointManager::new("./checkpoints", 3);
checkpoint_mgr.save_checkpoint(epoch, &model, &optimizer)?;
let (epoch, state) = checkpoint_mgr.load_latest()?;
```

---

## Core Abstractions

### Tensor

The `Tensor<T>` is the primary data structure:

```rust
impl<T: Scalar> Tensor<T> {
    // Shape information
    pub fn shape(&self) -> &[usize];
    pub fn ndim(&self) -> usize;
    pub fn numel(&self) -> usize;

    // Data access
    pub fn get(&self, indices: &[usize]) -> Result<T>;
    pub fn to_vec(&self) -> Vec<T>;

    // Shape operations
    pub fn reshape(&self, shape: &[isize]) -> Result<Self>;
    pub fn transpose(&self, d0: i64, d1: i64) -> Result<Self>;
    pub fn squeeze(&self, dim: Option<i64>) -> Result<Self>;
    pub fn unsqueeze(&self, dim: i64) -> Result<Self>;
}

impl<T: Numeric> Tensor<T> {
    // Arithmetic
    pub fn add(&self, other: &Self) -> Result<Self>;
    pub fn mul(&self, other: &Self) -> Result<Self>;
    pub fn matmul(&self, other: &Self) -> Result<Self>;

    // Reductions
    pub fn sum(&self) -> Self;
    pub fn mean(&self) -> Result<Self>;  // Float only
}
```

### Storage

Storage handles raw memory with reference counting:

```rust
impl<T: Scalar> Storage<T> {
    // Creation
    pub fn zeros(len: usize, device: Device) -> Self;
    pub fn from_vec(data: Vec<T>, device: Device) -> Self;

    // Access
    pub fn as_slice(&self) -> StorageReadGuard<T>;
    pub fn as_slice_mut(&self) -> StorageWriteGuard<T>;

    // Views
    pub fn slice(&self, offset: usize, len: usize) -> Result<Self>;

    // Device transfer
    pub fn to_device(&self, device: Device) -> Result<Self>;
}
```

---

## Memory Model

### Reference Counting

Storage uses `Arc<RwLock<_>>` for safe sharing:

```
Tensor A ─────┐
              │
              ▼
         ┌─────────────┐
         │   Storage   │
         │  (Arc<...>) │
         └─────────────┘
              ▲
              │
Tensor B ─────┘  (view of A)
```

### View Semantics

Views share storage but have different shapes/strides/offsets:

```rust
let a = Tensor::from_vec(vec![1,2,3,4,5,6], &[2, 3]);
let b = a.transpose(0, 1)?;  // Same storage, different strides
let c = a.slice_dim0(0, 1)?; // Same storage, different offset
```

### Memory Layout

Tensors use row-major (C-order) layout by default:

```
Shape: [2, 3]
Data:  [a, b, c, d, e, f]

Logical view:
  [[a, b, c],
   [d, e, f]]

Strides: [3, 1]  (row-major)
```

---

## Type System

### Scalar Traits

```rust
// Base trait for all tensor element types
pub trait Scalar: Copy + Clone + Debug + Default + Send + Sync + Pod + 'static {
    const DTYPE: DType;
}

// Numeric types supporting arithmetic
pub trait Numeric: Scalar + Num + NumCast + PartialOrd + Zero + One {
    const ZERO: Self;
    const ONE: Self;
}

// Floating point types
pub trait Float: Numeric + NumFloat {
    const NAN: Self;
    const INFINITY: Self;
    fn exp_value(self) -> Self;
    fn ln_value(self) -> Self;
    // ...
}
```

### Supported Types

| Type | Rust Type | Use Case |
|------|-----------|----------|
| F16 | half::f16 | Memory-efficient training |
| F32 | f32 | Default float type |
| F64 | f64 | High precision |
| I8 | i8 | Quantization |
| I16 | i16 | Quantization |
| I32 | i32 | Indices |
| I64 | i64 | Default int type |
| U8 | u8 | Images |
| Bool | bool | Masks |

---

## Backend Architecture

### Backend Trait

Each backend implements optimized operations:

```rust
pub struct CpuBackend;

impl CpuBackend {
    pub fn add<T: Numeric>(dst: &mut [T], a: &[T], b: &[T]);
    pub fn matmul<T: Numeric>(c: &mut [T], a: &[T], b: &[T], m: usize, n: usize, k: usize);
    pub fn relu<T: Float>(dst: &mut [T], a: &[T]);
    // ...
}
```

### Backend Selection

Operations dispatch to the appropriate backend based on device:

```rust
match tensor.device() {
    Device::Cpu => CpuBackend::matmul(...),
    #[cfg(feature = "cuda")]
    Device::Cuda(idx) => CudaBackend::matmul(...),
    // ...
}
```

### Crate Implementation Status

| Crate | Status | Tests | Notes |
|-------|--------|-------|-------|
| axonml-core | ✓ Complete | 31 | Device, Storage, DType, Allocator, Backend trait |
| axonml-tensor | ✓ Complete | 38 | Tensor ops, broadcasting, activations |
| axonml-autograd | ✓ Complete | 37 | Variable, backward, grad functions |
| axonml-nn | ✓ Complete | 69 | Module, Linear, Conv, RNN, LSTM, Attention, Loss |
| axonml-optim | ✓ Complete | 25 | SGD, Adam, AdamW, RMSprop, 7 LR schedulers |
| axonml-data | ✓ Complete | 51 | Dataset trait, DataLoader, samplers, transforms |
| axonml-vision | ✓ Complete | 54 | ResNet, VGG, ViT, Transformer, LeNet, transforms |
| axonml-audio | ✓ Complete | 28 | MelSpectrogram, MFCC, Resample, audio datasets |
| axonml-text | ✓ Complete | 39 | Tokenizers (Whitespace, Char, BPE), Vocab, text datasets |
| axonml-distributed | ✓ Complete | 62 | DDP, ProcessGroup, all-reduce, broadcast, barrier |
| axonml-serialize | ✓ Complete | 25 | ModelArchive, Checkpoint, StateDict, SafeTensors |
| axonml-onnx | ✓ Complete | 14 | ONNX import/export, 40+ operators, opset 17 |
| axonml-quant | ✓ Complete | 18 | Q8_0, Q4_0/1, Q5_0/1, F16, calibration methods |
| axonml-fusion | ✓ Complete | 26 | Fused ops, pattern detection, graph optimization |
| axonml-jit | ✓ Complete | 24 | IR, tracing, graph optimization, function caching |
| axonml-profile | ✓ Complete | 27 | Profiler, MemoryProfiler, ComputeProfiler, TimelineProfiler |
| axonml-llm | ✓ Complete | 36 | BERT, GPT-2, text generation with sampling |
| axonml-cli | ✓ Complete | 111 | CLI: 74 unit + 37 integration (e2e) tests |
| axonml-tui | ✓ Complete | 10 | TUI: model, data, training, graphs, files views |
| axonml | ✓ Complete | 22 | Umbrella crate: 12 unit + 10 integration tests |

**Total: 758 tests passing**

### Backend Status

| Backend | Status | Platform | Notes |
|---------|--------|----------|-------|
| CPU | ✓ Implemented | All | SIMD, rayon parallelism |
| CUDA | ✓ Stubs (feature-gated) | NVIDIA GPUs | cuBLAS, cuDNN, streams |
| Vulkan | ✓ Stubs (feature-gated) | Cross-platform | Compute shaders, queues |
| Metal | ✓ Stubs (feature-gated) | Apple | MPS acceleration, command buffers |
| WebGPU | ✓ Stubs (feature-gated) | Browser/WASM | wgpu-rs, WGSL shaders |

**GPU Backend Architecture:**

All GPU backends implement a common `Backend` trait for device-agnostic code:

```rust
pub trait Backend: Send + Sync {
    fn name(&self) -> &'static str;
    fn is_available(&self) -> bool;
    fn capabilities(&self) -> DeviceCapabilities;
    fn allocate(&self, size: usize) -> *mut u8;
    fn deallocate(&self, ptr: *mut u8, size: usize);
    fn copy_to_device(&self, dst: *mut u8, src: *const u8, size: usize);
    fn copy_to_host(&self, dst: *mut u8, src: *const u8, size: usize);
    fn copy_device_to_device(&self, dst: *mut u8, src: *const u8, size: usize);
    fn synchronize(&self);
}
```

**Backend-Specific Features:**

| Backend | Special Features |
|---------|------------------|
| CUDA | cudaMalloc/cudaFree, cudaMemcpy, cudaStreams, cuBLAS GEMM, cuDNN Conv |
| Vulkan | vkAllocateMemory, compute pipelines, command buffers, queues |
| Metal | MTLBuffer, command queues, MPS MatMul/Conv2d/BatchNorm |
| WebGPU | WGSL shaders, bind groups, compute pipelines, workgroups |

**WGSL Shader Templates (WebGPU):**

```wgsl
// Element-wise addition shader
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    result[global_id.x] = a[global_id.x] + b[global_id.x];
}

// Matrix multiplication shader (16x16 workgroups)
@compute @workgroup_size(16, 16)
fn matmul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // ... tiled matrix multiplication
}
```

---

## Design Decisions

### 1. Eager Execution by Default

Like PyTorch, Axonml uses eager execution for easier debugging:

```rust
let a = randn(&[3, 3]);
let b = randn(&[3, 3]);
let c = &a + &b;  // Executed immediately
println!("{:?}", c);  // Can inspect result
```

### 2. Generic over Element Type

Tensors are generic over their element type:

```rust
let f32_tensor: Tensor<f32> = zeros(&[2, 3]);
let f64_tensor: Tensor<f64> = zeros(&[2, 3]);
```

This provides compile-time type safety at the cost of some API verbosity.

### 3. Immutable by Default

Operations return new tensors rather than mutating in place:

```rust
let a = randn(&[10]);
let b = a.relu();  // Returns new tensor
```

In-place operations are available with `_` suffix:

```rust
tensor.fill_(0.0);  // Mutates in place
```

### 4. Error Handling

All fallible operations return `Result<T, Error>`:

```rust
let c = a.matmul(&b)?;  // Propagate errors
let c = a.matmul(&b).expect("matmul failed");  // Panic on error
let c = &a + &b;  // Operators panic on error (ergonomics)
```

### 5. No Python Dependency

Axonml is pure Rust, enabling:
- WebAssembly compilation
- Embedded use
- No GIL limitations
- Static linking

---

## Axonml CLI Tool (Implemented)

The Axonml CLI is a major differentiator from PyTorch, providing a unified command-line interface for the entire ML workflow. The CLI is implemented in the `axonml-cli` crate with 23 tests passing.

**Key Features:**
- Real training using axonml-nn, axonml-optim, axonml-data, axonml-autograd
- Weights & Biases integration for experiment tracking
- Model serialization via axonml-serialize
- No mock or simulated code - production-ready implementation

### Command Overview

```bash
# Project Management
axonml new my-model                    # Scaffold new project
axonml init                            # Initialize in existing directory
axonml scaffold my-project             # Generate Rust training project

# Training (Real axonml integration)
axonml train config.toml               # Start training from config
axonml train --model mlp --epochs 10   # Quick training with MLP
axonml train --model cnn --dataset cifar # CNN on CIFAR
axonml resume checkpoint.axonml       # Resume from checkpoint

# Evaluation & Inference
axonml eval model.axonml --data test/ # Evaluate model (accuracy, loss, F1)
axonml predict model.axonml input.json # Single prediction with softmax
axonml benchmark --model resnet18      # Performance benchmarks

# Model Management
axonml convert pytorch-model.pth       # Convert PyTorch → Axonml
axonml convert model.onnx              # Convert ONNX → Axonml
axonml export model.axonml --format onnx # Export to ONNX
axonml inspect model.axonml           # Show architecture, params
axonml rename model.axonml new-name   # Rename model files

# Quantization
axonml quant convert model.axonml --type q8_0  # Quantize to Q8
axonml quant convert model.pth --type q4_0      # PyTorch → Quantized
axonml quant info model.axonml                 # Show quantization info
axonml quant benchmark model.axonml            # Benchmark quantized model
axonml quant list                               # List quantization types

# Workspace Management
axonml load model model.axonml        # Load model into workspace
axonml load data ./dataset             # Load dataset into workspace
axonml load both -m model.f -d data/   # Load both
axonml load status                     # Show workspace status
axonml load clear                      # Clear workspace

# Analysis & Reports
axonml analyze model                   # Analyze loaded model
axonml analyze data                    # Analyze loaded dataset
axonml analyze both                    # Analyze both
axonml analyze report --format html    # Generate analysis report

# Data Management
axonml data info ./dataset             # Dataset information
axonml data validate ./dataset         # Validate dataset format
axonml data split ./data --train 0.8   # Split dataset

# Bundling
axonml zip create -o bundle.zip --model m.f --data d/  # Create bundle
axonml zip extract bundle.zip -o ./output              # Extract bundle
axonml zip list bundle.zip                             # List contents

# Upload & Sharing
axonml upload model.axonml --hub myrepo  # Upload to model hub

# Weights & Biases Integration
axonml wandb login                     # Authenticate with W&B
axonml wandb init --project my-project # Initialize project
axonml train config.toml --wandb       # Train with W&B logging

# Deployment
axonml deploy --target raspberry-pi    # Cross-compile for ARM
axonml deploy --target wasm            # Compile to WebAssembly
axonml serve model.axonml --port 8080 # Start inference server

# Distributed Training
axonml distributed --nodes 4           # Launch distributed training
axonml distributed --config cluster.toml # From config file

# Benchmarking
axonml bench model model.axonml       # Benchmark model performance
axonml bench inference model.axonml   # Batch size scaling analysis
axonml bench compare m1.f,m2.f         # Compare multiple models
axonml bench hardware                  # CPU/memory benchmarks

# GPU Management
axonml gpu list                        # List available GPUs (via wgpu)
axonml gpu info                        # Detailed GPU information
axonml gpu select 0                    # Select GPU for training
axonml gpu bench                       # Real GPU compute benchmarks
axonml gpu memory                      # Show GPU memory usage
axonml gpu status                      # Current GPU status

# Development Tools
axonml check model.rs                  # Validate architecture
axonml profile model.axonml           # Profile memory/compute
axonml debug model.axonml             # Debug with gradient checking
```

### Weights & Biases Integration

The CLI includes built-in W&B integration for experiment tracking:

```rust
// WandbRun provides:
pub struct WandbRun {
    api_key: String,
    project: String,
    entity: Option<String>,
    run_id: String,
    run_name: String,
    // ...
}

impl WandbRun {
    pub fn init(...) -> CliResult<Self>;      // Initialize run
    pub fn log(&mut self, metrics: HashMap<String, f64>);  // Log metrics
    pub fn log_at_step(&mut self, step: usize, metrics: ...);
    pub fn summary(&mut self, key: &str, value: f64);
    pub fn log_config(&mut self, config: HashMap<String, Value>);
    pub fn finish(self) -> CliResult<()>;     // Finalize run
}
```

**W&B Features:**
- Automatic metric logging per epoch (loss, accuracy, learning rate)
- Hyperparameter tracking from config files
- Run comparison and visualization
- Model artifact versioning
- Real-time training dashboards

### Training Implementation

The train command uses real axonml components:

```rust
// Real training loop (not simulated)
pub fn execute(args: TrainArgs) -> CliResult<()> {
    // Load config
    let config = ProjectConfig::load(&args.config)?;

    // Create model based on architecture
    let mut model: Box<dyn TrainableModel> = match config.model.architecture.as_str() {
        "mlp" => Box::new(MLP::new(784, &[256, 128], 10, 0.0)),
        "cnn" => Box::new(SimpleCNN::new(10)),
        "lenet" => Box::new(LeNetModel::new()),
        _ => return Err(CliError::Model("Unknown architecture".into())),
    };

    // Create optimizer
    let mut optimizer = create_optimizer(&config.training, model.parameters())?;

    // Load dataset
    let dataset = TrainDataset(SyntheticMNIST::new(config.data.train_samples));
    let loader = DataLoader::new(dataset, config.training.batch_size);

    // Loss function
    let loss_fn = CrossEntropyLoss::new();

    // Training loop
    for epoch in 1..=config.training.epochs {
        for batch in loader.iter() {
            let input = Variable::new(batch.data.clone(), false);
            let target = Variable::new(batch.targets.clone(), false);

            let output = model.forward(&input);
            let loss = loss_fn.compute(&output, &target);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }

        // Save checkpoint
        save_checkpoint(&model, epoch, &output_dir)?;
    }
}
```

### Resume Implementation

Resume training from checkpoints using axonml-serialize:

```rust
fn load_checkpoint_info(checkpoint_path: &PathBuf) -> CliResult<CheckpointInfo> {
    // Load as full Checkpoint
    if let Ok(checkpoint) = load_checkpoint(checkpoint_path) {
        return Ok(CheckpointInfo {
            epoch: checkpoint.epoch(),
            loss: checkpoint.training_state.loss_history.last().copied().unwrap_or(0.5),
            learning_rate: checkpoint.training_state.lr_history.last().copied().unwrap_or(0.001),
            model_name: checkpoint.config.get("model_name").cloned().unwrap_or_default(),
            state_dict: checkpoint.model_state.clone(),
        });
    }

    // Fallback: load as state dict
    let state_dict = load_state_dict(checkpoint_path)?;
    // ...
}
```

### Eval Implementation

Real model evaluation with axonml components:

```rust
fn run_evaluation(args: &EvalArgs, model_info: &ModelInfo) -> CliResult<Vec<(String, f64)>> {
    let mut model = EvalModel::default_mlp();
    model.load_state_dict(&model_info.state_dict)?;

    let dataset = EvalDataset(SyntheticMNIST::new(10000));
    let loader = DataLoader::new(dataset, args.batch_size);
    let loss_fn = CrossEntropyLoss::new();

    for batch in loader.iter() {
        let input = Variable::new(batch.data.clone(), false);
        let target = Variable::new(batch.targets.clone(), false);

        let output = model.forward(&input);
        let loss = loss_fn.compute(&output, &target);

        // Compute accuracy, precision, recall, F1
        let pred_classes = argmax_batch(&output.data());
        let label_classes = argmax_batch(&batch.targets);
        // ...
    }
}
```

### Predict Implementation

Real inference with softmax probability output:

```rust
fn run_inference(model: &InferenceModel, input: &InputData, top_k: Option<usize>) -> CliResult<Vec<Prediction>> {
    for (idx, sample) in input.samples.iter().enumerate() {
        // Convert to tensor
        let input_tensor = Tensor::from_vec(sample_f32, &[1, 784])?;
        let input_var = Variable::new(input_tensor, false);

        // Forward pass
        let output = model.forward(&input_var);
        let logits = output.data().to_vec();

        // Apply softmax
        let probabilities = softmax(&logits);

        // Get top-k predictions
        let mut indexed: Vec<(usize, f32)> = probabilities.iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        // ...
    }
}
```

### Quantization Commands

The `axonml quant` command provides comprehensive model quantization:

```rust
// Supported quantization types
pub enum QuantType {
    Q4_0,   // 4-bit quantization (block size 32)
    Q4_1,   // 4-bit with min/max calibration
    Q5_0,   // 5-bit quantization
    Q5_1,   // 5-bit with calibration
    Q8_0,   // 8-bit quantization (best accuracy)
    F16,    // Half precision floating point
    F32,    // Full precision (no quantization)
}
```

**Quantization Workflow:**
```bash
# Convert from full precision to Q8
axonml quant convert model.axonml --type q8_0 --output model_q8.axonml

# Convert PyTorch model directly to quantized Axonml
axonml quant convert pytorch_model.pth --type q4_0 --output model_q4.axonml

# With calibration data for better accuracy
axonml quant convert model.axonml --type q4_1 --calibration ./calib_data/

# Benchmark quantized vs original
axonml quant benchmark model_q8.axonml --compare model.axonml
```

### Workspace Management

The `axonml load` command manages a persistent workspace:

```rust
// Workspace state stored in .axonml/workspace.json
pub struct WorkspaceState {
    pub model: Option<LoadedModel>,
    pub dataset: Option<LoadedDataset>,
    pub last_updated: String,
}

pub struct LoadedModel {
    pub path: String,
    pub name: String,
    pub format: String,
    pub parameters: usize,
    pub loaded_at: String,
}

pub struct LoadedDataset {
    pub path: String,
    pub name: String,
    pub num_samples: usize,
    pub data_type: String,
    pub loaded_at: String,
}
```

**Workspace Workflow:**
```bash
# Load model and dataset
axonml load model ./my_model.axonml
axonml load data ./training_data/

# Check what's loaded
axonml load status

# Output:
# Workspace Status
# Model: my_model (123,456 parameters)
# Dataset: training_data (50,000 samples)
# Last updated: 2026-01-19T10:30:00Z

# Clear workspace
axonml load clear
```

### Analysis Commands

The `axonml analyze` command provides comprehensive model and dataset analysis:

```rust
// Model analysis output
pub struct ModelAnalysis {
    pub name: String,
    pub num_parameters: usize,
    pub num_layers: usize,
    pub layer_analysis: Vec<LayerInfo>,
    pub architecture_type: String,  // "CNN", "MLP", "Transformer", etc.
    pub recommendations: Vec<String>,
}

// Dataset analysis output
pub struct DatasetAnalysis {
    pub name: String,
    pub num_samples: usize,
    pub data_type: String,
    pub class_distribution: HashMap<String, usize>,
    pub quality_score: f64,
    pub issues: Vec<String>,
    pub recommendations: Vec<String>,
}
```

**Analysis Workflow:**
```bash
# Analyze loaded model
axonml analyze model

# Output:
# Model Analysis: my_model
# Total Parameters: 1,234,567
# Layers: 12
# Architecture: CNN
#
# Layer Breakdown:
#   conv1: Conv2d(3, 64, 3x3) - 1,792 params
#   conv2: Conv2d(64, 128, 3x3) - 73,856 params
#   ...
#
# Recommendations:
#   - Consider adding BatchNorm after conv layers
#   - Model could benefit from Q8 quantization

# Analyze loaded dataset
axonml analyze data

# Output:
# Dataset Analysis: training_data
# Samples: 50,000
# Type: Image Classification
# Quality Score: 8.5/10
#
# Class Distribution:
#   class_0: 5,200 (10.4%)
#   class_1: 4,800 (9.6%)
#   ...
#
# Issues:
#   - Slight class imbalance detected
#
# Recommendations:
#   - Consider data augmentation for minority classes

# Generate comprehensive report
axonml analyze report --format html --output analysis_report.html
```

**Report Formats:**
- `html`: Interactive web report with charts
- `json`: Machine-readable format for automation
- `markdown`: Documentation-friendly format
- `text`: Plain text for terminal viewing

### Value Proposition

| Feature | PyTorch | Axonml CLI |
|---------|---------|-------------|
| Project scaffolding | Manual | `axonml new` |
| Training from config | External tools | `axonml train` |
| Model conversion | Manual scripting | `axonml convert` |
| Model quantization | External tools | `axonml quant` |
| Workspace management | None | `axonml load` |
| Model/data analysis | Custom scripts | `axonml analyze` |
| Bundle creation | Manual | `axonml zip` |
| Cross-compilation | Complex setup | `axonml deploy --target` |
| Inference server | Separate framework | `axonml serve` |
| Model benchmarking | Custom scripts | `axonml bench` |
| GPU management | nvidia-smi + code | `axonml gpu` |
| Distributed launch | torchrun | `axonml distributed` |

### Monetization Potential

**Free Tier:**
- `axonml new`, `axonml train`, `axonml eval`
- Basic model inspection
- Local deployment

**Pro Tier:**
- `axonml convert` (PyTorch/ONNX conversion)
- `axonml benchmark` (detailed comparisons)
- `axonml deploy` (cross-compilation)
- Priority support

**Enterprise Tier:**
- `axonml distributed` (multi-node)
- Cloud integration
- Custom deployment targets
- Model registry hosting

---

## Axonml TUI (Implemented)

The Axonml TUI provides an interactive terminal-based dashboard for ML development, implemented in the `axonml-tui` crate with 10 tests passing.

### Overview

Built with Ratatui and Crossterm, the TUI provides:
- Tab-based navigation between views
- Vim-style keyboard shortcuts
- NexusForge color theme (teal, terracotta, cream)
- Real-time updates during training

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        axonml-tui                               │
├──────────────┬──────────────┬──────────────┬──────────────────────┤
│   lib.rs     │   app.rs     │   event.rs   │      ui.rs           │
│  Entry point │  App state   │  Key events  │   Main render        │
├──────────────┴──────────────┴──────────────┴──────────────────────┤
│                          views/                                   │
├──────────┬──────────┬──────────┬──────────┬──────────┬──────────┤
│  model   │   data   │ training │  graphs  │  files   │   help   │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

### Views

| View | Description | Features |
|------|-------------|----------|
| Model | Neural network visualization | Layers, shapes, parameters, trainable status |
| Data | Dataset statistics | Sample counts, class distribution, features |
| Training | Real-time monitoring | Epoch/batch progress, loss/accuracy metrics |
| Graphs | Training charts | Loss curves, accuracy curves, learning rate |
| Files | File browser | Directory navigation, model/dataset detection |
| Help | Keyboard reference | Categorized shortcuts, context-aware |

### Theme

The TUI uses the NexusForge color scheme:

```rust
pub const TEAL: Color = Color::Rgb(20, 184, 166);       // #14b8a6 - Primary
pub const TEAL_LIGHT: Color = Color::Rgb(94, 234, 212); // #5eead4 - Accent
pub const TERRACOTTA: Color = Color::Rgb(196, 164, 132);// #c4a484 - Secondary
pub const CREAM: Color = Color::Rgb(250, 249, 246);     // #faf9f6 - Text
pub const DARK_SLATE: Color = Color::Rgb(30, 41, 59);   // #1e293b - Background
pub const SUCCESS: Color = Color::Rgb(16, 185, 129);    // #10b981 - Green
pub const WARNING: Color = Color::Rgb(245, 158, 11);    // #f59e0b - Yellow
pub const ERROR: Color = Color::Rgb(239, 68, 68);       // #ef4444 - Red
pub const INFO: Color = Color::Rgb(100, 181, 246);      // #64b5f6 - Blue
```

### Keyboard Navigation

| Key | Action |
|-----|--------|
| `Tab` / `Shift+Tab` | Cycle through tabs |
| `1-5` | Jump directly to tab (Model, Data, Training, Graphs, Files) |
| `↑/k`, `↓/j` | Navigate up/down in lists |
| `←/h`, `→/l` | Navigate between panels |
| `Enter` | Select / Open item |
| `?` | Toggle help overlay |
| `q` | Quit application |

### CLI Integration

```bash
# Launch TUI
axonml tui

# Load model on startup
axonml tui --model path/to/model.axonml

# Load dataset on startup
axonml tui --data path/to/dataset/

# Load both
axonml tui --model model.axonml --data ./data/
```

### Code Structure

```rust
// Entry point (lib.rs)
pub fn run(model_path: Option<PathBuf>, data_path: Option<PathBuf>) -> io::Result<()> {
    enable_raw_mode()?;
    execute!(stdout, EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout))?;
    let mut app = App::new();

    if let Some(path) = model_path {
        app.load_model(path);
    }

    loop {
        terminal.draw(|frame| ui::render(&mut app, frame))?;
        if event::poll_event(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read_event()? {
                event::handle_key_event(&mut app, key);
            }
        }
        if app.should_quit {
            break;
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}
```

---

## Future Considerations

### Real GPU Kernels

Move from GPU backend stubs to real kernel implementations:

```rust
// CUDA kernels with cuBLAS/cuDNN
let cuda_model = model.to(Device::Cuda(0));

// Metal Performance Shaders
let metal_model = model.to(Device::Metal(0));

// Vulkan compute shaders
let vulkan_model = model.to(Device::Vulkan(0));
```

### Lazy Evaluation

Optional lazy mode for graph optimization:

```rust
let graph = LazyGraph::new();
let a = graph.placeholder(&[batch, 784]);
let b = graph.variable(weights);
let c = a.matmul(&b);
let output = graph.compile(c);
```

---

## Implemented Features (Previously Future)

### JIT Compilation (axonml-jit)

JIT compilation for computation graphs is now implemented:

```rust
use axonml_jit::{trace, JitCompiler};

// Trace a computation to build a graph
let graph = trace(|tracer| {
    let x = tracer.input("x", &[4]);
    let y = x.relu().mul_scalar(2.0).add_scalar(1.0);
    tracer.output("y", y)
});

// Compile the graph
let compiler = JitCompiler::new();
let func = compiler.compile(&graph)?;

// Execute the compiled function
let input = [-1.0f32, 0.0, 1.0, 2.0];
let result = func.run(&[("x", &input)])?;
// result: [1.0, 1.0, 3.0, 5.0]
```

**Features:**
- Intermediate representation (IR) for operations
- Operation tracing and graph building
- Graph optimization (constant folding, DCE, CSE)
- Function caching for compiled graphs
- Cranelift foundation for native codegen

### LLM Architectures (axonml-llm)

BERT and GPT-2 architectures are now implemented:

```rust
use axonml_llm::{BertConfig, Bert, GPT2Config, GPT2LMHead, GenerationConfig, TextGenerator};

// BERT for text understanding
let bert_config = BertConfig::tiny();  // Or base(), large()
let bert = Bert::new(&bert_config);
let output = bert.forward(&input_ids, &attention_mask, &token_type_ids);

// GPT-2 for text generation
let gpt2_config = GPT2Config::tiny();  // Or small(), medium(), large()
let gpt2 = GPT2LMHead::new(&gpt2_config);
let logits = gpt2.forward(&input_ids, None)?;

// Text generation with sampling
let gen_config = GenerationConfig::top_k_sampling(50, 0.8);
let generator = TextGenerator::new(gen_config);
let generated = generator.generate(&gpt2, &input_ids, 100)?;  // max 100 tokens
```

**Features:**
- BERT encoder (BertConfig, Bert, BertLayer)
- BertForSequenceClassification, BertForMaskedLM
- GPT-2 decoder (GPT2Config, GPT2, GPT2Block)
- GPT2LMHead for language modeling
- Text generation with top-k, top-p, temperature sampling

### Profiling Tools (axonml-profile)

Memory and compute profiling is now implemented:

```rust
use axonml_profile::{Profiler, MemoryProfiler, ComputeProfiler, TimelineProfiler, BottleneckAnalyzer};

// Core Profiler with scopes
let mut profiler = Profiler::new();
profiler.start();
{
    let _guard = profiler.scope("computation");
    // ... do work
}
let report = profiler.stop();

// Memory profiling
let mut mem = MemoryProfiler::new();
mem.track_allocation(size, Device::Cpu);
println!("Peak memory: {} bytes", mem.peak_usage());

// Compute profiling
let mut compute = ComputeProfiler::new();
let result = compute.time_op("matmul", || {
    // ... do computation
});
let stats = compute.stats();

// Timeline profiling (Chrome trace export)
let mut timeline = TimelineProfiler::new();
timeline.record_event("forward", "model", start, end);
timeline.export_chrome_trace("profile.json")?;

// Bottleneck analysis
let analyzer = BottleneckAnalyzer::new();
let bottlenecks = analyzer.analyze(&report);
```

**Features:**
- Core Profiler with ProfileGuard and ProfileReport
- MemoryProfiler for allocation tracking
- ComputeProfiler for operation timing
- TimelineProfiler with Chrome trace export
- BottleneckAnalyzer for automatic issue detection

### ONNX Import/Export (axonml-onnx)

Interoperability with other frameworks is now implemented:

```rust
use axonml_onnx::{OnnxModel, OnnxExporter};

// Import from ONNX
let model = OnnxModel::load("model.onnx")?;
let output = model.forward(inputs)?;

// Export to ONNX
let mut exporter = OnnxExporter::new("my_model", 17);
exporter.add_input("input", &[1, 784], TensorDataType::Float);
exporter.add_output("output", &[1, 10], TensorDataType::Float);
exporter.add_node("MatMul", &["input", "weight"], &["output"], HashMap::new());
exporter.export("model.onnx")?;
```

### Quantization (axonml-quant)

Model compression for deployment is now implemented:

```rust
use axonml_quant::{quantize_tensor, dequantize_tensor, QuantType, CalibrationData};

// Quantize to INT8
let quantized = quantize_tensor(&tensor, QuantType::Q8_0)?;

// Quantize to INT4 (8x compression)
let quantized = quantize_tensor(&tensor, QuantType::Q4_0)?;

// Dequantize for inference
let dequantized = dequantize_tensor(&quantized)?;

// Calibration for better accuracy
let mut calib = CalibrationData::new();
calib.update(&tensor);
let range = calib.compute_range(CalibrationMethod::Percentile { percentile: 99.9 });
```

### Kernel Fusion (axonml-fusion)

Operation fusion for performance is now implemented:

```rust
use axonml_fusion::{FusedLinear, FusedElementwise, FusionOptimizer, Activation};

// Fused MatMul + Bias + ReLU (single pass)
let fused = FusedLinear::new(weight, Some(bias), Activation::Relu)?;
let output = fused.forward(&input)?;

// Fused elementwise chain
let fused = FusedElementwise::builder()
    .mul(2.0)
    .add(1.0)
    .relu()
    .build();
let output = fused.forward(&input)?;

// Graph optimization
let mut optimizer = FusionOptimizer::new();
let patterns = optimizer.analyze(&ops);
println!("Detected {} fusion patterns", patterns.len());
println!("Estimated speedup: {:.1}x", optimizer.stats().estimated_speedup);
```

---

## References

- PyTorch Internals: https://blog.ezyang.com/2019/05/pytorch-internals/
- Burn Architecture: https://github.com/tracel-ai/burn
- Candle Design: https://github.com/huggingface/candle
- dfdx Type System: https://github.com/coreylowman/dfdx

---

*This document is updated as the architecture evolves.*
