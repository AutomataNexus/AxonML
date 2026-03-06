# Changelog

All notable changes to Axonml will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-03-04

### Milestone: Novel Capabilities Beyond PyTorch

AxonML now includes features that don't exist in any other ML framework. Five novel
subsystems extend the core crates, and a complete biometric identity framework (Aegis Identity)
demonstrates the framework's unique temporal, event-driven, and uncertainty-aware primitives.

### Added

#### Aegis Identity — Unified Biometric Framework (`axonml-vision`)
- **Mnemosyne** (~115K params) - Face identity via temporal crystallization: GRU hidden state
  converges to an identity attractor over multiple observations, quality-gated updates,
  attention-weighted multi-frame aggregation, temporal liveness detection, drift monitoring
- **Ariadne** (~65K params) - Fingerprint via ridge event fields: learned Gabor wavelet bank
  extracts 8-orientation ridge responses, ridge density mapping, core/delta singularity
  detection via Poincare index, partial fingerprint matching
- **Echo** (~68K params) - Voice via predictive speaker residuals: a generic speech predictor
  learns to predict the next mel frame; prediction errors ARE the speaker identity (identity =
  what cannot be predicted), replay detection, VAD, speaking rate estimation
- **Argus** (~65K params) - Iris via polar-native radial phase encoding: separate radial and
  angular 1D convolutions on polar-unwrapped iris, multi-resolution encoding at 3 scales,
  Hamming distance matching with binarized codes, fragile bit masking
- **Themis** (~49K params) - Multimodal belief propagation fusion: uncertainty-aware dynamic
  weighting, cross-modal consistency checking, GRU temporal belief accumulation, evidential
  uncertainty (Dirichlet-based), conflict detection, modality reliability tracking
- **AegisIdentity** unified API - enroll/verify/identify with any subset of modalities,
  forensic verification with audit trails, batch operations, identity drift detection,
  quality assessment, liveness detection, secure verification pipeline, operating curve computation
- Biometric-specific losses: CrystallizationLoss, ContrastiveLoss, PredictiveCodingLoss,
  PhaseConsistencyLoss, CenterLoss, AngularMarginLoss, DiversityRegularization, LivenessLoss
- Iris polar unwrap utilities with rotation estimation via cross-correlation
- Total: ~362K params, <2MB, each modality independently deployable on Raspberry Pi

#### Graph Inspection API (`axonml-autograd`)
- `trace_backward(variable)` — DFS walk through grad_fn chain to capture computation graph
- `to_dot(snapshot)` — Export computation graph to Graphviz DOT format for visualization
- `GraphSnapshot` with `node_count()`, `depth()`, `leaf_count()`, `operation_names()`
- `gradient_flow_summary()` — Analyze gradient flow health through the graph
- Native capability (unlike PyTorch which requires external `torchviz` package)

#### Lazy Tensor Computation (`axonml-tensor`)
- `LazyTensor` — Deferred execution model where operations build an expression tree
- Algebraic optimization pass before materialization: constant folding, identity elimination,
  double negation cancellation, inverse operation cancellation, scalar folding
- Supports all unary, binary, reduction, and shape operations
- `materialize()` evaluates the optimized expression tree into a concrete Tensor
- Built into the tensor type — no external JIT compiler needed

#### Differentiable Structured Sparsity (`axonml-nn`)
- `SparseLinear` — Linear layer with learnable pruning mask via soft thresholding:
  `sigmoid((|weight| - threshold) * temperature)` makes the mask differentiable
- `GroupSparsity` — Group L1/L2 regularization for structured (row/column/block) sparsity
- `LotteryTicket` — Lottery Ticket Hypothesis implementation: snapshot initial weights,
  iterative magnitude pruning, rewind to initial weights with discovered mask
- The pruning mask is end-to-end differentiable, unlike PyTorch's binary masking

#### Training Health Monitor (`axonml-optim`)
- `TrainingMonitor` — Self-monitoring training diagnostics attached to the optimizer
- Detects: NaN loss/gradients, gradient explosion/vanishing, loss plateau, loss oscillation,
  learning rate too high/low, dead neurons, training divergence
- `LossTrend` analysis: Decreasing, Stable, Increasing, Oscillating, Converged
- `suggest_lr()` — Automatic learning rate suggestions based on gradient statistics
- `convergence_score()` — Quantified convergence metric
- `HealthReport` with per-step alerts at Info/Warning/Critical severity levels

### Changed
- Test count: 1076+ → 1575+ across all crates
- axonml-autograd: 52 → 105 tests
- axonml-tensor: 64 → 98 tests
- axonml-nn: 76 → 171 tests
- axonml-optim: 40 → 79 tests
- axonml-vision: 75 → 607 tests

## [0.3.0] - 2026-02-27

### Milestone: Production Edge Inference

AxonML models are running live production inference on 6 edge controllers (Raspberry Pi),
monitoring HVAC equipment across 5 buildings. 12 models (6 anomaly detectors + 6 failure
predictors) deployed via cross-compiled ARM binaries, each running at ~2-3 MB RSS.

### Added

#### Autograd Fixes (`axonml-autograd`, `axonml-nn`)
- Fixed critical autograd graph-severing bug where `Variable::new()` was used for
  intermediate results, creating leaf variables that blocked gradient flow
- Fixed LSTM/GRU weight transpose operations (6 instances in `rnn.rs`)
- Fixed `stack_outputs` in RNN/LSTM/GRU to use `unsqueeze` + `Variable::cat`
- Added `CrossEntropyBackward` gradient function for proper backpropagation
- Made `Variable::from_operation` public for custom gradient-tracked operations

#### Tensor Operations (`axonml-tensor`)
- `Tensor::cat(tensors, dim)` with `CatBackward` gradient function
- `Variable::cat(vars, dim)` for autograd-tracked concatenation
- `Tensor::sum_dim(dim, keepdim)` with `SumDimBackward` gradient function
- `Variable::sum_dim(dim)` for autograd-tracked dimension reduction

#### CUDA Backend (`axonml-core`, `axonml-tensor`)
- CUDA matrix multiplication dispatch via cuBLAS GEMM

#### Serialization (`axonml-serialize`)
- Model save/load for production deployment (`.axonml` format)
- StateDict extraction for weight export

#### Production Edge Inference
- Pure-tensor inference daemons (no autograd overhead) for ARM deployment
- Cross-compilation pipeline for `armv7-unknown-linux-musleabihf` (static musl)
- HTTP API endpoints (`/health`, `/api/inference/latest`) for integration
- Rolling window buffers for time-series LSTM/GRU inference
- PM2 process management for production uptime

### Production Deployments

| Building | Unit | Anomaly Model | Failure Predictor | Controller |
|----------|------|---------------|-------------------|------------|
| FCOG | Mechroom | Erebus (128K params) | Kairos (288K params) | 100.123.60.69 |
| Warren | AHU-1 | Aether (32K params) | Moros (73K params) | 100.124.76.93 |
| Warren | AHU-2 | Phanes (71K params) | Hecate (162K params) | 100.95.58.104 |
| Warren | AHU-4 | Nyctos (32K params) | Cassandra (73K params) | 100.121.143.51 |
| Warren | AHU-7 | Poseidon (32K params) | Triton (73K params) | 100.125.245.8 |
| Huntington | Mechroom | Plutus (127K params) | Moira (288K params) | 100.73.201.107 |

### Changed
- Bumped version from 0.2.8 to 0.3.0

## [0.1.0] - 2024-XX-XX

### Added

#### Core (`axonml-core`)
- Device abstraction (CPU, CUDA, Vulkan, Metal, WebGPU)
- Data type system (F32, F64, I32, I64, Bool, etc.)
- Unified error handling
- Memory storage primitives
- CPU backend implementation

#### Tensor (`axonml-tensor`)
- N-dimensional Tensor struct with shape/strides
- Tensor creation functions (zeros, ones, rand, randn, arange, linspace)
- Arithmetic operations (+, -, *, /, matmul)
- Broadcasting support
- Shape operations (reshape, transpose, squeeze, unsqueeze, permute)
- Slicing and indexing (select, narrow, chunk, split)
- Reduction operations (sum, mean, max, min)
- Activation functions (relu, sigmoid, tanh, softmax, gelu)

#### Autograd (`axonml-autograd`)
- Variable wrapper with gradient tracking
- Dynamic computational graph
- Backward pass with automatic differentiation
- Gradient functions for all tensor operations
- `no_grad` context manager
- Gradient accumulation support

#### Neural Networks (`axonml-nn`)
- Module trait for neural network components
- Parameter wrapper for trainable weights
- Sequential container
- Linear (fully connected) layer
- Convolutional layers (Conv1d, Conv2d, Conv3d)
- Pooling layers (MaxPool2d, AvgPool2d, GlobalAvgPool2d)
- Normalization (BatchNorm1d, BatchNorm2d, LayerNorm)
- Dropout regularization
- Recurrent layers (RNN, LSTM, GRU)
- Multi-head attention
- Embedding layer
- Activation modules (ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, GELU, SiLU)
- Loss functions (MSELoss, CrossEntropyLoss, BCELoss, L1Loss)
- Weight initialization (Xavier, Kaiming, normal, uniform)

#### Optimizers (`axonml-optim`)
- Optimizer trait
- SGD with momentum and Nesterov
- Adam optimizer
- AdamW (decoupled weight decay)
- RMSprop
- Learning rate schedulers (StepLR, ExponentialLR, CosineAnnealingLR)

#### Data Loading (`axonml-data`)
- Dataset trait
- DataLoader with batching
- Shuffling support
- Sequential and random samplers
- Transform trait for data preprocessing

#### Vision (`axonml-vision`)
- Image transforms (Resize, CenterCrop, RandomHorizontalFlip, Normalize)
- SyntheticMNIST dataset
- SyntheticCIFAR dataset
- LeNet architecture
- SimpleCNN architecture

#### Text (`axonml-text`)
- Tokenizer trait
- WhitespaceTokenizer
- CharTokenizer
- BasicBPETokenizer (Byte-Pair Encoding)
- Vocabulary management
- TextDataset
- LanguageModelDataset
- SyntheticSentimentDataset

#### Audio (`axonml-audio`)
- Resample transform
- MelSpectrogram transform
- MFCC (Mel-frequency cepstral coefficients)
- Audio normalization
- AddNoise augmentation
- SyntheticCommandDataset
- SyntheticMusicDataset

#### Distributed (`axonml-distributed`)
- DistributedDataParallel (DDP) wrapper
- Process group management
- World abstraction
- Communication primitives (all_reduce, broadcast, barrier)
- Mock backend for testing

#### Umbrella Crate (`axonml`)
- Re-exports all subcrates
- Prelude module for convenient imports
- Feature flags for modular builds

### Documentation
- Comprehensive README
- Architecture documentation
- Per-module documentation in `/docs/`
- Code examples in `/examples/`

### Examples
- `simple_training.rs` - XOR problem with MLP
- `mnist_training.rs` - CNN training on SyntheticMNIST
- `nlp_audio_test.rs` - Text and audio processing demo

---

## Version History

- **0.4.0**: Novel capabilities beyond PyTorch — Aegis Identity biometric framework, graph inspection, lazy tensors, differentiable sparsity, training health monitor
- **0.3.0**: Production edge inference — 12 models deployed across 6 controllers
- **0.1.0**: Initial release with complete ML framework

[Unreleased]: https://github.com/AutomataNexus/AxonML/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/AutomataNexus/AxonML/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/AutomataNexus/AxonML/compare/v0.1.0...v0.3.0
[0.1.0]: https://github.com/AutomataNexus/AxonML/releases/tag/v0.1.0
