# Changelog

All notable changes to Axonml will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2026-04-08

### Summary
Production readiness release. **100+ new tests** across all foundation crates, comprehensive CPU backward pass optimization, and zero clippy warnings. Test suite grows from 2,141 to **2,240+** passing tests. Every critical training path — forward, loss, backward, optimizer step, checkpoint save/load — now has verified correctness tests. CPU performance improved by eliminating unnecessary memory copies in backward passes and adding fast paths for contiguous tensor operations.

**14 files changed, 2,142 lines added, 218 lines removed.**

### Added

#### Test Coverage (100+ new tests)

##### `axonml-optim` (+18 tests)
- Adam step mathematical correctness (verifies update formula)
- Adam/AdamW convergence on quadratic via autograd (end-to-end training loop test)
- `zero_grad()` clears all parameter gradients
- Frozen parameter handling (optimizer skips requires_grad=false)
- Weight decay shrinks parameters even with zero gradient
- Learning rate get/set management
- ReduceLROnPlateau: max mode, min_lr floor, cooldown behavior
- OneCycleLR: full cycle shape, monotonic warmup/annealing phases
- CosineAnnealingLR: monotonic decrease, eta_min convergence
- WarmupLR: constant after warmup completes

##### `axonml-nn` (+35 tests)
- **Loss functions**: MSE gradient correctness + sum reduction, L1 basic/zero, BCE perfect/worst prediction, BCEWithLogits numerical stability (large logits) + zero logits + reduction modes, SmoothL1 small/large error regimes, CrossEntropy batch independence + 100-class scaling
- **Normalization**: LayerNorm zero-mean/unit-variance output + gradient flow + batch independence + parameter count, BatchNorm1d per-channel normalization + train/eval mode difference, BatchNorm2d gradient flow, GroupNorm gradient flow
- **RNN/LSTM/GRU**: LSTMCell forward_step, LSTM multi-layer + gradient flow + different sequence lengths + bounded outputs + parameter count, GRUCell forward_step, GRU multi-layer + forward_mean + forward_last + gradient flow + hidden state evolution, RNNCell gradient flow, RNN multi-layer, GRU numerical stability with large inputs
- **Conv2d**: Conv1d with padding/stride + multi-channel, grouped Conv2d gradient flow, groups=2, depthwise separable pattern (dw+pw chain), ConvTranspose2d upsample verification + gradient correctness + multi-channel

##### `axonml-autograd` (+17 tests)
- Arithmetic backward: add, sub, mul, div, mul_scalar — all verified against analytical gradients
- Activation backward: relu, sigmoid, tanh — derivative values checked at known points
- Chain rule: mul-then-add composite, nested relu(x^2-1) composite
- Reductions: sum backward (all-ones gradient), mean backward (1/N gradient)
- Matmul backward: gradient shapes and non-zero flow for both operands
- Edge cases: no_grad skips backward, detach stops gradient flow, backward accumulation, reshape preserves gradient

##### `axonml-serialize` (+4 tests)
- Model save/load roundtrip (Linear weights survive exactly)
- `StateDict::from_module()` extracts correct parameter count and finite values
- Full checkpoint roundtrip with model state + training state + config + metrics
- TrainingState custom metrics tracking and best-metric update logic

##### `axonml-tensor` (+14 tests)
- `where_cond` basic conditional selection
- `scatter` duplicate index handling
- `unique` all-same and all-unique edge cases
- `flip` both dimensions, column-only flip
- `roll` 2D rolling, full-cycle (roll by length = identity)
- `nonzero` all-zeros and all-nonzero edge cases
- `softmax` numerical stability with large values (+1000) and negative values (-200)
- `clamp_min` no-op and all-negative cases

#### Tensor API
- `Tensor::map()` — apply function element-wise, single allocation
- `Tensor::zip_map()` — binary element-wise operation, single allocation (replaces to_vec + zip + from_vec pattern)
- `Tensor::zip_map3()` — ternary element-wise operation

### Changed

#### CPU Performance Optimizations

##### Backward Pass — Eliminated 22 unnecessary `to_vec()` memory copies
- **ReluBackward**: `zip_map(|x, g| if x > 0 { g } else { 0 })` — one allocation instead of three
- **SigmoidBackward**: `zip_map(|o, g| g * o * (1-o))` — same
- **TanhBackward**: `zip_map(|o, g| g * (1 - o*o))` — same
- **GeluBackward**: `zip_map` with full GELU derivative formula
- **SiluBackward**: `zip_map` with SiLU derivative
- **EluBackward**: `zip_map` with alpha * exp(x) for negative inputs
- **LeakyReluBackward**: `zip_map` with negative_slope
- **ClampBackward**: `zip_map` with range check
- **BceLossBackward**: `zip_map` with gradient formula
- **L1LossBackward**: `zip_map` with sign function
- **SmoothL1LossBackward**: `zip_map` with beta threshold

##### Tensor Element-Wise Operations — Fast path for contiguous same-shape tensors
- `add()`, `sub()`, `mul()`, `div()` now skip `unravel_index` + `linear_index` per-element when both tensors have the same shape and are contiguous
- Eliminates O(ndim) index computation per element — the most common case in backward passes
- Broadcast path unchanged for shape-mismatched tensors

### Fixed
- 3 clippy warnings (`explicit .into_iter()` in zip arguments)
- All code formatted with `cargo fmt --all`

## [0.5.0] - 2026-03-31

### Summary
Full-crate audit and fix sweep across all 22 crates. 121 files changed, ~5,000 lines added, ~1,700 removed. **2,141 tests passing** (up from 1,988). Every crate audited for correctness, performance, security, and completeness.

### Critical Fixes

#### GPU Training Correctness (`axonml-distributed`)
- **DDP gradient sync was a no-op** --- `sync_gradients()` computed all-reduced gradients but discarded them. Every DDP training run produced unsynchronized models. Now writes back via `param.set_grad()`.
- **FSDP gradient sync was a no-op** --- Same issue for ZeRO-2/ZeRO-3 strategies. Fixed for all sharding modes (NoShard, ShardGradOp, FullShard, HybridShard).
- Implemented real **1F1B pipeline schedule** (was falling back to GPipe). Memory-efficient 3-phase warmup/steady/cooldown.

#### Security (`axonml-server`, `axonml-dashboard`)
- **Terminal endpoint now requires admin role** --- Any authenticated user previously got unrestricted shell access. Added role check + audit logging.
- **Rate limiting on login/register/MFA** --- Added IP-based sliding-window rate limiter (10 req/60s) to prevent brute-force.
- **JWT tokens moved from localStorage to sessionStorage** --- Access tokens cleared on browser close, reducing XSS exposure window.
- **SVG icon sanitization** --- `inner_html` paths now reject `<script>`, `javascript:`, and event handlers.
- **Error boundary** --- Wraps all dashboard routes; component errors show fallback UI instead of crashing WASM.

#### ONNX Interoperability (`axonml-onnx`)
- **Export now produces real protobuf binary** --- Was outputting JSON. Models are now compatible with ONNX Runtime, TensorRT, OpenVINO. Added prost Message structs with correct field tags.

#### Audio Performance (`axonml-audio`)
- **O(n^2) DFT replaced with O(n log n) FFT** via rustfft --- ~100x speedup for MelSpectrogram/MFCC on real audio.

### Fixed

#### `axonml-core`
- `from_size_align_unchecked` potential UB --- round allocation size to alignment multiple
- 20 new tests for GPU memory pool and backend traits

#### `axonml-data`
- `TensorDataset::get()` copied entire dataset per access --- cache flat vecs at construction, O(row) per get
- `prefetch_to_gpu` eagerly materialized all batches --- streaming via bounded channel, 2-batch buffer
- `Normalize` per-channel support (ImageNet preset)
- `RandomFlip` generic N-dimensional (was 2D only)
- `WeightedRandomSampler` binary search (was O(n) linear scan)
- `DropoutTransform` train/eval mode
- `concat_tensors` non-dim-0 interleaving
- `StackCollate` non-zero dim

#### `axonml-llm`
- **WindowedAttnBackward** real gradient computation (was identity pass-through with 7 dead fields)
- LLaMA `generate()` proper sampling (was greedy-only)
- Beam search execution loop (config existed but no logic)
- TridentRMSNorm deduplication (3 copies -> 1 shared)
- `From<HubError> for LLMError` error composition
- Attention dropout now applied in Hydra

#### `axonml-text`
- `TextDataset` stores tokenizer (was hardcoding whitespace split)
- BPE priority-based merges (was greedy left-to-right)
- Unigram Viterbi segmentation (was greedy longest-match)
- `Vocab::from_tokens()` auto-adds UNK/PAD
- Serde serialization for `Vocab` (save/load)

#### `axonml-cli`
- `--seed` flag now applied (was printed and discarded)
- Quant command delegates to `axonml-quant` (removed 400 lines of duplicate code)

#### `axonml-tui`
- Dataset loading reads real CSV files (was always demo data)
- Training view reads log files for live updates (was simulating fake counters)
- File browser loads real filesystem (was hardcoded demo entries)
- Zoom toggle implemented for graphs view

#### `axonml-vision`
- 15 new NightVision backbone/neck/head tests
- Octree affected node tracking (was TODO)

#### `axonml-audio`
- All 4 dataset types return class index labels (were one-hot, incompatible with CrossEntropyLoss)

#### `axonml-text`
- All dataset types return class index labels

#### `axonml-dashboard`
- 94 panicking `unwrap()` calls in system.rs -> `.ok()` (graceful degradation)
- Consolidated 3 duplicate token key definitions into `constants.rs`
- Storage errors logged instead of silently discarded
- Client-side validation for login/register/MFA forms

### Changed
- Version bump to 0.5.0 across all 22 crates
- `ReduceOp::Average` clean sum+divide path (was redundant pairwise+recompute)
- Dead code cleanup: removed `#[allow(dead_code)]` from 30+ fields across distributed, llm, vision crates

### Added
- `axonml-server`: `RateLimiter` module with sliding-window IP rate limiting
- `axonml-dashboard`: `PageErrorBoundary` component, `constants.rs` module, `js_helpers.rs` utilities
- `axonml-text`: `Vocab::save()`/`Vocab::load()` JSON persistence
- `axonml-data`: `Normalize::per_channel()`, `Normalize::imagenet()`
- `axonml-llm`: `TextGenerator::generate_beam_search()`
- `axonml-onnx`: Prost binary encoding structs, 3 roundtrip tests

## [0.4.2] - 2026-03-24

### Added

#### NightVision — Multi-Domain Thermal IR Detector (`axonml-vision`)
- **NightVision** (~2.6M params) — CSP backbone + Thermal FPN + YOLOX decoupled heads for infrared object detection
- 5 detection domains: Wildlife, Human, Interstellar, Vehicle, General — each with domain-specific class sets
- Thermal FPN with domain-adaptive feature modulation (learned scale/bias per domain)
- YOLOX-style decoupled heads (separate cls/reg/obj branches) with anchor-free detection
- Full model documentation in `crates/axonml-vision/src/models/nightvision/`

#### Biometric GPU Training Pipelines (`axonml-vision`)
- **Mnemosyne** face verification training — GPU-accelerated with ArcFace loss, checkpoint/resume, LFW dataset support
- **Argus** iris recognition training — GPU-accelerated with phase consistency loss, CASIA-Iris dataset support
- **Ariadne** gait recognition training — GPU-accelerated with triplet loss, FVC2000 dataset support
- Pre-computed polar coordinate cache for Argus iris training (eliminates per-epoch recomputation)
- `bench_mnemosyne` benchmark example for face verification evaluation
- Biometric model documentation: 6 READMEs covering Argus, Ariadne, Echo, Mnemosyne, Themis, and suite overview

### Changed
- CUDA enabled by default in `axonml-vision` Cargo.toml
- Default trait implementations added for 20 types across the workspace
- All clippy warnings fixed (0 warnings with `-D warnings`)

### Security
- `rustls-webpki` updated to 0.103.10 (security patch)

## [0.4.1] - 2026-03-08

### Performance: Framework-Wide Optimization Pass

Comprehensive performance audit and optimization across all 22 crates. All 1,988 tests pass with zero failures.

### Changed

#### Conv Backward — BLAS Acceleration (`axonml-autograd`)
- **Conv2dBackward**: Replaced 7-deep nested loops with im2col + matrixmultiply GEMM — **12x faster** (107ms → 8.7ms on BlazeFace)
- **GroupedConv2dBackward**: Same im2col + GEMM treatment for depthwise/grouped convolutions
- **Conv1dBackward**: Replaced 6 nested loops with im2col + GEMM
- **ConvTranspose2dBackward**: Reordered loops for cache locality, vectorized bias gradient
- Bias gradients across all conv ops vectorized with `iter().sum()`

#### Normalization Backward — Loop Fusion (`axonml-autograd`)
- **BatchNorm2d/1dBackward**: Cached x_hat values to eliminate recomputation, hoisted scale factor
- **AvgPool2d/1dBackward**: Replaced double kernel loop with analytical count computation
- **SoftmaxBackward**: Fixed O(N²) stride recomputation with precomputed strides array

#### RNN/LSTM/GRU — Batched Matmul (`axonml-nn`)
- **LSTM/GRU/RNN**: Pre-compute input-to-hidden projection for ALL timesteps in one GEMM (replaces seq_len separate matmuls)
- **GRU**: Eliminated per-timestep `ones` tensor allocation — rewrote `(1-z)*n + z*h` as `n + z*(h-n)`

#### Optimizer Step — Fused Loops (`axonml-optim`)
- **Adam/AdamW**: Fused 3 separate loops into single pass, eliminated intermediate Vec allocations
- **SGD**: Eliminated redundant `.clone()`, in-place momentum and parameter updates
- **RMSprop**: Eliminated intermediate Vec allocation, fused denominator computation
- **LAMB**: Fused update direction + norm computation into single pass

#### LLM Inference (`axonml-llm`)
- **RMSNorm**: Hoisted weight vector outside batch loop (was allocating per element)
- **RoPE**: Narrow cos/sin cached tensors to needed positions before copy
- **Causal mask**: Eliminated branch check in inner loop

#### Tensor Core Ops (`axonml-tensor`)
- **var_dim**: Reduced from 4 full-data passes to 3 via E[x²] - E[x]² approach
- **cat**: Replaced element-by-element copy with `copy_from_slice` (memcpy)

#### Vision (`axonml-vision`)
- **BlazeFace**: Dynamic feature map sizing — works with any input resolution, not just 128×128
- **NMS**: Skip already-processed boxes in inner loop
- **positional_encoding_2d**: Precomputed frequency table outside spatial loops

### Fixed
- **JWT authentication** (`axonml-server`): Added `rust_crypto` feature for jsonwebtoken v10 — fixes 3 test failures
- **BlazeFace multi-resolution**: Hardcoded 16×16/8×8 feature map sizes → dynamic from actual output shapes
- **Server integration tests**: Enhanced `require_server!()` macro to check admin login — tests skip gracefully when DB not initialized
- **coco_bench_face_detectors**: Fixed shape mismatch when running BlazeFace at 256×256

### Test Results
- **1,988 tests passed, 0 failures** across entire workspace
- All 22 crates compile and pass tests

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

| Building | Unit | Anomaly Model | Failure Predictor |
|----------|------|---------------|-------------------|
| Site A | Mechroom | Erebus (128K params) | Kairos (288K params) |
| Site B | AHU-1 | Aether (32K params) | Moros (73K params) |
| Site B | AHU-2 | Phanes (71K params) | Hecate (162K params) |
| Site B | AHU-4 | Nyctos (32K params) | Cassandra (73K params) |
| Site B | AHU-7 | Poseidon (32K params) | Triton (73K params) |
| Site C | Mechroom | Plutus (127K params) | Moira (288K params) |

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

- **0.4.2**: NightVision multi-domain IR detector, biometric GPU training pipelines with checkpoint/resume
- **0.4.1**: Framework-wide performance optimization pass (conv backward 12x faster, fused optimizer loops)
- **0.4.0**: Novel capabilities beyond PyTorch — Aegis Identity biometric framework, graph inspection, lazy tensors, differentiable sparsity, training health monitor
- **0.3.0**: Production edge inference — 12 models deployed across 6 controllers
- **0.1.0**: Initial release with complete ML framework

[Unreleased]: https://github.com/AutomataNexus/AxonML/compare/v0.4.2...HEAD
[0.4.2]: https://github.com/AutomataNexus/AxonML/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/AutomataNexus/AxonML/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/AutomataNexus/AxonML/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/AutomataNexus/AxonML/compare/v0.1.0...v0.3.0
[0.1.0]: https://github.com/AutomataNexus/AxonML/releases/tag/v0.1.0
