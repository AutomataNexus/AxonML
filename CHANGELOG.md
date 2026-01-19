# Changelog

All notable changes to Axonml will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

- **0.1.0**: Initial release with complete ML framework

[Unreleased]: https://github.com/automatanexus/axonml/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/automatanexus/axonml/releases/tag/v0.1.0
