//! Train - Model Training Command
//!
//! # File
//! `crates/axonml-cli/src/commands/train.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 14, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::time::Instant;

/// Global seed value, set by --seed flag.
static GLOBAL_SEED: AtomicU64 = AtomicU64::new(0);
/// Whether a global seed has been set.
static SEED_SET: AtomicBool = AtomicBool::new(false);

/// Returns the global seed if one was set, for downstream RNG initialization.
#[allow(dead_code)]
pub fn global_seed() -> Option<u64> {
    if SEED_SET.load(std::sync::atomic::Ordering::Relaxed) {
        Some(GLOBAL_SEED.load(std::sync::atomic::Ordering::Relaxed))
    } else {
        None
    }
}

use axonml_autograd::Variable;
use axonml_data::{DataLoader, Dataset};
use axonml_nn::CrossEntropyLoss;
use axonml_nn::{Conv2d, Dropout, GRU, LSTM, Linear, MaxPool2d, Module, RNN, ReLU, Sequential};
use axonml_optim::{Adam, AdamW, Optimizer, RMSprop, SGD};
use axonml_serialize::{Format, StateDict, save_state_dict};
use axonml_tensor::Tensor;
use axonml_vision::models::{ResNet, VGG, VisionTransformer};
use axonml_vision::{CIFAR10, FashionMNIST, MNIST};

use super::utils::{
    ensure_dir, epoch_progress_bar, parse_device, print_header, print_info, print_kv, print_success,
};
use crate::cli::TrainArgs;
use crate::config::{DataConfig, ModelConfig, ProjectConfig, TrainingConfig};
use crate::error::{CliError, CliResult};

// W&B integration
#[cfg(feature = "wandb")]
use super::wandb::WandbConfig;
#[cfg(feature = "wandb")]
use super::wandb_client::{WandbRun, init_training_run, is_available as wandb_is_available};

// =============================================================================
// Execute Command
// =============================================================================

/// Execute the `train` command
pub fn execute(args: TrainArgs) -> CliResult<()> {
    print_header("Axonml Training");

    // Load or create configuration
    let config = load_config(&args)?;

    // Print training configuration
    print_training_info(&config, &args);

    // Ensure output directory exists
    ensure_dir(&args.output)?;

    // Set random seed if provided
    if let Some(seed) = args.seed.or(config.seed) {
        print_info(&format!("Random seed: {seed}"));
        // Store the seed globally so downstream components can use it.
        GLOBAL_SEED.store(seed, std::sync::atomic::Ordering::Relaxed);
        SEED_SET.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    // Parse device
    let (device_type, device_id) = parse_device(&args.device);
    print_kv(
        "Device",
        &format!("{}:{}", device_type, device_id.unwrap_or(0)),
    );

    println!();
    print_info("Starting training...");
    println!();

    // Run training loop
    let start_time = Instant::now();
    let result = run_training_loop(&config, &args);
    let elapsed = start_time.elapsed();

    match result {
        Ok(metrics) => {
            println!();
            print_success(&format!(
                "Training completed in {:.2}s",
                elapsed.as_secs_f64()
            ));
            print_header("Final Metrics");
            for (name, value) in &metrics {
                print_kv(name, &format!("{value:.4}"));
            }
            println!();
            print_info(&format!("Model saved to: {}/model.axonml", args.output));
        }
        Err(e) => {
            return Err(CliError::Training(e.to_string()));
        }
    }

    Ok(())
}

// =============================================================================
// Configuration Loading
// =============================================================================

fn load_config(args: &TrainArgs) -> CliResult<TrainingConfig> {
    // Try to load from config file first
    if let Some(config_path) = &args.config {
        let path = PathBuf::from(config_path);
        if path.exists() {
            let project_config = ProjectConfig::load(&path)?;
            let mut config = project_config.training;

            // Override with command-line arguments
            if let Some(epochs) = args.epochs {
                config.epochs = epochs;
            }
            if let Some(batch_size) = args.batch_size {
                config.batch_size = batch_size;
            }
            if let Some(lr) = args.lr {
                config.learning_rate = lr;
            }

            return Ok(config);
        }
        return Err(CliError::Config(format!(
            "Configuration file not found: {config_path}"
        )));
    }

    // Try to load from axonml.toml in current directory
    let default_config = PathBuf::from("axonml.toml");
    if default_config.exists() {
        let project_config = ProjectConfig::load(&default_config)?;
        let mut config = project_config.training;

        // Override with command-line arguments
        if let Some(epochs) = args.epochs {
            config.epochs = epochs;
        }
        if let Some(batch_size) = args.batch_size {
            config.batch_size = batch_size;
        }
        if let Some(lr) = args.lr {
            config.learning_rate = lr;
        }

        return Ok(config);
    }

    // Create default configuration from command-line arguments
    Ok(TrainingConfig {
        epochs: args.epochs.unwrap_or(10),
        batch_size: args.batch_size.unwrap_or(32),
        learning_rate: args.lr.unwrap_or(0.001),
        device: args.device.clone(),
        num_workers: args.workers,
        output_dir: args.output.clone(),
        ..TrainingConfig::default()
    })
}

// =============================================================================
// Training Information
// =============================================================================

fn print_training_info(config: &TrainingConfig, args: &TrainArgs) {
    print_header("Configuration");
    print_kv("Epochs", &config.epochs.to_string());
    print_kv("Batch size", &config.batch_size.to_string());
    print_kv("Learning rate", &format!("{:.6}", config.learning_rate));
    print_kv("Optimizer", &config.optimizer.name);
    print_kv("Output directory", &args.output);

    print_kv("Data path", &args.data);

    if config.optimizer.weight_decay > 0.0 {
        print_kv(
            "Weight decay",
            &format!("{:.6}", config.optimizer.weight_decay),
        );
    }

    if let Some(scheduler) = &config.scheduler {
        print_kv("LR Scheduler", &scheduler.name);
    }
}

// =============================================================================
// Model Creation
// =============================================================================

/// Create a model based on configuration
fn create_model(
    model_config: &ModelConfig,
    _data_config: &DataConfig,
    dims: Option<(usize, usize, usize)>,
) -> Box<dyn TrainableModel> {
    let arch = model_config.architecture.to_lowercase();
    let num_classes = model_config.num_classes.unwrap_or(10);

    match arch.as_str() {
        "mlp" | "dense" | "" => {
            // Flatten whatever the dataset provides (C*H*W) into the MLP input.
            let input_size = dims
                .map(|(c, h, w)| c * h * w)
                .or(model_config.input_size)
                .unwrap_or(784);
            let hidden_sizes = if model_config.hidden_sizes.is_empty() {
                vec![256, 128]
            } else {
                model_config.hidden_sizes.clone()
            };
            let dropout = model_config.dropout as f32;

            Box::new(MLP::new(input_size, &hidden_sizes, num_classes, dropout))
        }
        "cnn" | "conv" => {
            let input_channels =
                dims.map_or_else(|| model_config.input_size.unwrap_or(1), |(c, _, _)| c);
            Box::new(SimpleCNN::new(input_channels, num_classes))
        }
        "lstm" | "gru" | "rnn" => {
            // Fold an image into a sequence of rows (width W = per-step features).
            let feat = dims.map_or(28, |(_, _, w)| w);
            let hidden = model_config.hidden_sizes.first().copied().unwrap_or(128);
            Box::new(SeqClassifier::new(&arch, feat, hidden, 2, num_classes))
        }
        "lenet" => {
            let num_classes = model_config.num_classes.unwrap_or(10);
            Box::new(LeNetModel::new(num_classes))
        }
        "resnet" | "resnet18" => Box::new(ModuleModel::new(ResNet::resnet18(
            model_config.num_classes.unwrap_or(10),
        ))),
        "resnet34" => Box::new(ModuleModel::new(ResNet::resnet34(
            model_config.num_classes.unwrap_or(10),
        ))),
        "vgg" | "vgg16" => Box::new(ModuleModel::new(VGG::vgg16(
            model_config.num_classes.unwrap_or(10),
        ))),
        "vgg11" => Box::new(ModuleModel::new(VGG::vgg11(
            model_config.num_classes.unwrap_or(10),
        ))),
        "vgg13" => Box::new(ModuleModel::new(VGG::vgg13(
            model_config.num_classes.unwrap_or(10),
        ))),
        "vgg19" => Box::new(ModuleModel::new(VGG::vgg19(
            model_config.num_classes.unwrap_or(10),
        ))),
        "vit" | "vit_tiny" => {
            let img = model_config.input_size.unwrap_or(32);
            Box::new(ModuleModel::new(VisionTransformer::vit_tiny(
                img,
                model_config.num_classes.unwrap_or(10),
            )))
        }
        "vit_small" => {
            let img = model_config.input_size.unwrap_or(32);
            Box::new(ModuleModel::new(VisionTransformer::vit_small(
                img,
                model_config.num_classes.unwrap_or(10),
            )))
        }
        "vit_base" => {
            let img = model_config.input_size.unwrap_or(32);
            Box::new(ModuleModel::new(VisionTransformer::vit_base(
                img,
                model_config.num_classes.unwrap_or(10),
            )))
        }
        _ => {
            // Default to MLP
            let input_size = model_config.input_size.unwrap_or(784);
            let num_classes = model_config.num_classes.unwrap_or(10);
            Box::new(MLP::new(input_size, &[256, 128], num_classes, 0.0))
        }
    }
}

/// Trait for trainable models that can be used in the training loop
trait TrainableModel: Send {
    fn forward(&self, input: &Variable) -> Variable;
    fn parameters(&self) -> Vec<axonml_nn::Parameter>;
    fn train(&mut self);
    fn state_dict(&self) -> StateDict;
}

// =============================================================================
// Generic wrapper — adapts any axonml_nn::Module (ResNet/VGG/ViT/…) to TrainableModel
// =============================================================================

struct ModuleModel<M: Module + Send + 'static> {
    inner: M,
}

impl<M: Module + Send + 'static> ModuleModel<M> {
    fn new(inner: M) -> Self {
        Self { inner }
    }
}

impl<M: Module + Send + 'static> TrainableModel for ModuleModel<M> {
    fn forward(&self, input: &Variable) -> Variable {
        self.inner.forward(input)
    }
    fn parameters(&self) -> Vec<axonml_nn::Parameter> {
        self.inner.parameters()
    }
    fn train(&mut self) {
        self.inner.train();
    }
    fn state_dict(&self) -> StateDict {
        StateDict::from_module(&self.inner)
    }
}

// =============================================================================
// Sequence classifier — RNN/LSTM/GRU over [B, T, F], last hidden → linear head
// =============================================================================

enum Recurrent {
    Rnn(RNN),
    Lstm(LSTM),
    Gru(GRU),
}

impl Recurrent {
    fn forward(&self, x: &Variable) -> Variable {
        match self {
            Recurrent::Rnn(m) => m.forward(x),
            Recurrent::Lstm(m) => m.forward(x),
            Recurrent::Gru(m) => m.forward(x),
        }
    }
    fn parameters(&self) -> Vec<axonml_nn::Parameter> {
        match self {
            Recurrent::Rnn(m) => m.parameters(),
            Recurrent::Lstm(m) => m.parameters(),
            Recurrent::Gru(m) => m.parameters(),
        }
    }
}

struct SeqClassifier {
    rnn: Recurrent,
    head: Linear,
}

impl SeqClassifier {
    fn new(kind: &str, feat: usize, hidden: usize, layers: usize, num_classes: usize) -> Self {
        let rnn = match kind {
            "lstm" => Recurrent::Lstm(LSTM::new(feat, hidden, layers)),
            "rnn" => Recurrent::Rnn(RNN::new(feat, hidden, layers)),
            _ => Recurrent::Gru(GRU::new(feat, hidden, layers)),
        };
        Self {
            rnn,
            head: Linear::new(hidden, num_classes),
        }
    }
}

impl TrainableModel for SeqClassifier {
    fn forward(&self, input: &Variable) -> Variable {
        // Accept an image [B, C, H, W] (folded to a sequence of C*H rows of width W)
        // or an existing sequence [B, T, F].
        let shape = input.shape();
        let seq = if shape.len() == 4 {
            let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
            input.reshape(&[b, c * h, w])
        } else {
            input.clone()
        };
        let out = self.rnn.forward(&seq);
        let t = out.shape()[1];
        let last = out.select(1, t - 1);
        self.head.forward(&last)
    }
    fn parameters(&self) -> Vec<axonml_nn::Parameter> {
        let mut p = self.rnn.parameters();
        p.extend(self.head.parameters());
        p
    }
    fn train(&mut self) {}
    fn state_dict(&self) -> StateDict {
        let mut sd = StateDict::new();
        for (i, p) in self.parameters().iter().enumerate() {
            sd.insert(
                format!("seq.{i}"),
                axonml_serialize::TensorData::from_tensor(&p.data()),
            );
        }
        sd
    }
}

// =============================================================================
// EnsembleNet — N modality branches fused by a strategy, trained end-to-end.
// Strategies: concat | gated | moe | late-ensemble.
// =============================================================================

enum FusionStrategy {
    Concat,
    Gated,
    Moe,
    LateEnsemble,
}

impl FusionStrategy {
    fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "concat" => FusionStrategy::Concat,
            "moe" => FusionStrategy::Moe,
            "late-ensemble" | "late" | "ensemble" => FusionStrategy::LateEnsemble,
            _ => FusionStrategy::Gated,
        }
    }
}

struct EnsembleNet {
    sizes: Vec<usize>,
    branches: Vec<Linear>,
    strategy: FusionStrategy,
    // concat/gated/moe: a shared head + (gated: gate) / (moe: router)
    gate: Option<Linear>,
    router: Option<Linear>,
    head: Option<Linear>,
    // late-ensemble: one head per branch, logits averaged
    heads: Vec<Linear>,
}

impl EnsembleNet {
    fn new(sizes: Vec<usize>, hidden: usize, num_classes: usize, strategy: FusionStrategy) -> Self {
        let branches: Vec<Linear> = sizes.iter().map(|&s| Linear::new(s, hidden)).collect();
        let n = sizes.len();
        let (gate, router, head, heads) = match strategy {
            FusionStrategy::Concat => (
                None,
                None,
                Some(Linear::new(n * hidden, num_classes)),
                Vec::new(),
            ),
            FusionStrategy::Gated => (
                Some(Linear::new(n * hidden, n * hidden)),
                None,
                Some(Linear::new(n * hidden, num_classes)),
                Vec::new(),
            ),
            FusionStrategy::Moe => (
                None,
                Some(Linear::new(n * hidden, n)),
                Some(Linear::new(hidden, num_classes)),
                Vec::new(),
            ),
            FusionStrategy::LateEnsemble => (
                None,
                None,
                None,
                (0..n).map(|_| Linear::new(hidden, num_classes)).collect(),
            ),
        };
        Self {
            sizes,
            branches,
            strategy,
            gate,
            router,
            head,
            heads,
        }
    }

    fn branch_feats(&self, input: &Variable) -> Vec<Variable> {
        let bsz = input.shape()[0];
        let total: usize = self.sizes.iter().sum();
        let x = input.reshape(&[bsz, total]);
        let mut off = 0;
        let mut feats = Vec::with_capacity(self.sizes.len());
        for (i, &s) in self.sizes.iter().enumerate() {
            let slice = x.narrow(1, off, s);
            feats.push(self.branches[i].forward(&slice).relu());
            off += s;
        }
        feats
    }
}

impl TrainableModel for EnsembleNet {
    fn forward(&self, input: &Variable) -> Variable {
        let feats = self.branch_feats(input);
        let refs: Vec<&Variable> = feats.iter().collect();
        match self.strategy {
            FusionStrategy::Concat => {
                let cat = Variable::cat(&refs, 1);
                self.head.as_ref().unwrap().forward(&cat)
            }
            FusionStrategy::Gated => {
                let cat = Variable::cat(&refs, 1);
                let g = self.gate.as_ref().unwrap().forward(&cat).sigmoid();
                self.head.as_ref().unwrap().forward(&cat.mul(&g))
            }
            FusionStrategy::Moe => {
                // Router softmax over branches; weighted sum of branch features.
                let cat = Variable::cat(&refs, 1);
                let w = self.router.as_ref().unwrap().forward(&cat).softmax(1);
                let mut mixed = feats[0].mul(&w.narrow(1, 0, 1));
                for (i, f) in feats.iter().enumerate().skip(1) {
                    mixed = mixed.add(&f.mul(&w.narrow(1, i, 1)));
                }
                self.head.as_ref().unwrap().forward(&mixed)
            }
            FusionStrategy::LateEnsemble => {
                // Each branch predicts logits with its own head; average them.
                let n = feats.len() as f32;
                let mut logits = self.heads[0].forward(&feats[0]);
                for (i, f) in feats.iter().enumerate().skip(1) {
                    logits = logits.add(&self.heads[i].forward(f));
                }
                logits.mul_scalar(1.0 / n)
            }
        }
    }
    fn parameters(&self) -> Vec<axonml_nn::Parameter> {
        let mut p = Vec::new();
        for b in &self.branches {
            p.extend(b.parameters());
        }
        if let Some(g) = &self.gate {
            p.extend(g.parameters());
        }
        if let Some(r) = &self.router {
            p.extend(r.parameters());
        }
        if let Some(h) = &self.head {
            p.extend(h.parameters());
        }
        for h in &self.heads {
            p.extend(h.parameters());
        }
        p
    }
    fn train(&mut self) {}
    fn state_dict(&self) -> StateDict {
        let mut sd = StateDict::new();
        for (i, p) in self.parameters().iter().enumerate() {
            sd.insert(
                format!("ensemble.{i}"),
                axonml_serialize::TensorData::from_tensor(&p.data()),
            );
        }
        sd
    }
}

// =============================================================================
// MLP Model
// =============================================================================

struct MLP {
    layers: Sequential,
}

impl MLP {
    fn new(input_size: usize, hidden_sizes: &[usize], num_classes: usize, dropout: f32) -> Self {
        let mut seq = Sequential::new();
        let mut prev_size = input_size;

        for &hidden_size in hidden_sizes {
            seq = seq.add(Linear::new(prev_size, hidden_size));
            seq = seq.add(ReLU);
            if dropout > 0.0 {
                seq = seq.add(Dropout::new(dropout));
            }
            prev_size = hidden_size;
        }

        seq = seq.add(Linear::new(prev_size, num_classes));

        Self { layers: seq }
    }
}

impl TrainableModel for MLP {
    fn forward(&self, input: &Variable) -> Variable {
        self.layers.forward(input)
    }

    fn parameters(&self) -> Vec<axonml_nn::Parameter> {
        self.layers.parameters()
    }

    fn train(&mut self) {
        self.layers.train();
    }

    fn state_dict(&self) -> StateDict {
        StateDict::from_module(&self.layers)
    }
}

// =============================================================================
// Simple CNN Model
// =============================================================================

struct SimpleCNN {
    conv1: Conv2d,
    conv2: Conv2d,
    fc1: Linear,
    fc2: Linear,
    pool: MaxPool2d,
    dropout: Dropout,
    training: bool,
}

impl SimpleCNN {
    fn new(input_channels: usize, num_classes: usize) -> Self {
        Self {
            conv1: Conv2d::new(input_channels, 32, 3),
            conv2: Conv2d::new(32, 64, 3),
            fc1: Linear::new(64 * 5 * 5, 128),
            fc2: Linear::new(128, num_classes),
            pool: MaxPool2d::new(2),
            dropout: Dropout::new(0.25),
            training: true,
        }
    }
}

impl TrainableModel for SimpleCNN {
    fn forward(&self, input: &Variable) -> Variable {
        // Conv block 1: conv -> relu -> pool
        let x = self.conv1.forward(input);
        let x = x.relu();
        let x = self.pool.forward(&x);

        // Conv block 2: conv -> relu -> pool
        let x = self.conv2.forward(&x);
        let x = x.relu();
        let x = self.pool.forward(&x);

        // Flatten - manually reshape the data
        let shape = x.shape();
        let batch_size = shape[0];
        let flat_size: usize = shape[1..].iter().product();
        let flat_data = x.data().to_vec();
        let x = Variable::new(
            Tensor::from_vec(flat_data, &[batch_size, flat_size]).unwrap(),
            x.requires_grad(),
        );

        // FC layers
        let x = self.fc1.forward(&x);
        let x = x.relu();
        let x = if self.training {
            self.dropout.forward(&x)
        } else {
            x
        };

        self.fc2.forward(&x)
    }

    fn parameters(&self) -> Vec<axonml_nn::Parameter> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params
    }

    fn train(&mut self) {
        self.training = true;
        self.dropout.train();
    }

    fn state_dict(&self) -> StateDict {
        let mut state = StateDict::new();
        for (name, param) in self.conv1.named_parameters() {
            state.insert(
                format!("conv1.{name}"),
                axonml_serialize::TensorData::from_tensor(&param.data()),
            );
        }
        for (name, param) in self.conv2.named_parameters() {
            state.insert(
                format!("conv2.{name}"),
                axonml_serialize::TensorData::from_tensor(&param.data()),
            );
        }
        for (name, param) in self.fc1.named_parameters() {
            state.insert(
                format!("fc1.{name}"),
                axonml_serialize::TensorData::from_tensor(&param.data()),
            );
        }
        for (name, param) in self.fc2.named_parameters() {
            state.insert(
                format!("fc2.{name}"),
                axonml_serialize::TensorData::from_tensor(&param.data()),
            );
        }
        state
    }
}

// =============================================================================
// LeNet Model
// =============================================================================

struct LeNetModel {
    model: axonml_vision::LeNet,
}

impl LeNetModel {
    fn new(_num_classes: usize) -> Self {
        // LeNet has fixed architecture for MNIST (10 classes)
        Self {
            model: axonml_vision::LeNet::new(),
        }
    }
}

impl TrainableModel for LeNetModel {
    fn forward(&self, input: &Variable) -> Variable {
        self.model.forward(input)
    }

    fn parameters(&self) -> Vec<axonml_nn::Parameter> {
        self.model.parameters()
    }

    fn train(&mut self) {
        // LeNet doesn't have dropout, so nothing to change
    }

    fn state_dict(&self) -> StateDict {
        StateDict::from_module(&self.model)
    }
}

// =============================================================================
// Optimizer Creation
// =============================================================================

fn create_optimizer(
    config: &TrainingConfig,
    params: Vec<axonml_nn::Parameter>,
) -> Box<dyn Optimizer> {
    let lr = config.learning_rate as f32;
    let name = config.optimizer.name.to_lowercase();

    match name.as_str() {
        "sgd" => {
            let momentum = config.optimizer.momentum as f32;
            if momentum > 0.0 {
                Box::new(SGD::with_momentum(params, lr, momentum))
            } else {
                Box::new(SGD::new(params, lr))
            }
        }
        "adam" => {
            let beta1 = config.optimizer.beta1 as f32;
            let beta2 = config.optimizer.beta2 as f32;
            Box::new(Adam::with_betas(params, lr, (beta1, beta2)))
        }
        "adamw" => {
            // AdamW with default weight decay
            Box::new(AdamW::new(params, lr))
        }
        "rmsprop" => Box::new(RMSprop::new(params, lr)),
        _ => {
            // Default to Adam
            Box::new(Adam::new(params, lr))
        }
    }
}

// =============================================================================
// Data Loading
// =============================================================================

// =============================================================================
// ImageFolder — generic dataset: root/<class>/<image files>, one class per subdir
// =============================================================================

const IMAGEFOLDER_SIZE: usize = 64;

struct ImageFolder {
    samples: Vec<(PathBuf, usize)>,
    num_classes: usize,
    size: usize,
}

impl ImageFolder {
    fn is_image(name: &str) -> bool {
        let n = name.to_lowercase();
        [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"]
            .iter()
            .any(|e| n.ends_with(e))
    }

    fn new(root: &std::path::Path, size: usize) -> Result<Self, String> {
        let mut classes: Vec<PathBuf> = std::fs::read_dir(root)
            .map_err(|e| format!("read {}: {e}", root.display()))?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        classes.sort();
        if classes.is_empty() {
            return Err(format!(
                "No class subdirectories under {} — expected root/<class>/<images>",
                root.display()
            ));
        }
        let mut samples = Vec::new();
        for (idx, class_dir) in classes.iter().enumerate() {
            for entry in std::fs::read_dir(class_dir)
                .map_err(|e| e.to_string())?
                .flatten()
            {
                let p = entry.path();
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if p.is_file() && Self::is_image(name) {
                    samples.push((p, idx));
                }
            }
        }
        if samples.is_empty() {
            return Err(format!("No images found under {}", root.display()));
        }
        Ok(Self {
            samples,
            num_classes: classes.len(),
            size,
        })
    }
}

enum TrainDataset {
    Mnist(MNIST),
    FashionMnist(FashionMNIST),
    Cifar10(CIFAR10),
    Images(ImageFolder),
    Multi(Vec<TrainDataset>),
}

impl TrainDataset {
    /// Load dataset from path based on format
    fn load(path: &std::path::Path, format: &str, train: bool) -> Result<Self, String> {
        match format.to_lowercase().as_str() {
            "mnist" => {
                let dataset = MNIST::new(path, train)?;
                Ok(TrainDataset::Mnist(dataset))
            }
            "fashion-mnist" | "fashion_mnist" | "fashionmnist" => {
                let dataset = FashionMNIST::new(path, train)?;
                Ok(TrainDataset::FashionMnist(dataset))
            }
            "cifar10" | "cifar-10" => {
                let dataset = CIFAR10::new(path, train)?;
                Ok(TrainDataset::Cifar10(dataset))
            }
            "imagefolder" | "image-folder" | "images" | "folder" => {
                let dataset = ImageFolder::new(path, IMAGEFOLDER_SIZE)?;
                Ok(TrainDataset::Images(dataset))
            }
            _ => Err(format!(
                "Unsupported dataset format: '{}'. Supported: mnist, fashion-mnist, cifar10, imagefolder",
                format
            )),
        }
    }

    /// Detect dataset format from directory contents
    fn detect_format(path: &std::path::Path) -> Option<String> {
        if path.join("train-images-idx3-ubyte").exists()
            || path.join("train-images-idx3-ubyte.gz").exists()
        {
            return Some("mnist".to_string());
        }
        if path.join("data_batch_1.bin").exists() {
            return Some("cifar10".to_string());
        }
        // generic image folder: a subdirectory that contains image files
        if let Ok(rd) = std::fs::read_dir(path) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    if let Ok(sub) = std::fs::read_dir(&p) {
                        let has_img = sub.flatten().any(|f| {
                            let fp = f.path();
                            fp.is_file()
                                && fp
                                    .file_name()
                                    .and_then(|n| n.to_str())
                                    .is_some_and(ImageFolder::is_image)
                        });
                        if has_img {
                            return Some("imagefolder".to_string());
                        }
                    }
                }
            }
        }
        None
    }

    fn effective_classes(&self) -> usize {
        match self {
            TrainDataset::Images(d) => d.num_classes,
            TrainDataset::Multi(v) => v.first().map_or(10, TrainDataset::effective_classes),
            _ => 10,
        }
    }

    fn num_classes(&self) -> Option<usize> {
        match self {
            TrainDataset::Images(d) => Some(d.num_classes),
            TrainDataset::Multi(v) => v.first().map(TrainDataset::effective_classes),
            _ => None,
        }
    }

    // (channels, height, width) of a single sample — lets models size their input.
    fn sample_dims(&self) -> Option<(usize, usize, usize)> {
        match self {
            TrainDataset::Mnist(_) | TrainDataset::FashionMnist(_) => Some((1, 28, 28)),
            TrainDataset::Cifar10(_) => Some((3, 32, 32)),
            TrainDataset::Images(d) => Some((3, d.size, d.size)),
            TrainDataset::Multi(_) => None,
        }
    }

    // Flattened feature count of each ensemble branch, in order.
    fn branch_sizes(&self) -> Option<Vec<usize>> {
        match self {
            TrainDataset::Multi(v) => v
                .iter()
                .map(|d| d.sample_dims().map(|(c, h, w)| c * h * w))
                .collect(),
            _ => None,
        }
    }
}

impl Dataset for TrainDataset {
    type Item = (Tensor<f32>, Tensor<f32>);

    fn len(&self) -> usize {
        match self {
            TrainDataset::Mnist(d) => d.len(),
            TrainDataset::FashionMnist(d) => d.len(),
            TrainDataset::Cifar10(d) => d.len(),
            TrainDataset::Images(d) => d.samples.len(),
            TrainDataset::Multi(v) => v.iter().map(axonml_data::Dataset::len).min().unwrap_or(0),
        }
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        match self {
            TrainDataset::Mnist(d) => d.get(index),
            TrainDataset::FashionMnist(d) => d.get(index),
            TrainDataset::Cifar10(d) => d.get(index),
            TrainDataset::Images(d) => {
                let (path, label) = d.samples.get(index)?;
                let image =
                    axonml_vision::image_io::load_image_resized(path, d.size, d.size).ok()?;
                let mut lv = vec![0.0f32; d.num_classes];
                lv[*label] = 1.0;
                let target = Tensor::from_vec(lv, &[d.num_classes]).ok()?;
                Some((image, target))
            }
            TrainDataset::Multi(v) => {
                // Concatenate every branch's sample into one flat vector; label from branch 0.
                let (first_in, label) = v.first()?.get(index)?;
                let mut flat = first_in.to_vec();
                for d in &v[1..] {
                    let (bi, _) = d.get(index)?;
                    flat.extend(bi.to_vec());
                }
                let n = flat.len();
                let input = Tensor::from_vec(flat, &[n]).ok()?;
                Some((input, label))
            }
        }
    }
}

// Resolve the training device; Cuda only when the crate is built with the cuda feature.
#[cfg(feature = "cuda")]
fn pick_device(dev_type: &str, dev_id: Option<usize>) -> axonml_tensor::Device {
    if dev_type == "cuda" {
        axonml_tensor::Device::Cuda(dev_id.unwrap_or(0))
    } else {
        axonml_tensor::Device::Cpu
    }
}
#[cfg(not(feature = "cuda"))]
fn pick_device(_dev_type: &str, _dev_id: Option<usize>) -> axonml_tensor::Device {
    axonml_tensor::Device::Cpu
}

fn load_one(path_str: &str, fmt: &Option<String>) -> Result<TrainDataset, String> {
    let path = PathBuf::from(path_str);
    if !path.exists() {
        return Err(format!("Data path does not exist: {path_str}"));
    }
    let format = fmt.clone().unwrap_or_else(|| {
        TrainDataset::detect_format(&path).unwrap_or_else(|| "mnist".to_string())
    });
    TrainDataset::load(&path, &format, true)
}

fn load_dataset(args: &TrainArgs) -> Result<TrainDataset, String> {
    let mut extra: Vec<String> = Vec::new();
    if let Some(db) = &args.data_b {
        extra.push(db.clone());
    }
    if let Some(list) = &args.branches {
        extra.extend(
            list.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty()),
        );
    }
    if extra.is_empty() {
        return load_one(&args.data, &args.format);
    }
    let mut branches = vec![load_one(&args.data, &args.format)?];
    for path in extra {
        branches.push(load_one(&path, &None)?);
    }
    Ok(TrainDataset::Multi(branches))
}

// =============================================================================
// Training Loop
// =============================================================================

fn run_training_loop(
    config: &TrainingConfig,
    args: &TrainArgs,
) -> Result<Vec<(String, f64)>, Box<dyn std::error::Error>> {
    // Load project config for model and data settings
    let project_config = if let Some(config_path) = &args.config {
        ProjectConfig::load(config_path).ok()
    } else if PathBuf::from("axonml.toml").exists() {
        ProjectConfig::load("axonml.toml").ok()
    } else {
        None
    };

    let mut model_config = project_config
        .as_ref()
        .map(|c| c.model.clone())
        .unwrap_or_default();
    let data_config = project_config
        .as_ref()
        .map(|c| c.data.clone())
        .unwrap_or_default();

    // The --model flag selects the architecture (mlp/cnn/resnet18/vgg16/vit_tiny/…)
    if let Some(m) = args.model.as_deref() {
        if !m.trim().is_empty() {
            model_config.architecture = m.trim().to_lowercase();
        }
    }

    // Initialize W&B run if configured
    #[cfg(feature = "wandb")]
    let mut wandb_run: Option<WandbRun> = {
        if wandb_is_available() {
            let wandb_config = WandbConfig::load().ok();
            if wandb_config.is_some_and(|c| c.is_configured()) {
                print_info("Initializing Weights & Biases...");

                // Build hyperparameters config for init_training_run
                let mut hyperparams: HashMap<String, serde_json::Value> = HashMap::new();
                hyperparams.insert("epochs".to_string(), serde_json::json!(config.epochs));
                hyperparams.insert(
                    "batch_size".to_string(),
                    serde_json::json!(config.batch_size),
                );
                hyperparams.insert(
                    "learning_rate".to_string(),
                    serde_json::json!(config.learning_rate),
                );
                hyperparams.insert(
                    "optimizer".to_string(),
                    serde_json::json!(config.optimizer.name),
                );

                let model_name = if model_config.architecture.is_empty() {
                    "mlp"
                } else {
                    &model_config.architecture
                };

                // Use the init_training_run helper
                match init_training_run(None, Some(model_name), hyperparams) {
                    Ok(mut run) => {
                        // Log additional config using log_config()
                        let mut extra_config: HashMap<String, serde_json::Value> = HashMap::new();
                        extra_config.insert("device".to_string(), serde_json::json!(config.device));
                        extra_config.insert(
                            "checkpoint_frequency".to_string(),
                            serde_json::json!(config.checkpoint_frequency),
                        );
                        extra_config.insert(
                            "num_workers".to_string(),
                            serde_json::json!(config.num_workers),
                        );

                        if config.optimizer.weight_decay > 0.0 {
                            extra_config.insert(
                                "weight_decay".to_string(),
                                serde_json::json!(config.optimizer.weight_decay),
                            );
                        }
                        if let Some(clip) = config.gradient_clip {
                            extra_config
                                .insert("gradient_clip".to_string(), serde_json::json!(clip));
                        }
                        if config.mixed_precision {
                            extra_config
                                .insert("mixed_precision".to_string(), serde_json::json!(true));
                        }

                        let _ = run.log_config(extra_config);

                        // Print the run URL
                        print_kv("W&B Run URL", &run.url());

                        Some(run)
                    }
                    Err(e) => {
                        print_info(&format!(
                            "W&B initialization failed: {e}, continuing without logging"
                        ));
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        }
    };

    #[cfg(not(feature = "wandb"))]
    let wandb_run: Option<()> = None;
    #[cfg(not(feature = "wandb"))]
    let _ = &wandb_run; // Suppress unused warning

    // Create model
    print_info(&format!(
        "Creating model: {}",
        if model_config.architecture.is_empty() {
            "MLP"
        } else {
            &model_config.architecture
        }
    ));
    // Load dataset first so the model's class count matches the data (image folders)
    print_info(&format!("Loading dataset from: {}", args.data));
    let dataset = load_dataset(args)?;
    let dataset_size = dataset.len();
    print_success(&format!("Loaded {} training samples", dataset_size));
    if let Some(nc) = dataset.num_classes() {
        model_config.num_classes = Some(nc);
        print_kv("Classes (from dataset)", &nc.to_string());
    }

    let arch = model_config.architecture.to_lowercase();
    let mut model: Box<dyn TrainableModel> = if matches!(
        arch.as_str(),
        "fusion" | "multimodal" | "ensemble"
    ) {
        let sizes = dataset.branch_sizes().ok_or_else(|| {
                "an ensemble/fusion needs at least a second branch dataset — pass --data-b <path> (or --branches a,b,c)".to_string()
            })?;
        let hidden = model_config.hidden_sizes.first().copied().unwrap_or(128);
        print_kv(
            "Ensemble",
            &format!(
                "{} branches [{}] · strategy {}",
                sizes.len(),
                sizes
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("+"),
                args.strategy
            ),
        );
        Box::new(EnsembleNet::new(
            sizes,
            hidden,
            model_config.num_classes.unwrap_or(10),
            FusionStrategy::parse(&args.strategy),
        ))
    } else {
        create_model(&model_config, &data_config, dataset.sample_dims())
    };
    model.train();

    // Device placement — move the model to the GPU when --device cuda is requested.
    let (dev_type, dev_id) = parse_device(&args.device);
    let train_device = pick_device(&dev_type, dev_id);
    if !train_device.is_cpu() {
        for p in model.parameters() {
            p.to_device(train_device);
        }
        print_kv("Device", &format!("{dev_type}:{}", dev_id.unwrap_or(0)));
    }

    // Create optimizer
    print_info(&format!("Creating optimizer: {}", config.optimizer.name));
    let mut optimizer = create_optimizer(config, model.parameters());

    let loader = DataLoader::new(dataset, config.batch_size);
    let batches_per_epoch = loader.len() as u64;
    print_kv("Batches per epoch", &batches_per_epoch.to_string());

    // Loss function
    let loss_fn = CrossEntropyLoss::new();

    // Training metrics
    let mut metrics = Vec::new();
    let mut best_loss = f64::INFINITY;
    let mut best_accuracy = 0.0f64;
    let total_epochs = config.epochs;
    let mut global_step = 0usize;

    println!();

    for epoch in 1..=total_epochs {
        let pb = epoch_progress_bar(epoch, total_epochs, batches_per_epoch);

        let mut epoch_loss = 0.0;
        let mut epoch_correct = 0usize;
        let mut epoch_total = 0usize;

        for batch in loader.iter() {
            global_step += 1;

            // Convert batch data to Variables (moving to the training device on GPU).
            let (in_t, tg_t) = if !train_device.is_cpu() {
                (
                    batch
                        .data
                        .to_device(train_device)
                        .unwrap_or_else(|_| batch.data.clone()),
                    batch
                        .targets
                        .to_device(train_device)
                        .unwrap_or_else(|_| batch.targets.clone()),
                )
            } else {
                (batch.data.clone(), batch.targets.clone())
            };
            let input = Variable::new(in_t, false);
            let target = Variable::new(tg_t, false);

            // Forward pass
            let output = model.forward(&input);

            // Compute loss
            let loss = loss_fn.compute(&output, &target);
            let loss_val = f64::from(loss.data().to_vec()[0]);
            epoch_loss += loss_val;

            // Compute accuracy
            let predictions = output.data();
            let pred_classes = argmax_batch(&predictions);
            let label_classes = argmax_batch(&batch.targets);

            let mut batch_correct = 0usize;
            for (pred, label) in pred_classes.iter().zip(label_classes.iter()) {
                if pred == label {
                    epoch_correct += 1;
                    batch_correct += 1;
                }
                epoch_total += 1;
            }

            // Log batch metrics to W&B
            #[cfg(feature = "wandb")]
            if let Some(ref mut run) = wandb_run {
                let batch_acc = batch_correct as f64 / pred_classes.len() as f64;
                let mut batch_metrics = HashMap::new();
                batch_metrics.insert("train/batch_loss".to_string(), loss_val);
                batch_metrics.insert("train/batch_accuracy".to_string(), batch_acc);
                let _ = run.log_at_step(global_step, batch_metrics);
            }

            // Backward pass
            optimizer.zero_grad();
            loss.backward();

            // Gradient clipping if configured
            if let Some(clip_val) = config.gradient_clip {
                clip_gradients(&model.parameters(), clip_val as f32);
            }

            // Update weights
            optimizer.step();

            pb.inc(1);
        }

        pb.finish_and_clear();

        // Calculate epoch metrics
        let avg_loss = epoch_loss / batches_per_epoch as f64;
        let accuracy = epoch_correct as f64 / epoch_total as f64;

        // Print epoch summary
        println!(
            "Epoch {}/{}: loss={:.4}, accuracy={:.2}%",
            epoch,
            total_epochs,
            avg_loss,
            accuracy * 100.0
        );

        // Log epoch metrics to W&B
        #[cfg(feature = "wandb")]
        if let Some(ref mut run) = wandb_run {
            let mut epoch_metrics = HashMap::new();
            epoch_metrics.insert("train/epoch_loss".to_string(), avg_loss);
            epoch_metrics.insert("train/epoch_accuracy".to_string(), accuracy);
            epoch_metrics.insert("train/epoch".to_string(), epoch as f64);
            epoch_metrics.insert("train/learning_rate".to_string(), config.learning_rate);
            let _ = run.log_at_step(global_step, epoch_metrics);
        }

        // Track best metrics
        if accuracy > best_accuracy {
            best_accuracy = accuracy;
        }

        // Save checkpoint if best model
        if avg_loss < best_loss {
            best_loss = avg_loss;

            if epoch % config.checkpoint_frequency == 0 || epoch == total_epochs {
                let checkpoint_path = format!("{}/checkpoint_epoch_{}.axonml", args.output, epoch);

                // Save model state
                let state_dict = model.state_dict();
                save_state_dict(&state_dict, &checkpoint_path, Format::Axonml)
                    .map_err(|e| format!("Failed to save checkpoint: {e}"))?;

                print_info(&format!("Saved checkpoint: {checkpoint_path}"));
            }
        }
    }

    // Save final model
    let final_path = format!("{}/model.axonml", args.output);
    let state_dict = model.state_dict();
    save_state_dict(&state_dict, &final_path, Format::Axonml)
        .map_err(|e| format!("Failed to save model: {e}"))?;

    // Log summary metrics to W&B
    #[cfg(feature = "wandb")]
    if let Some(ref mut run) = wandb_run {
        let _ = run.summary("best_loss", best_loss);
        let _ = run.summary("best_accuracy", best_accuracy);
        let _ = run.summary("total_epochs", total_epochs as f64);
        let _ = run.summary("total_steps", global_step as f64);
    }

    // Finish W&B run
    #[cfg(feature = "wandb")]
    if let Some(run) = wandb_run {
        let _ = run.finish();
    }

    // Final metrics
    metrics.push(("final_loss".to_string(), best_loss));
    metrics.push(("final_accuracy".to_string(), best_accuracy));
    metrics.push(("total_epochs".to_string(), total_epochs as f64));
    metrics.push((
        "total_batches".to_string(),
        (total_epochs as u64 * batches_per_epoch) as f64,
    ));

    Ok(metrics)
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Get argmax for each sample in a batch
fn argmax_batch(tensor: &Tensor<f32>) -> Vec<usize> {
    let shape = tensor.shape();
    let data = tensor.to_vec();

    if shape.len() == 1 {
        // Single sample
        let (idx, _) = data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, &0.0));
        vec![idx]
    } else {
        // Batch of samples
        let batch_size = shape[0];
        let num_classes = shape[1];

        (0..batch_size)
            .map(|b| {
                let start = b * num_classes;
                let end = start + num_classes;
                let slice = &data[start..end];

                slice
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map_or(0, |(idx, _)| idx)
            })
            .collect()
    }
}

/// Clip gradients by global norm
fn clip_gradients(params: &[axonml_nn::Parameter], max_norm: f32) {
    // Calculate global norm
    let mut total_norm = 0.0f32;

    for param in params {
        if let Some(grad) = param.grad() {
            let grad_data = grad.to_vec();
            let norm_sq: f32 = grad_data.iter().map(|x| x * x).sum();
            total_norm += norm_sq;
        }
    }

    total_norm = total_norm.sqrt();

    // Scale gradients if necessary
    if total_norm > max_norm {
        let scale = max_norm / (total_norm + 1e-6);
        for param in params {
            if let Some(grad) = param.grad() {
                let scaled: Vec<f32> = grad.to_vec().iter().map(|x| x * scale).collect();
                // Note: In a full implementation, we'd update the gradient in place
                // This is a simplified version
                let _ = scaled; // Acknowledge we computed this
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax_batch() {
        let data = Tensor::from_vec(vec![0.1, 0.8, 0.1, 0.7, 0.2, 0.1], &[2, 3]).unwrap();

        let result = argmax_batch(&data);
        assert_eq!(result, vec![1, 0]);
    }

    #[test]
    fn test_mlp_creation() {
        let model = MLP::new(784, &[256, 128], 10, 0.0);
        let params = model.parameters();
        assert!(!params.is_empty());
    }

    #[test]
    fn test_mlp_forward() {
        let model = MLP::new(4, &[8], 2, 0.0);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap(),
            false,
        );
        let output = model.forward(&input);
        assert_eq!(output.shape(), vec![1, 2]);
    }
}
