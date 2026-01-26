//! CLI - Command Line Interface Definitions
//!
//! Defines the CLI structure using clap derive macros.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use clap::{Parser, Subcommand};

// =============================================================================
// Main CLI Structure
// =============================================================================

/// Axonml - A high-performance ML framework for Rust
#[derive(Parser, Debug)]
#[command(
    name = "axonml",
    author = "AutomataNexus Development Team",
    version,
    about = "Axonml ML Framework CLI - Train, evaluate, and deploy ML models",
    long_about = "Axonml is a PyTorch-equivalent machine learning framework written in pure Rust.\n\n\
                  Use this CLI to manage projects, train models, evaluate performance, and deploy to production."
)]
pub struct Cli {
    /// Enable verbose output
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Suppress all output except errors
    #[arg(short, long, global = true)]
    pub quiet: bool,

    #[command(subcommand)]
    pub command: Commands,
}

// =============================================================================
// Subcommands
// =============================================================================

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Create a new Axonml project
    New(NewArgs),

    /// Initialize Axonml in an existing directory
    Init(InitArgs),

    /// Train a model from configuration
    Train(TrainArgs),

    /// Resume training from a checkpoint
    Resume(ResumeArgs),

    /// Evaluate model performance
    Eval(EvalArgs),

    /// Make predictions with a trained model
    Predict(PredictArgs),

    /// Convert models between formats
    Convert(ConvertArgs),

    /// Export models for deployment
    Export(ExportArgs),

    /// Inspect model architecture and parameters
    Inspect(InspectArgs),

    /// Generate comprehensive evaluation report
    Report(ReportArgs),

    /// Start an inference server
    #[cfg(feature = "serve")]
    Serve(ServeArgs),

    /// Configure Weights & Biases integration
    #[cfg(feature = "wandb")]
    Wandb(WandbArgs),

    /// Upload a model file to Axonml
    Upload(UploadArgs),

    /// Analyze and configure a dataset for training
    Data(DataArgs),

    /// Generate a Rust training project scaffold
    Scaffold(ScaffoldArgs),

    /// Create and manage model/dataset bundles
    Zip(ZipArgs),

    /// Rename models and datasets
    Rename(RenameArgs),

    /// Quantize models to reduce size and improve inference
    Quant(QuantArgs),

    /// Load models and datasets into workspace
    Load(LoadArgs),

    /// Analyze loaded models and datasets
    Analyze(AnalyzeArgs),

    /// Benchmark models and hardware
    Bench(BenchArgs),

    /// GPU detection and management
    Gpu(GpuArgs),

    /// Launch the terminal user interface
    Tui(TuiArgs),

    /// Kaggle dataset integration
    Kaggle(KaggleArgs),

    /// Pretrained model hub
    Hub(HubArgs),

    /// Dataset management (NexusConnectBridge)
    Dataset(DatasetArgs),

    /// Start the AxonML dashboard and API server
    Start(StartArgs),

    /// Stop running AxonML services
    Stop(StopArgs),

    /// Check status of AxonML services
    Status(StatusArgs),

    /// View logs from AxonML services
    Logs(LogsArgs),

    /// Login to AxonML server (sync with webapp)
    #[cfg(feature = "server-sync")]
    Login(LoginArgs),

    /// Logout from AxonML server
    #[cfg(feature = "server-sync")]
    Logout,

    /// Check sync status with server
    #[cfg(feature = "server-sync")]
    Sync(SyncArgs),
}

// =============================================================================
// Project Commands
// =============================================================================

/// Arguments for the `new` command
#[derive(Parser, Debug)]
pub struct NewArgs {
    /// Name of the project to create
    pub name: String,

    /// Project template to use
    #[arg(short, long, default_value = "default")]
    pub template: String,

    /// Skip git initialization
    #[arg(long)]
    pub no_git: bool,

    /// Directory to create the project in (defaults to current directory)
    #[arg(short, long)]
    pub path: Option<String>,
}

/// Arguments for the `init` command
#[derive(Parser, Debug)]
pub struct InitArgs {
    /// Project name (defaults to directory name)
    #[arg(short, long)]
    pub name: Option<String>,

    /// Skip git initialization
    #[arg(long)]
    pub no_git: bool,

    /// Force initialization even if files exist
    #[arg(short, long)]
    pub force: bool,
}

// =============================================================================
// Training Commands
// =============================================================================

/// Arguments for the `train` command
#[derive(Parser, Debug)]
pub struct TrainArgs {
    /// Path to configuration file (TOML or JSON)
    #[arg(short, long)]
    pub config: Option<String>,

    /// Path to model definition file
    #[arg(short, long)]
    pub model: Option<String>,

    /// Path to training data directory (required)
    #[arg(short, long)]
    pub data: String,

    /// Dataset format (mnist, fashion-mnist, cifar10). Auto-detected if not specified.
    #[arg(long)]
    pub format: Option<String>,

    /// Number of epochs to train
    #[arg(short, long)]
    pub epochs: Option<usize>,

    /// Batch size
    #[arg(short, long)]
    pub batch_size: Option<usize>,

    /// Learning rate
    #[arg(short, long)]
    pub lr: Option<f64>,

    /// Output directory for checkpoints
    #[arg(short, long, default_value = "./output")]
    pub output: String,

    /// Device to train on (cpu, cuda:0, etc.)
    #[arg(long, default_value = "cpu")]
    pub device: String,

    /// Random seed for reproducibility
    #[arg(long)]
    pub seed: Option<u64>,

    /// Number of workers for data loading
    #[arg(long, default_value = "4")]
    pub workers: usize,
}

/// Arguments for the `resume` command
#[derive(Parser, Debug)]
pub struct ResumeArgs {
    /// Path to checkpoint file
    pub checkpoint: String,

    /// Path to training data directory (required)
    #[arg(short, long)]
    pub data: String,

    /// Dataset format (mnist, fashion-mnist, cifar10). Auto-detected if not specified.
    #[arg(long)]
    pub format: Option<String>,

    /// Additional epochs to train
    #[arg(short, long)]
    pub epochs: Option<usize>,

    /// Override learning rate
    #[arg(short, long)]
    pub lr: Option<f64>,

    /// Batch size for training
    #[arg(short, long, default_value = "32")]
    pub batch_size: usize,

    /// Output directory for new checkpoints
    #[arg(short, long)]
    pub output: Option<String>,
}

// =============================================================================
// Evaluation Commands
// =============================================================================

/// Arguments for the `eval` command
#[derive(Parser, Debug)]
pub struct EvalArgs {
    /// Path to model file
    pub model: String,

    /// Path to evaluation data directory
    pub data: String,

    /// Dataset format (mnist, fashion-mnist, cifar10). Auto-detected if not specified.
    #[arg(long)]
    pub format: Option<String>,

    /// Batch size for evaluation
    #[arg(short, long, default_value = "32")]
    pub batch_size: usize,

    /// Device to evaluate on
    #[arg(long, default_value = "cpu")]
    pub device: String,

    /// Output file for metrics (JSON)
    #[arg(short, long)]
    pub output: Option<String>,

    /// Metrics to compute (comma-separated)
    #[arg(short, long, default_value = "accuracy,loss")]
    pub metrics: String,
}

/// Arguments for the `predict` command
#[derive(Parser, Debug)]
pub struct PredictArgs {
    /// Path to model file
    pub model: String,

    /// Input data (file path or JSON string)
    pub input: String,

    /// Output file for predictions
    #[arg(short, long)]
    pub output: Option<String>,

    /// Output format (json, csv, text)
    #[arg(short, long, default_value = "json")]
    pub format: String,

    /// Device to run inference on
    #[arg(long, default_value = "cpu")]
    pub device: String,

    /// Return top-k predictions for classification
    #[arg(short, long)]
    pub top_k: Option<usize>,
}

// =============================================================================
// Model Management Commands
// =============================================================================

/// Arguments for the `convert` command
#[derive(Parser, Debug)]
pub struct ConvertArgs {
    /// Input model file
    pub input: String,

    /// Output model file
    pub output: String,

    /// Input format (auto-detect if not specified)
    #[arg(long)]
    pub from: Option<String>,

    /// Output format (ferrite, onnx, safetensors)
    #[arg(long)]
    pub to: Option<String>,

    /// Optimize model during conversion
    #[arg(long)]
    pub optimize: bool,
}

/// Arguments for the `export` command
#[derive(Parser, Debug)]
pub struct ExportArgs {
    /// Input model file
    pub model: String,

    /// Output directory or file
    pub output: String,

    /// Export format (onnx, torchscript, safetensors)
    #[arg(short, long, default_value = "onnx")]
    pub format: String,

    /// Target platform (cpu, cuda, wasm, arm)
    #[arg(long, default_value = "cpu")]
    pub target: String,

    /// Enable quantization
    #[arg(long)]
    pub quantize: bool,

    /// Quantization precision (int8, fp16)
    #[arg(long, default_value = "fp16")]
    pub precision: String,
}

/// Arguments for the `inspect` command
#[derive(Parser, Debug)]
pub struct InspectArgs {
    /// Path to model file
    pub model: String,

    /// Show detailed layer information
    #[arg(short, long)]
    pub detailed: bool,

    /// Show parameter values (first N per layer)
    #[arg(long)]
    pub show_params: Option<usize>,

    /// Output format (text, json)
    #[arg(short, long, default_value = "text")]
    pub format: String,
}

// =============================================================================
// Report Command
// =============================================================================

/// Arguments for the `report` command
#[derive(Parser, Debug)]
pub struct ReportArgs {
    /// Path to model file
    pub model: String,

    /// Path to evaluation data directory (required)
    #[arg(short, long)]
    pub data: String,

    /// Dataset format (mnist, fashion-mnist, cifar10). Auto-detected if not specified.
    #[arg(long)]
    pub dataset_format: Option<String>,

    /// Batch size for evaluation
    #[arg(short, long, default_value = "32")]
    pub batch_size: usize,

    /// Device to evaluate on
    #[arg(long, default_value = "cpu")]
    pub device: String,

    /// Output directory for report files
    #[arg(short, long)]
    pub output: Option<String>,

    /// Output format (html, json, text, all)
    #[arg(short, long, default_value = "html")]
    pub format: String,

    /// Include confusion matrix in report
    #[arg(long, default_value = "true")]
    pub confusion_matrix: bool,

    /// Include loss curves in report
    #[arg(long, default_value = "true")]
    pub loss_curves: bool,

    /// Number of classes (auto-detected if not specified)
    #[arg(long)]
    pub num_classes: Option<usize>,

    /// Class names (comma-separated)
    #[arg(long)]
    pub class_names: Option<String>,

    /// Include per-class metrics breakdown
    #[arg(long, default_value = "true")]
    pub per_class_metrics: bool,

    /// Training history file (JSON) for loss curves
    #[arg(long)]
    pub history: Option<String>,
}

// =============================================================================
// Deployment Commands
// =============================================================================

/// Arguments for the `serve` command
#[cfg(feature = "serve")]
#[derive(Parser, Debug)]
pub struct ServeArgs {
    /// Path to model file
    pub model: String,

    /// Port to listen on
    #[arg(short, long, default_value = "8080")]
    pub port: u16,

    /// Host to bind to
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Number of worker threads
    #[arg(long, default_value = "4")]
    pub workers: usize,

    /// Enable batching for inference
    #[arg(long)]
    pub batch: bool,

    /// Maximum batch size
    #[arg(long, default_value = "32")]
    pub max_batch_size: usize,

    /// Request timeout in milliseconds
    #[arg(long, default_value = "30000")]
    pub timeout: u64,
}

// =============================================================================
// Weights & Biases Commands
// =============================================================================

/// Arguments for the `wandb` command
#[cfg(feature = "wandb")]
#[derive(Parser, Debug)]
pub struct WandbArgs {
    #[command(subcommand)]
    pub action: WandbSubcommand,
}

/// Subcommands for wandb
#[cfg(feature = "wandb")]
#[derive(Subcommand, Debug)]
pub enum WandbSubcommand {
    /// Log in to Weights & Biases (interactive)
    Login,

    /// Log out from Weights & Biases
    Logout,

    /// Show current W&B configuration status
    Status,

    /// Configure W&B settings
    Config(WandbConfigArgs),

    /// Enable W&B logging for training
    Enable,

    /// Disable W&B logging for training
    Disable,
}

/// Arguments for wandb config subcommand
#[cfg(feature = "wandb")]
#[derive(Parser, Debug)]
pub struct WandbConfigArgs {
    /// W&B API key
    #[arg(long)]
    pub api_key: Option<String>,

    /// W&B entity (username or team name)
    #[arg(short, long)]
    pub entity: Option<String>,

    /// Default project name
    #[arg(short, long)]
    pub project: Option<String>,

    /// W&B base URL (for self-hosted instances)
    #[arg(long)]
    pub base_url: Option<String>,

    /// Log frequency (every N steps)
    #[arg(long)]
    pub log_frequency: Option<usize>,

    /// Log model checkpoints to W&B
    #[arg(long)]
    pub log_checkpoints: Option<bool>,

    /// Log system metrics (GPU, CPU, memory)
    #[arg(long)]
    pub log_system_metrics: Option<bool>,
}

// =============================================================================
// Upload Command
// =============================================================================

/// Arguments for the `upload` command
#[derive(Parser, Debug)]
pub struct UploadArgs {
    /// Path to the model file to upload
    pub path: String,

    /// Model name (defaults to filename)
    #[arg(short, long)]
    pub name: Option<String>,

    /// Model description
    #[arg(short, long)]
    pub description: Option<String>,

    /// Model format (auto-detect if not specified)
    #[arg(short, long)]
    pub format: Option<String>,

    /// Model version tag
    #[arg(long, default_value = "latest")]
    pub version: String,

    /// Destination directory for storing the model
    #[arg(short, long, default_value = "./models")]
    pub output: String,

    /// Validate model structure after upload
    #[arg(long, default_value = "true")]
    pub validate: bool,

    /// Overwrite existing model with same name
    #[arg(long)]
    pub overwrite: bool,

    /// Extract and display model architecture info
    #[arg(long)]
    pub inspect: bool,
}

// =============================================================================
// Data Command
// =============================================================================

/// Arguments for the `data` command
#[derive(Parser, Debug)]
pub struct DataArgs {
    #[command(subcommand)]
    pub action: DataSubcommand,
}

/// Subcommands for data
#[derive(Subcommand, Debug)]
pub enum DataSubcommand {
    /// Upload and analyze a new dataset
    Upload(DataUploadArgs),

    /// Analyze an existing dataset
    Analyze(DataAnalyzeArgs),

    /// List available datasets
    List(DataListArgs),

    /// Generate preprocessing configuration
    Config(DataConfigArgs),

    /// Preview dataset samples
    Preview(DataPreviewArgs),

    /// Validate dataset structure
    Validate(DataValidateArgs),
}

/// Arguments for data upload subcommand
#[derive(Parser, Debug)]
pub struct DataUploadArgs {
    /// Path to dataset file or directory
    pub path: String,

    /// Dataset name
    #[arg(short, long)]
    pub name: Option<String>,

    /// Dataset type (image, tabular, text, audio)
    #[arg(short = 't', long)]
    pub data_type: Option<String>,

    /// Task type (classification, regression, segmentation, etc.)
    #[arg(long)]
    pub task: Option<String>,

    /// Destination directory
    #[arg(short, long, default_value = "./data")]
    pub output: String,

    /// Split ratio for train/val/test (e.g., "0.8,0.1,0.1")
    #[arg(long)]
    pub split: Option<String>,

    /// Automatically analyze after upload
    #[arg(long, default_value = "true")]
    pub analyze: bool,
}

/// Arguments for data analyze subcommand
#[derive(Parser, Debug)]
pub struct DataAnalyzeArgs {
    /// Path to dataset file or directory
    pub path: String,

    /// Dataset type hint (image, tabular, text, audio)
    #[arg(short = 't', long)]
    pub data_type: Option<String>,

    /// Maximum samples to analyze
    #[arg(long, default_value = "1000")]
    pub max_samples: usize,

    /// Output format (text, json)
    #[arg(short, long, default_value = "text")]
    pub format: String,

    /// Save analysis to file
    #[arg(short, long)]
    pub output: Option<String>,

    /// Include detailed statistics
    #[arg(long)]
    pub detailed: bool,

    /// Generate recommended training configuration
    #[arg(long, default_value = "true")]
    pub recommend: bool,
}

/// Arguments for data list subcommand
#[derive(Parser, Debug)]
pub struct DataListArgs {
    /// Directory to search for datasets
    #[arg(short, long, default_value = "./data")]
    pub path: String,

    /// Show detailed information
    #[arg(short, long)]
    pub detailed: bool,
}

/// Arguments for data config subcommand
#[derive(Parser, Debug)]
pub struct DataConfigArgs {
    /// Path to dataset
    pub path: String,

    /// Output config file path
    #[arg(short, long, default_value = "data_config.toml")]
    pub output: String,

    /// Config format (toml, json)
    #[arg(short, long, default_value = "toml")]
    pub format: String,
}

/// Arguments for data preview subcommand
#[derive(Parser, Debug)]
pub struct DataPreviewArgs {
    /// Path to dataset
    pub path: String,

    /// Number of samples to preview
    #[arg(short, long, default_value = "5")]
    pub num_samples: usize,

    /// Random sampling
    #[arg(long)]
    pub random: bool,
}

/// Arguments for data validate subcommand
#[derive(Parser, Debug)]
pub struct DataValidateArgs {
    /// Path to dataset
    pub path: String,

    /// Expected number of classes
    #[arg(long)]
    pub num_classes: Option<usize>,

    /// Expected input shape (e.g., "3,224,224")
    #[arg(long)]
    pub input_shape: Option<String>,

    /// Check for missing values
    #[arg(long, default_value = "true")]
    pub check_missing: bool,

    /// Check for class imbalance
    #[arg(long, default_value = "true")]
    pub check_balance: bool,
}

// =============================================================================
// Scaffold Command
// =============================================================================

/// Arguments for the `scaffold` command
#[derive(Parser, Debug)]
pub struct ScaffoldArgs {
    #[command(subcommand)]
    pub action: ScaffoldSubcommand,
}

/// Subcommands for scaffold
#[derive(Subcommand, Debug)]
pub enum ScaffoldSubcommand {
    /// Generate a new Rust training project
    Generate(ScaffoldGenerateArgs),

    /// List available project templates
    Templates(ScaffoldTemplatesArgs),
}

/// Arguments for scaffold generate subcommand
#[derive(Parser, Debug)]
pub struct ScaffoldGenerateArgs {
    /// Project name
    pub name: String,

    /// Output directory
    #[arg(short, long, default_value = ".")]
    pub output: String,

    /// Path to model file (optional, for fine-tuning)
    #[arg(short, long)]
    pub model: Option<String>,

    /// Path to dataset or data config
    #[arg(short, long)]
    pub data: Option<String>,

    /// Project template (training, evaluation, inference, full)
    #[arg(short, long, default_value = "training")]
    pub template: String,

    /// Task type (classification, regression, generation)
    #[arg(long)]
    pub task: Option<String>,

    /// Model architecture to use
    #[arg(long)]
    pub architecture: Option<String>,

    /// Include W&B integration
    #[arg(long)]
    pub wandb: bool,

    /// Include data augmentation in generated code
    #[arg(long)]
    pub augmentation: bool,

    /// Overwrite existing project directory
    #[arg(long)]
    pub overwrite: bool,
}

/// Arguments for scaffold templates subcommand
#[derive(Parser, Debug)]
pub struct ScaffoldTemplatesArgs {
    /// Show detailed template descriptions
    #[arg(short, long)]
    pub detailed: bool,
}

// =============================================================================
// Zip Command
// =============================================================================

/// Arguments for the `zip` command
#[derive(Parser, Debug)]
pub struct ZipArgs {
    #[command(subcommand)]
    pub action: ZipSubcommand,
}

/// Subcommands for zip
#[derive(Subcommand, Debug)]
pub enum ZipSubcommand {
    /// Create a bundle from model and/or dataset
    Create(ZipCreateArgs),

    /// Extract a bundle
    Extract(ZipExtractArgs),

    /// List contents of a bundle
    List(ZipListArgs),
}

/// Arguments for zip create subcommand
#[derive(Parser, Debug)]
pub struct ZipCreateArgs {
    /// Output bundle file path
    #[arg(short, long)]
    pub output: String,

    /// Path to model file to include
    #[arg(short, long)]
    pub model: Option<String>,

    /// Path to dataset directory to include
    #[arg(short, long)]
    pub data: Option<String>,

    /// Include config files (axonml.toml, etc.)
    #[arg(long)]
    pub include_config: bool,

    /// Overwrite existing bundle file
    #[arg(long)]
    pub overwrite: bool,
}

/// Arguments for zip extract subcommand
#[derive(Parser, Debug)]
pub struct ZipExtractArgs {
    /// Input bundle file path
    pub input: String,

    /// Output directory for extracted files
    #[arg(short, long, default_value = ".")]
    pub output: String,

    /// Print extracted files
    #[arg(short, long)]
    pub verbose: bool,
}

/// Arguments for zip list subcommand
#[derive(Parser, Debug)]
pub struct ZipListArgs {
    /// Input bundle file path
    pub input: String,

    /// Show detailed file information
    #[arg(short, long)]
    pub detailed: bool,
}

// =============================================================================
// Rename Command
// =============================================================================

/// Arguments for the `rename` command
#[derive(Parser, Debug)]
pub struct RenameArgs {
    #[command(subcommand)]
    pub action: RenameSubcommand,
}

/// Subcommands for rename
#[derive(Subcommand, Debug)]
pub enum RenameSubcommand {
    /// Rename a model file
    Model(RenameModelArgs),

    /// Rename a dataset
    Data(RenameDataArgs),
}

/// Arguments for rename model subcommand
#[derive(Parser, Debug)]
pub struct RenameModelArgs {
    /// Path to the model file
    pub path: String,

    /// New name for the model
    pub new_name: String,

    /// Force rename even if destination exists
    #[arg(short, long)]
    pub force: bool,
}

/// Arguments for rename data subcommand
#[derive(Parser, Debug)]
pub struct RenameDataArgs {
    /// Path to the dataset file or directory
    pub path: String,

    /// New name for the dataset
    pub new_name: String,

    /// Force rename even if destination exists
    #[arg(short, long)]
    pub force: bool,
}

// =============================================================================
// Quant Command
// =============================================================================

/// Arguments for the `quant` command
#[derive(Parser, Debug)]
pub struct QuantArgs {
    #[command(subcommand)]
    pub action: QuantSubcommand,
}

/// Subcommands for quant
#[derive(Subcommand, Debug)]
pub enum QuantSubcommand {
    /// Convert model to a quantized format
    Convert(QuantConvertArgs),

    /// Show quantization info about a model
    Info(QuantInfoArgs),

    /// Benchmark different quantization levels
    Benchmark(QuantBenchmarkArgs),

    /// List available quantization types
    List,
}

/// Arguments for quant convert subcommand
#[derive(Parser, Debug)]
pub struct QuantConvertArgs {
    /// Input model file (`PyTorch`, `SafeTensors`, ONNX, Axonml)
    pub input: String,

    /// Target quantization type (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, F16, F32)
    #[arg(short, long)]
    pub target: String,

    /// Output file path
    #[arg(short, long)]
    pub output: String,

    /// Source format override (auto-detect if not specified)
    #[arg(long)]
    pub from: Option<String>,

    /// Calibration dataset for better quantization (optional)
    #[arg(long)]
    pub calibration_data: Option<String>,

    /// Number of calibration samples
    #[arg(long, default_value = "100")]
    pub calibration_samples: usize,

    /// Overwrite existing output file
    #[arg(long)]
    pub overwrite: bool,
}

/// Arguments for quant info subcommand
#[derive(Parser, Debug)]
pub struct QuantInfoArgs {
    /// Input model file
    pub input: String,

    /// Show detailed layer-by-layer info
    #[arg(short, long)]
    pub detailed: bool,
}

/// Arguments for quant benchmark subcommand
#[derive(Parser, Debug)]
pub struct QuantBenchmarkArgs {
    /// Input model file
    pub input: String,

    /// Number of iterations for benchmarking
    #[arg(short, long, default_value = "10")]
    pub iterations: usize,

    /// Specific quantization types to benchmark (comma-separated)
    #[arg(short, long)]
    pub types: Option<String>,
}

// =============================================================================
// Load Command
// =============================================================================

/// Arguments for the `load` command
#[derive(Parser, Debug)]
pub struct LoadArgs {
    #[command(subcommand)]
    pub action: LoadSubcommand,
}

/// Subcommands for load
#[derive(Subcommand, Debug)]
pub enum LoadSubcommand {
    /// Load a model into the workspace
    Model(LoadModelArgs),

    /// Load a dataset into the workspace
    Data(LoadDataArgs),

    /// Load both a model and dataset
    Both(LoadBothArgs),

    /// Show current workspace status
    Status,

    /// Clear loaded items from workspace
    Clear,
}

/// Arguments for load model subcommand
#[derive(Parser, Debug)]
pub struct LoadModelArgs {
    /// Path to the model file
    pub path: String,

    /// Model name (defaults to filename)
    #[arg(short, long)]
    pub name: Option<String>,

    /// Model format override (auto-detect if not specified)
    #[arg(short, long)]
    pub format: Option<String>,
}

/// Arguments for load data subcommand
#[derive(Parser, Debug)]
pub struct LoadDataArgs {
    /// Path to the dataset file or directory
    pub path: String,

    /// Dataset name (defaults to directory name)
    #[arg(short, long)]
    pub name: Option<String>,

    /// Dataset type override (auto-detect if not specified)
    #[arg(short = 't', long)]
    pub data_type: Option<String>,
}

/// Arguments for load both subcommand
#[derive(Parser, Debug)]
pub struct LoadBothArgs {
    /// Path to the model file
    #[arg(short, long)]
    pub model: String,

    /// Path to the dataset file or directory
    #[arg(short, long)]
    pub data: String,
}

// =============================================================================
// Analyze Command
// =============================================================================

/// Arguments for the `analyze` command
#[derive(Parser, Debug)]
pub struct AnalyzeArgs {
    #[command(subcommand)]
    pub action: AnalyzeSubcommand,
}

/// Subcommands for analyze
#[derive(Subcommand, Debug)]
pub enum AnalyzeSubcommand {
    /// Analyze the loaded or specified model
    Model(AnalyzeModelArgs),

    /// Analyze the loaded or specified dataset
    Data(AnalyzeDataArgs),

    /// Analyze both model and dataset for compatibility
    Both(AnalyzeBothArgs),

    /// Generate a comprehensive analysis report
    Report(AnalyzeReportArgs),
}

/// Arguments for analyze model subcommand
#[derive(Parser, Debug)]
pub struct AnalyzeModelArgs {
    /// Path to model (uses loaded model if not specified)
    #[arg(short, long)]
    pub path: Option<String>,

    /// Show detailed layer-by-layer analysis
    #[arg(short, long)]
    pub detailed: bool,

    /// Output report file path
    #[arg(short, long)]
    pub output: Option<String>,

    /// Output format (json, text)
    #[arg(short, long, default_value = "text")]
    pub format: String,
}

/// Arguments for analyze data subcommand
#[derive(Parser, Debug)]
pub struct AnalyzeDataArgs {
    /// Path to dataset (uses loaded dataset if not specified)
    #[arg(short, long)]
    pub path: Option<String>,

    /// Show detailed statistics
    #[arg(short, long)]
    pub detailed: bool,

    /// Maximum samples to analyze
    #[arg(long, default_value = "10000")]
    pub max_samples: usize,

    /// Output report file path
    #[arg(short, long)]
    pub output: Option<String>,

    /// Output format (json, text)
    #[arg(short, long, default_value = "text")]
    pub format: String,
}

/// Arguments for analyze both subcommand
#[derive(Parser, Debug)]
pub struct AnalyzeBothArgs {
    /// Output report file path
    #[arg(short, long)]
    pub output: Option<String>,
}

/// Arguments for analyze report subcommand
#[derive(Parser, Debug)]
pub struct AnalyzeReportArgs {
    /// Output file path
    #[arg(short, long, default_value = "analysis_report.html")]
    pub output: String,

    /// Report format (html, json, md, text)
    #[arg(short, long, default_value = "html")]
    pub format: String,

    /// Include visualizations in report
    #[arg(long)]
    pub visualize: bool,
}

// =============================================================================
// Bench Command
// =============================================================================

/// Arguments for the `bench` command
#[derive(Parser, Debug)]
pub struct BenchArgs {
    #[command(subcommand)]
    pub action: BenchSubcommand,
}

/// Subcommands for bench
#[derive(Subcommand, Debug)]
pub enum BenchSubcommand {
    /// Benchmark a model's performance
    Model(BenchModelArgs),

    /// Benchmark inference at different batch sizes
    Inference(BenchInferenceArgs),

    /// Compare multiple models
    Compare(BenchCompareArgs),

    /// Benchmark hardware capabilities
    Hardware(BenchHardwareArgs),
}

/// Arguments for bench model subcommand
#[derive(Parser, Debug)]
pub struct BenchModelArgs {
    /// Input model file
    pub input: String,

    /// Number of benchmark iterations
    #[arg(short, long, default_value = "100")]
    pub iterations: usize,

    /// Number of warmup iterations
    #[arg(short, long, default_value = "10")]
    pub warmup: usize,

    /// Batch size for benchmarking
    #[arg(short, long, default_value = "1")]
    pub batch_size: usize,

    /// Device to benchmark on (cpu, cuda:0, etc.)
    #[arg(long, default_value = "cpu")]
    pub device: String,

    /// Output file for results (JSON)
    #[arg(short, long)]
    pub output: Option<String>,
}

/// Arguments for bench inference subcommand
#[derive(Parser, Debug)]
pub struct BenchInferenceArgs {
    /// Input model file
    pub model: String,

    /// Batch sizes to test (comma-separated)
    #[arg(short, long, default_value = "1,2,4,8,16,32")]
    pub batch_sizes: String,

    /// Number of iterations per batch size
    #[arg(short, long, default_value = "50")]
    pub iterations: usize,

    /// Number of warmup iterations
    #[arg(short, long, default_value = "5")]
    pub warmup: usize,

    /// Device to benchmark on
    #[arg(long, default_value = "cpu")]
    pub device: String,

    /// Output file for results (JSON)
    #[arg(short, long)]
    pub output: Option<String>,
}

/// Arguments for bench compare subcommand
#[derive(Parser, Debug)]
pub struct BenchCompareArgs {
    /// Model files to compare (comma-separated)
    pub models: String,

    /// Number of iterations
    #[arg(short, long, default_value = "50")]
    pub iterations: usize,

    /// Batch size for comparison
    #[arg(short, long, default_value = "1")]
    pub batch_size: usize,

    /// Device to benchmark on
    #[arg(long, default_value = "cpu")]
    pub device: String,

    /// Output file for results (JSON)
    #[arg(short, long)]
    pub output: Option<String>,
}

/// Arguments for bench hardware subcommand
#[derive(Parser, Debug)]
pub struct BenchHardwareArgs {
    /// Number of iterations for each test
    #[arg(short, long, default_value = "10")]
    pub iterations: usize,

    /// Output file for results (JSON)
    #[arg(short, long)]
    pub output: Option<String>,
}

// =============================================================================
// GPU Command
// =============================================================================

/// Arguments for the `gpu` command
#[derive(Parser, Debug)]
pub struct GpuArgs {
    #[command(subcommand)]
    pub action: GpuSubcommand,
}

/// Subcommands for gpu
#[derive(Subcommand, Debug)]
pub enum GpuSubcommand {
    /// List available GPU devices
    List,

    /// Show detailed GPU information
    Info,

    /// Select a GPU device for training
    Select(GpuSelectArgs),

    /// Benchmark GPU performance
    Bench(GpuBenchArgs),

    /// Show GPU memory usage
    Memory,

    /// Show current GPU status
    Status,
}

/// Arguments for gpu select subcommand
#[derive(Parser, Debug)]
pub struct GpuSelectArgs {
    /// Device to select (e.g., 0, cuda:0, auto)
    pub device: String,

    /// Save selection persistently
    #[arg(short, long)]
    pub persistent: bool,
}

/// Arguments for gpu bench subcommand
#[derive(Parser, Debug)]
pub struct GpuBenchArgs {
    /// Specific device to benchmark
    #[arg(short, long)]
    pub device: Option<String>,

    /// Benchmark all available GPUs
    #[arg(short, long)]
    pub all: bool,

    /// Number of iterations
    #[arg(short, long, default_value = "10")]
    pub iterations: usize,
}

// =============================================================================
// TUI Command
// =============================================================================

/// Arguments for the `tui` command
#[derive(Parser, Debug)]
pub struct TuiArgs {
    /// Path to a model file to load on startup
    #[arg(short, long)]
    pub model: Option<String>,

    /// Path to a dataset directory to load on startup
    #[arg(short, long)]
    pub data: Option<String>,
}

// =============================================================================
// Kaggle Command
// =============================================================================

/// Arguments for the `kaggle` command
#[derive(Parser, Debug)]
pub struct KaggleArgs {
    #[command(subcommand)]
    pub action: KaggleSubcommand,
}

/// Subcommands for kaggle
#[derive(Subcommand, Debug)]
pub enum KaggleSubcommand {
    /// Configure Kaggle API credentials
    Login(KaggleLoginArgs),

    /// Check Kaggle configuration status
    Status,

    /// Search for datasets on Kaggle
    Search(KaggleSearchArgs),

    /// Download a dataset from Kaggle
    Download(KaggleDownloadArgs),

    /// List downloaded Kaggle datasets
    List,
}

/// Arguments for kaggle login subcommand
#[derive(Parser, Debug)]
pub struct KaggleLoginArgs {
    /// Kaggle username
    #[arg(short, long)]
    pub username: String,

    /// Kaggle API key
    #[arg(short, long)]
    pub key: String,
}

/// Arguments for kaggle search subcommand
#[derive(Parser, Debug)]
pub struct KaggleSearchArgs {
    /// Search query
    pub query: String,

    /// Maximum number of results
    #[arg(short, long, default_value = "10")]
    pub limit: usize,
}

/// Arguments for kaggle download subcommand
#[derive(Parser, Debug)]
pub struct KaggleDownloadArgs {
    /// Dataset reference (e.g., "username/dataset-name")
    pub dataset: String,

    /// Output directory
    #[arg(short, long)]
    pub output: Option<String>,
}

// =============================================================================
// Hub Command
// =============================================================================

/// Arguments for the `hub` command
#[derive(Parser, Debug)]
pub struct HubArgs {
    #[command(subcommand)]
    pub action: HubSubcommand,
}

/// Subcommands for hub
#[derive(Subcommand, Debug)]
pub enum HubSubcommand {
    /// List available pretrained models
    List,

    /// Show information about a specific model
    Info(HubInfoArgs),

    /// Download pretrained weights
    Download(HubDownloadArgs),

    /// List cached models
    Cached,

    /// Clear cached models
    Clear(HubClearArgs),
}

/// Arguments for hub info subcommand
#[derive(Parser, Debug)]
pub struct HubInfoArgs {
    /// Model name (e.g., "resnet18", "vgg16")
    pub model: String,
}

/// Arguments for hub download subcommand
#[derive(Parser, Debug)]
pub struct HubDownloadArgs {
    /// Model name to download
    pub model: String,

    /// Force re-download even if cached
    #[arg(short, long)]
    pub force: bool,
}

/// Arguments for hub clear subcommand
#[derive(Parser, Debug)]
pub struct HubClearArgs {
    /// Specific model to clear (clears all if not specified)
    pub model: Option<String>,
}

// =============================================================================
// Dataset Command
// =============================================================================

/// Arguments for the `dataset` command
#[derive(Parser, Debug)]
pub struct DatasetArgs {
    #[command(subcommand)]
    pub action: DatasetSubcommand,
}

/// Subcommands for dataset
#[derive(Subcommand, Debug)]
pub enum DatasetSubcommand {
    /// List available datasets
    List(DatasetListArgs),

    /// Show information about a specific dataset
    Info(DatasetInfoArgs),

    /// Search for datasets
    Search(DatasetSearchArgs),

    /// Download a dataset
    Download(DatasetDownloadArgs),

    /// List available data sources
    Sources,
}

/// Arguments for dataset list subcommand
#[derive(Parser, Debug)]
pub struct DatasetListArgs {
    /// Filter by source (builtin, kaggle, uci, etc.)
    #[arg(short, long)]
    pub source: Option<String>,
}

/// Arguments for dataset info subcommand
#[derive(Parser, Debug)]
pub struct DatasetInfoArgs {
    /// Dataset ID (e.g., "mnist", "cifar-10")
    pub dataset: String,
}

/// Arguments for dataset search subcommand
#[derive(Parser, Debug)]
pub struct DatasetSearchArgs {
    /// Search query
    pub query: String,

    /// Filter by source (kaggle, uci, data.gov, all)
    #[arg(short, long)]
    pub source: Option<String>,

    /// Maximum number of results
    #[arg(short, long, default_value = "20")]
    pub limit: usize,
}

/// Arguments for dataset download subcommand
#[derive(Parser, Debug)]
pub struct DatasetDownloadArgs {
    /// Dataset ID to download
    pub dataset: String,

    /// Output directory
    #[arg(short, long)]
    pub output: Option<String>,
}

// =============================================================================
// Dashboard/Server Commands
// =============================================================================

/// Arguments for the `start` command
#[derive(Parser, Debug)]
pub struct StartArgs {
    /// Start only the API server (backend)
    #[arg(long, conflicts_with = "dashboard")]
    pub server: bool,

    /// Start only the dashboard (frontend)
    #[arg(long, conflicts_with = "server")]
    pub dashboard: bool,

    /// API server port
    #[arg(long, default_value = "3000")]
    pub port: u16,

    /// Dashboard port
    #[arg(long, default_value = "8080")]
    pub dashboard_port: u16,

    /// Host to bind to
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Run in foreground (don't daemonize)
    #[arg(short, long)]
    pub foreground: bool,

    /// Path to config file
    #[arg(short, long)]
    pub config: Option<String>,
}

/// Arguments for the `stop` command
#[derive(Parser, Debug)]
pub struct StopArgs {
    /// Stop only the API server
    #[arg(long, conflicts_with = "dashboard")]
    pub server: bool,

    /// Stop only the dashboard
    #[arg(long, conflicts_with = "server")]
    pub dashboard: bool,

    /// Force stop (SIGKILL instead of SIGTERM)
    #[arg(short, long)]
    pub force: bool,
}

/// Arguments for the `status` command
#[derive(Parser, Debug)]
pub struct StatusArgs {
    /// Show detailed status information
    #[arg(short, long)]
    pub detailed: bool,

    /// Output format (text, json)
    #[arg(short, long, default_value = "text")]
    pub format: String,
}

/// Arguments for the `logs` command
#[derive(Parser, Debug)]
pub struct LogsArgs {
    /// Show only server logs
    #[arg(long, conflicts_with = "dashboard")]
    pub server: bool,

    /// Show only dashboard logs
    #[arg(long, conflicts_with = "server")]
    pub dashboard: bool,

    /// Number of lines to show
    #[arg(short, long, default_value = "50")]
    pub lines: usize,

    /// Follow logs in real-time
    #[arg(short, long)]
    pub follow: bool,

    /// Filter logs by level (error, warn, info, debug, trace)
    #[arg(long)]
    pub level: Option<String>,
}

// =============================================================================
// Server Sync Commands
// =============================================================================

/// Arguments for the `login` command
#[cfg(feature = "server-sync")]
#[derive(Parser, Debug)]
pub struct LoginArgs {
    /// Username or email
    #[arg(short, long)]
    pub username: Option<String>,

    /// Password (will prompt if not provided)
    #[arg(short, long)]
    pub password: Option<String>,

    /// Server URL (defaults to http://localhost:3021)
    #[arg(short, long)]
    pub server: Option<String>,
}

/// Arguments for the `sync` command
#[cfg(feature = "server-sync")]
#[derive(Parser, Debug)]
pub struct SyncArgs {
    /// Check sync status only
    #[arg(long)]
    pub status: bool,

    /// Force re-sync all data
    #[arg(long)]
    pub force: bool,
}
