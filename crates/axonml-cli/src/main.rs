//! Axonml CLI - Command Line Interface for Axonml ML Framework
//!
//! The main entry point for the Axonml command-line tool.
//!
//! # Commands
//! - `axonml new` - Create a new Axonml project
//! - `axonml init` - Initialize Axonml in an existing directory
//! - `axonml train` - Train a model from configuration
//! - `axonml eval` - Evaluate model performance
//! - `axonml predict` - Make predictions with a trained model
//! - `axonml convert` - Convert models between formats
//! - `axonml export` - Export models for deployment
//! - `axonml report` - Generate comprehensive evaluation reports
//! - `axonml serve` - Start an inference server
//! - `axonml inspect` - Inspect model architecture and parameters
//! - `axonml wandb` - Configure Weights & Biases integration
//! - `axonml upload` - Upload model files
//! - `axonml data` - Analyze and manage datasets
//! - `axonml scaffold` - Generate Rust training projects
//! - `axonml zip` - Create model/dataset bundles
//! - `axonml rename` - Rename models and datasets
//! - `axonml quant` - Quantize models (Q4, Q8, F16, etc.)
//! - `axonml load` - Load models and datasets into workspace
//! - `axonml analyze` - Comprehensive analysis and reporting
//! - `axonml bench` - Benchmark models and hardware
//! - `axonml gpu` - GPU detection and management
//! - `axonml tui` - Launch the terminal user interface
//! - `axonml kaggle` - Kaggle dataset integration
//! - `axonml hub` - Pretrained model hub
//! - `axonml dataset` - Dataset management (NexusConnectBridge)
//! - `axonml start` - Start dashboard and API server
//! - `axonml stop` - Stop running services
//! - `axonml status` - Check service status
//! - `axonml logs` - View service logs
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
// CLI-specific allowances
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::ptr_arg)]
#![allow(clippy::unused_self)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::single_match_else)]
#![allow(clippy::format_push_string)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::erasing_op)]
#![allow(clippy::use_debug)]
#![allow(clippy::case_sensitive_file_extension_comparisons)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::struct_field_names)]
#![allow(clippy::manual_let_else)]
#![allow(clippy::explicit_iter_loop)]
#![allow(clippy::default_trait_access)]
#![allow(clippy::only_used_in_recursion)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::ref_option)]
#![allow(clippy::if_not_else)]
#![allow(clippy::panic_in_result_fn)]
#![allow(clippy::indexing_slicing)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::multiple_bound_locations)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::manual_assert)]
#![allow(clippy::unnecessary_debug_formatting)]
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::assigning_clones)]
#![allow(clippy::needless_range_loop)]

use clap::Parser;
use colored::Colorize;

#[cfg(feature = "server-sync")]
mod api_client;
mod cli;
mod commands;
mod config;
mod error;

use cli::{Cli, Commands};
use error::CliResult;

fn main() {
    if let Err(e) = run() {
        eprintln!("{} {}", "error:".red().bold(), e);
        std::process::exit(1);
    }
}

fn run() -> CliResult<()> {
    let cli = Cli::parse();

    // Set verbosity level
    if cli.verbose {
        // Would set up logging here
    }

    match cli.command {
        Commands::New(args) => commands::new::execute(args),
        Commands::Init(args) => commands::init::execute(args),
        Commands::Train(args) => commands::train::execute(args),
        Commands::Resume(args) => commands::resume::execute(args),
        Commands::Eval(args) => commands::eval::execute(args),
        Commands::Predict(args) => commands::predict::execute(args),
        Commands::Convert(args) => commands::convert::execute(args),
        Commands::Export(args) => commands::export::execute(args),
        Commands::Inspect(args) => commands::inspect::execute(args),
        Commands::Report(args) => commands::report::execute(args),
        #[cfg(feature = "wandb")]
        Commands::Wandb(args) => commands::wandb::execute(args),
        #[cfg(feature = "serve")]
        Commands::Serve(args) => commands::serve::execute(args),
        Commands::Upload(args) => commands::upload::execute(args),
        Commands::Data(args) => commands::data::execute(args),
        Commands::Scaffold(args) => commands::scaffold::execute(args),
        Commands::Zip(args) => commands::zip::execute(args),
        Commands::Rename(args) => commands::rename::execute(args),
        Commands::Quant(args) => commands::quant::execute(args),
        Commands::Load(args) => commands::load::execute(args),
        Commands::Analyze(args) => commands::analyze::execute(args),
        Commands::Bench(args) => commands::bench::execute(args),
        Commands::Gpu(args) => commands::gpu::execute(args),
        Commands::Tui(args) => commands::tui::execute(args),
        Commands::Kaggle(args) => execute_kaggle(args),
        Commands::Hub(args) => execute_hub(args),
        Commands::Dataset(args) => execute_dataset(args),
        Commands::Start(args) => commands::dashboard::execute_start(args),
        Commands::Stop(args) => commands::dashboard::execute_stop(args),
        Commands::Status(args) => commands::dashboard::execute_status(args),
        Commands::Logs(args) => commands::dashboard::execute_logs(args),
        #[cfg(feature = "server-sync")]
        Commands::Login(args) => execute_async(commands::sync::login(&args)),
        #[cfg(feature = "server-sync")]
        Commands::Logout => execute_async(commands::sync::logout()),
        #[cfg(feature = "server-sync")]
        Commands::Sync(args) => execute_async(commands::sync::sync(&args)),
    }
}

#[cfg(feature = "server-sync")]
fn execute_async<F: std::future::Future<Output = CliResult<()>>>(future: F) -> CliResult<()> {
    tokio::runtime::Runtime::new()
        .map_err(|e| error::CliError::Other(e.to_string()))?
        .block_on(future)
}

fn execute_kaggle(args: cli::KaggleArgs) -> CliResult<()> {
    use cli::KaggleSubcommand;

    match args.action {
        KaggleSubcommand::Login(login_args) => {
            commands::kaggle::execute_login(&login_args.username, &login_args.key)
                .map_err(|e| error::CliError::Other(e))
        }
        KaggleSubcommand::Status => {
            commands::kaggle::execute_status().map_err(|e| error::CliError::Other(e))
        }
        KaggleSubcommand::Search(search_args) => {
            commands::kaggle::execute_search(&search_args.query, search_args.limit)
                .map_err(|e| error::CliError::Other(e))
        }
        KaggleSubcommand::Download(dl_args) => {
            commands::kaggle::execute_download(&dl_args.dataset, dl_args.output.as_deref())
                .map_err(|e| error::CliError::Other(e))
        }
        KaggleSubcommand::List => {
            commands::kaggle::execute_list().map_err(|e| error::CliError::Other(e))
        }
    }
}

fn execute_hub(args: cli::HubArgs) -> CliResult<()> {
    use cli::HubSubcommand;

    match args.action {
        HubSubcommand::List => commands::hub::execute_list().map_err(|e| error::CliError::Other(e)),
        HubSubcommand::Info(info_args) => {
            commands::hub::execute_info(&info_args.model).map_err(|e| error::CliError::Other(e))
        }
        HubSubcommand::Download(dl_args) => {
            commands::hub::execute_download(&dl_args.model, dl_args.force)
                .map_err(|e| error::CliError::Other(e))
        }
        HubSubcommand::Cached => {
            commands::hub::execute_cached().map_err(|e| error::CliError::Other(e))
        }
        HubSubcommand::Clear(clear_args) => {
            commands::hub::execute_clear(clear_args.model.as_deref())
                .map_err(|e| error::CliError::Other(e))
        }
    }
}

fn execute_dataset(args: cli::DatasetArgs) -> CliResult<()> {
    use cli::DatasetSubcommand;

    match args.action {
        DatasetSubcommand::List(list_args) => {
            commands::dataset::execute_list(list_args.source.as_deref())
                .map_err(|e| error::CliError::Other(e))
        }
        DatasetSubcommand::Info(info_args) => commands::dataset::execute_info(&info_args.dataset)
            .map_err(|e| error::CliError::Other(e)),
        DatasetSubcommand::Search(search_args) => commands::dataset::execute_search(
            &search_args.query,
            search_args.source.as_deref(),
            search_args.limit,
        )
        .map_err(|e| error::CliError::Other(e)),
        DatasetSubcommand::Download(dl_args) => {
            commands::dataset::execute_download(&dl_args.dataset, dl_args.output.as_deref())
                .map_err(|e| error::CliError::Other(e))
        }
        DatasetSubcommand::Sources => {
            commands::dataset::execute_sources().map_err(|e| error::CliError::Other(e))
        }
    }
}
