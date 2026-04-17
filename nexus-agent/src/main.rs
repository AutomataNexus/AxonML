//! nexus-agent — CLI Entry Point
//!
//! Command-line driver for the nexus-agent framework. Parses flags with clap
//! (`--url` for the nexus-serve endpoint, `--model` to override the per-agent
//! default, `--anthropic` to route through the native Anthropic Messages API
//! instead of the OpenAI-compatible `/v1/chat/completions` path), then
//! dispatches to one of the specialized agent subcommands: knowledge,
//! retrain, fieldtech, research, orchestrator, ci-fixer, shield, code,
//! models, and health.
//!
//! Each agent subcommand loads a pre-tuned `AgentConfig` from `agents::*`,
//! registers all tools via `tools::register_all`, and invokes `react_loop`.
//! The `Health` command pings the backend's `health_check`, and `Models`
//! lists loaded models. The backend is selected at runtime as a trait object
//! (`AnthropicBackend` vs `LocalBackend`) so both paths share one dispatcher.
//!
//! Usage:
//!   nexus-agent knowledge "Scan /opt/AxonML and update its Obsidian vault docs"
//!   nexus-agent knowledge "Check all 15 project WORK_STATE files for staleness"
//!   nexus-agent --model gemma4 knowledge "Review NexusEdge_Rust CI status"
//!   nexus-agent --url http://localhost:11435 knowledge "..."
//!
//! # File
//! `nexus-agent/src/main.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 16, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use clap::Parser;
use nexus_agent::{
    agents,
    backend::{anthropic::AnthropicBackend, local::LocalBackend, LlmBackend},
    react_loop, tools, ToolRegistry,
};

// =============================================================================
// CLI Argument Parsing
// =============================================================================

#[derive(Parser)]
#[command(name = "nexus-agent")]
#[command(about = "Autonomous AI agent powered by local LLM inference")]
#[command(version)]
struct Args {
    /// nexus-serve URL (default: http://127.0.0.1:11435)
    #[arg(long, default_value = "http://127.0.0.1:11435")]
    url: String,

    /// Model to use. Overrides the agent's default model (each agent picks
    /// the model it was tuned for — `code` → `deepseek`, `knowledge` →
    /// `qwen3`, etc.). Pass this only to force a different model.
    #[arg(long, short)]
    model: Option<String>,

    /// Route through nexus-serve's Anthropic Messages API (`/v1/messages`)
    /// instead of the OpenAI-compatible `/v1/chat/completions`. Enables
    /// native `tool_use` / `tool_result` content blocks end-to-end. This
    /// is the recommended path for BitNet-based agents; the OpenAI path
    /// remains the default for legacy agents built against /v1/chat.
    #[arg(long)]
    anthropic: bool,

    /// Agent type
    #[command(subcommand)]
    agent: AgentCommand,
}

// -----------------------------------------------------------------------------
// Agent Subcommands
// -----------------------------------------------------------------------------

#[derive(clap::Subcommand)]
enum AgentCommand {
    /// Knowledge agent — reads codebases, maintains Obsidian vault documentation
    Knowledge { task: String },
    /// Retrain agent — monitors model performance, triggers retraining on regression
    Retrain { task: String },
    /// Field tech agent — HVAC fault detection, controller monitoring, tech alerting
    Fieldtech { task: String },
    /// Research agent — literature review, paper drafting, citation management
    Research { task: String },
    /// Orchestrator — training queue manager, GPU scheduling, job coordination
    Orchestrator { task: String },
    /// CI Fixer — invoked by the ticker ralph loop for test/assertion failures
    /// that `cargo fmt` and `cargo clippy --fix` can't resolve.
    CiFixer { task: String },
    /// Shield Agent — invoked by the security-ticker drill-down modal to
    /// investigate stat-chip events and propose user-acceptable fixes.
    Shield { task: String },
    /// Code agent — local agentic coder. Defaults to BitNet-2B via
    /// nexus-serve's Anthropic Messages API. Pair with `--anthropic`.
    Code { task: String },
    /// List available models on the nexus-serve backend
    Models,
    /// Health check — verify nexus-serve is reachable
    Health,
}

// =============================================================================
// Entry Point
// =============================================================================

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "nexus_agent=info".into()),
        )
        .init();

    let args = Args::parse();

    // Dispatch through a trait object so both backend types share one code
    // path. Clippy would prefer an `enum BackendChoice`, but the trait
    // object keeps `run_agent` monomorphisation-free and lets future
    // backends (cloud Anthropic, OpenAI cloud) slot in behind the same
    // --anthropic-style flags.
    let backend: Box<dyn LlmBackend> = if args.anthropic {
        Box::new(AnthropicBackend::with_url(&args.url))
    } else {
        Box::new(LocalBackend::with_url(&args.url))
    };

    match args.agent {
        AgentCommand::Health => {
            if backend.health_check().await {
                println!("nexus-serve is healthy at {}", args.url);
                let models: Vec<String> = backend.list_models().await.unwrap_or_default();
                if !models.is_empty() {
                    println!("Available models:");
                    for m in &models {
                        println!("  - {m}");
                    }
                }
            } else {
                eprintln!("Cannot reach nexus-serve at {}", args.url);
                std::process::exit(1);
            }
        }

        AgentCommand::Models => {
            let models = backend.list_models().await?;
            if models.is_empty() {
                println!("No models loaded on nexus-serve at {}", args.url);
            } else {
                println!("Models on {}:", args.url);
                for m in &models {
                    println!("  {m}");
                }
            }
        }

        AgentCommand::Knowledge { task } => {
            run_agent("knowledge", agents::knowledge::config(), args.model.as_deref(), &task, backend.as_ref()).await?;
        }
        AgentCommand::Retrain { task } => {
            run_agent("retrain", agents::retrain::config(), args.model.as_deref(), &task, backend.as_ref()).await?;
        }
        AgentCommand::Fieldtech { task } => {
            run_agent("fieldtech", agents::fieldtech::config(), args.model.as_deref(), &task, backend.as_ref()).await?;
        }
        AgentCommand::Research { task } => {
            run_agent("research", agents::research::config(), args.model.as_deref(), &task, backend.as_ref()).await?;
        }
        AgentCommand::Orchestrator { task } => {
            run_agent("orchestrator", agents::orchestrator::config(), args.model.as_deref(), &task, backend.as_ref()).await?;
        }
        AgentCommand::CiFixer { task } => {
            run_agent("ci-fixer", agents::ci_fixer::config(), args.model.as_deref(), &task, backend.as_ref()).await?;
        }
        AgentCommand::Shield { task } => {
            run_agent("shield", agents::shield::config(), args.model.as_deref(), &task, backend.as_ref()).await?;
        }
        AgentCommand::Code { task } => {
            run_agent("code", agents::code::config(), args.model.as_deref(), &task, backend.as_ref()).await?;
        }
    }

    Ok(())
}

// =============================================================================
// Agent Runner
// =============================================================================

async fn run_agent(
    name: &str,
    mut config: nexus_agent::AgentConfig,
    model_override: Option<&str>,
    task: &str,
    backend: &dyn nexus_agent::backend::LlmBackend,
) -> anyhow::Result<()> {
    if let Some(m) = model_override {
        config.model = m.to_string();
    }

    let mut registry = ToolRegistry::new();
    tools::register_all(&mut registry);

    println!("Agent: {name}");
    println!("Model: {}", config.model);
    println!("Task:  {task}");
    println!("---");

    let result = react_loop(backend, &registry, &config, task).await?;

    println!("---");
    println!("{result}");
    Ok(())
}
