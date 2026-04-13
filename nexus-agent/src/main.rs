//! nexus-agent — CLI entry point
//!
//! Usage:
//!   nexus-agent knowledge "Scan /opt/AxonML and update its Obsidian vault docs"
//!   nexus-agent knowledge "Check all 15 project WORK_STATE files for staleness"
//!   nexus-agent --model gemma4 knowledge "Review NexusEdge_Rust CI status"
//!   nexus-agent --url http://localhost:11435 knowledge "..."

use clap::Parser;
use nexus_agent::{
    agents, backend::local::LocalBackend, backend::LlmBackend, react_loop, tools, ToolRegistry,
};

#[derive(Parser)]
#[command(name = "nexus-agent")]
#[command(about = "Autonomous AI agent powered by local LLM inference")]
#[command(version)]
struct Args {
    /// nexus-serve URL (default: http://127.0.0.1:11435)
    #[arg(long, default_value = "http://127.0.0.1:11435")]
    url: String,

    /// Model to use (default: qwen3)
    #[arg(long, short, default_value = "qwen3")]
    model: String,

    /// Agent type
    #[command(subcommand)]
    agent: AgentCommand,
}

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
    /// List available models on the nexus-serve backend
    Models,
    /// Health check — verify nexus-serve is reachable
    Health,
}

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

    // Create backend
    let backend = LocalBackend::with_url(&args.url);

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
            run_agent("knowledge", agents::knowledge::config(), &args.model, &task, &backend).await?;
        }
        AgentCommand::Retrain { task } => {
            run_agent("retrain", agents::retrain::config(), &args.model, &task, &backend).await?;
        }
        AgentCommand::Fieldtech { task } => {
            run_agent("fieldtech", agents::fieldtech::config(), &args.model, &task, &backend).await?;
        }
        AgentCommand::Research { task } => {
            run_agent("research", agents::research::config(), &args.model, &task, &backend).await?;
        }
        AgentCommand::Orchestrator { task } => {
            run_agent("orchestrator", agents::orchestrator::config(), &args.model, &task, &backend).await?;
        }
        AgentCommand::CiFixer { task } => {
            run_agent("ci-fixer", agents::ci_fixer::config(), &args.model, &task, &backend).await?;
        }
    }

    Ok(())
}

async fn run_agent(
    name: &str,
    mut config: nexus_agent::AgentConfig,
    model_override: &str,
    task: &str,
    backend: &impl nexus_agent::backend::LlmBackend,
) -> anyhow::Result<()> {
    config.model = model_override.to_string();

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
