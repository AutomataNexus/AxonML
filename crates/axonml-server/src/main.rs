//! AxonML Server - REST API for Machine Learning
//!
//! Provides HTTP API for training management, model registry, and inference.
//!
//! # Usage
//!
//! ```bash
//! # Start the server
//! axonml-server
//!
//! # With custom host and port
//! axonml-server --host 0.0.0.0 --port 3000
//! ```

mod api;
mod auth;
mod config;
mod db;
mod email;
mod inference;
mod llm;
mod secrets;
mod training;

use api::{create_router, AppState};
use auth::JwtAuth;
use clap::Parser;
use config::Config;
use db::{schema::Schema, Database};
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::{error, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// AxonML Server - REST API for Machine Learning
#[derive(Parser, Debug)]
#[command(name = "axonml-server")]
#[command(about = "AxonML REST API Server")]
#[command(version)]
struct Args {
    /// Host to bind to
    #[arg(short = 'H', long, default_value = "0.0.0.0")]
    host: String,

    /// Port to listen on
    #[arg(short, long, default_value = "3000")]
    port: u16,

    /// Path to config file
    #[arg(short, long)]
    config: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args = Args::parse();

    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "axonml_server=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Starting AxonML Server v{}", env!("CARGO_PKG_VERSION"));

    // Load configuration
    let config = if let Some(config_path) = args.config {
        Config::load_from_path(&std::path::PathBuf::from(config_path))?
    } else {
        Config::load()?
    };

    // Ensure directories exist
    config.ensure_directories()?;

    // SECURITY: Always validate configuration on startup
    config.validate()?;

    // Show informational warnings
    for warning in config.validate_warnings() {
        tracing::warn!("{}", warning);
    }

    info!("Data directory: {:?}", config.data_dir());
    info!("Models directory: {:?}", config.models_dir());
    info!("Runs directory: {:?}", config.runs_dir());

    // Initialize secrets manager
    // Priority: Vault -> Environment variables -> Config file
    let mut secrets_manager = secrets::SecretsManager::new();

    // Try Vault first (production)
    match secrets::vault::VaultBackend::from_env().await {
        Ok(Some(vault)) => {
            info!("Vault secrets backend enabled");
            let vault = Arc::new(vault);

            // Start background token renewal
            let vault_clone = vault.clone();
            vault_clone.start_token_renewal();

            secrets_manager = secrets_manager.with_backend(vault);
        }
        Ok(None) => {
            tracing::debug!("Vault not configured (VAULT_ADDR not set)");
        }
        Err(e) => {
            tracing::warn!("Failed to initialize Vault: {}", e);
        }
    }

    // Add environment variable backend (development)
    secrets_manager = secrets_manager.with_backend(Arc::new(secrets::env::EnvBackend::default()));

    info!(
        backends = ?secrets_manager.backend_names(),
        "Secrets manager initialized"
    );

    // Load secrets with config file fallback
    let jwt_secret = secrets_manager
        .get_secret(secrets::SecretKey::JWT_SECRET)
        .await
        .unwrap_or_else(|_| config.auth.jwt_secret.clone());

    let db_username = secrets_manager
        .get_secret(secrets::SecretKey::DB_USERNAME)
        .await
        .unwrap_or_else(|_| config.aegis.username.clone());

    let db_password = secrets_manager
        .get_secret(secrets::SecretKey::DB_PASSWORD)
        .await
        .unwrap_or_else(|_| config.aegis.password.clone());

    let email_api_key = secrets_manager
        .get_secret_optional(secrets::SecretKey::RESEND_API_KEY)
        .await
        .ok()
        .flatten();

    // Validate required secrets
    if jwt_secret.is_empty() {
        error!("JWT secret is required. Set via Vault, AXONML_JWT_SECRET env var, or config file.");
        return Err("JWT secret is required".into());
    }
    if jwt_secret.len() < 32 {
        error!("JWT secret must be at least 32 characters for security.");
        return Err("JWT secret must be at least 32 characters".into());
    }

    // Connect to Aegis-DB with secrets-loaded credentials
    info!("Connecting to Aegis-DB at {}", config.aegis_url());
    let mut aegis_config = config.aegis.clone();
    aegis_config.username = db_username;
    aegis_config.password = db_password;

    let db = match Database::new(&aegis_config).await {
        Ok(db) => {
            info!("Connected to Aegis-DB");
            db
        }
        Err(e) => {
            error!("Failed to connect to Aegis-DB: {}", e);
            error!(
                "Make sure Aegis-DB is running on {}:{}",
                config.aegis.host, config.aegis.port
            );
            error!("Start it with: aegis start");
            return Err(e.into());
        }
    };

    // Initialize database schema
    if let Err(e) = Schema::init(&db).await {
        error!("Failed to initialize database schema: {}", e);
        // Continue anyway - tables might already exist
    }

    // Create default admin user if not exists (with secure random password)
    // SECURITY: Generate a random password for the default admin
    let random_password: String = {
        use rand::Rng;
        const CHARSET: &[u8] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*";
        let mut rng = rand::thread_rng();
        (0..24)
            .map(|_| {
                let idx = rng.gen_range(0..CHARSET.len());
                CHARSET[idx] as char
            })
            .collect()
    };
    let default_password_hash = auth::hash_password(&random_password)?;
    match Schema::create_default_admin(&db, &default_password_hash).await {
        Ok(_) => {
            // Only show password if admin was just created
            tracing::warn!("========================================");
            tracing::warn!("DEFAULT ADMIN ACCOUNT CREATED");
            tracing::warn!("Email: admin@axonml.local");
            tracing::warn!("Password: {}", random_password);
            tracing::warn!("PLEASE CHANGE THIS PASSWORD IMMEDIATELY!");
            tracing::warn!("========================================");
        }
        Err(e) => {
            // This is expected if admin already exists
            tracing::debug!("Admin user creation: {}", e);
        }
    }

    // Create DevOps admin user (Andrew Jewell)
    let devops_password_hash = auth::hash_password("Invertedskynet2$")?;
    match Schema::create_devops_admin(&db, &devops_password_hash).await {
        Ok(_) => {
            info!("DevOps admin user ready (DevOps@AutomataNexus.com)");
        }
        Err(e) => {
            tracing::debug!("DevOps user creation: {}", e);
        }
    }

    // Initialize JWT authentication with secrets-loaded secret
    let jwt = JwtAuth::new(&jwt_secret, config.auth.jwt_expiry_hours);

    // Initialize email service with secrets-loaded API key
    let email = email::EmailService::new(email_api_key);
    if !email.is_configured() {
        tracing::warn!("Email API key not configured - email functionality will be disabled");
        tracing::warn!("Set via Vault (resend_api_key), AXONML_RESEND_API_KEY env var, or RESEND_API_KEY env var");
    }

    // Initialize inference server
    let inference =
        inference::server::InferenceServer::new(inference::server::InferenceConfig::default());

    // Initialize training tracker for real-time metrics broadcasting
    let db_arc = Arc::new(db);
    let tracker = Arc::new(training::tracker::TrainingTracker::new(db_arc.clone()));

    // Initialize training executor
    let executor = Arc::new(training::executor::TrainingExecutor::new(
        db_arc.clone(),
        tracker.clone(),
        config.models_dir(),
    ));

    // Initialize model pool for managing loaded model instances (max 100 models, 5 minute idle timeout)
    let model_pool = inference::pool::ModelPool::new(100, 300);

    // Initialize inference metrics collector
    let inference_metrics = inference::metrics::InferenceMetrics::new();

    // Initialize Ollama client for AI assistance
    let ollama = llm::OllamaClient::new();
    if ollama.is_available().await {
        info!(
            "Ollama LLM service available at {}",
            llm::DEFAULT_OLLAMA_URL
        );
    } else {
        info!("Ollama LLM service not available - AI assistance will be limited");
    }

    // Initialize notebook executor for running code cells
    let notebook_executor = Arc::new(training::notebook_executor::NotebookExecutor::default());
    info!("Notebook executor initialized");

    // Create application state
    let state = AppState {
        db: db_arc,
        jwt: Arc::new(jwt),
        config: Arc::new(config.clone()),
        email: Arc::new(email),
        inference: Arc::new(inference),
        tracker,
        executor,
        notebook_executor,
        model_pool: Arc::new(model_pool),
        inference_metrics: Arc::new(inference_metrics),
        metrics_history: Arc::new(tokio::sync::Mutex::new(api::system::SystemMetricsHistory {
            timestamps: Vec::new(),
            cpu_history: Vec::new(),
            memory_history: Vec::new(),
            disk_io_read: Vec::new(),
            disk_io_write: Vec::new(),
            network_rx: Vec::new(),
            network_tx: Vec::new(),
            gpu_utilization: Vec::new(),
        })),
        ollama: Arc::new(ollama),
    };

    // Spawn background task to collect system metrics
    let metrics_history_clone = state.metrics_history.clone();
    tokio::spawn(async move {
        use sysinfo::System;

        let mut sys = System::new_all();
        let max_history_points = 60; // Keep 60 seconds of history

        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

            sys.refresh_all();

            let cpu_usage = sys.global_cpu_usage() as f64;
            let total_memory = sys.total_memory() as f64;
            let used_memory = sys.used_memory() as f64;
            let memory_percent = if total_memory > 0.0 {
                (used_memory / total_memory) * 100.0
            } else {
                0.0
            };

            let timestamp = chrono::Utc::now().format("%H:%M:%S").to_string();

            let mut history = metrics_history_clone.lock().await;

            // Add new data points
            history.timestamps.push(timestamp);
            history.cpu_history.push(cpu_usage);
            history.memory_history.push(memory_percent);

            // For disk I/O and network, use sysinfo data or defaults
            history.disk_io_read.push(0.0); // Would need more complex tracking
            history.disk_io_write.push(0.0);
            history.network_rx.push(0.0);
            history.network_tx.push(0.0);

            // GPU utilization placeholder (would need NVML or similar)
            if history.gpu_utilization.is_empty() {
                history.gpu_utilization.push(Vec::new());
            }
            history.gpu_utilization[0].push(0.0);

            // Trim to max history length
            if history.timestamps.len() > max_history_points {
                history.timestamps.remove(0);
                history.cpu_history.remove(0);
                history.memory_history.remove(0);
                history.disk_io_read.remove(0);
                history.disk_io_write.remove(0);
                history.network_rx.remove(0);
                history.network_tx.remove(0);
                history.gpu_utilization[0].remove(0);
            }
        }
    });

    // Create router
    let app = create_router(state);

    // Determine bind address
    let host = if args.host != config.server.host {
        args.host
    } else {
        config.server.host
    };
    let port = if args.port != 3000 {
        args.port
    } else {
        config.server.port
    };

    let addr: SocketAddr = format!("{}:{}", host, port).parse()?;

    info!("AxonML Server listening on http://{}", addr);
    info!("API documentation: http://{}/api", addr);
    info!("Health check: http://{}/health", addr);

    // Print startup banner
    println!();
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║                                                           ║");
    println!("║     ██╗  ██╗ ██████╗ ███╗   ██╗███╗   ███╗██╗            ║");
    println!("║     ╚██╗██╔╝██╔═══██╗████╗  ██║████╗ ████║██║            ║");
    println!("║      ╚███╔╝ ██║   ██║██╔██╗ ██║██╔████╔██║██║            ║");
    println!("║      ██╔██╗ ██║   ██║██║╚██╗██║██║╚██╔╝██║██║            ║");
    println!("║     ██╔╝ ██╗╚██████╔╝██║ ╚████║██║ ╚═╝ ██║███████╗       ║");
    println!("║     ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝     ╚═╝╚══════╝       ║");
    println!("║                                                           ║");
    println!(
        "║              AxonML Server v{}                       ║",
        env!("CARGO_PKG_VERSION")
    );
    println!("║                                                           ║");
    println!("║     Server:  http://{}                           ║", addr);
    println!("║     Health:  http://{}/health                    ║", addr);
    println!("║                                                           ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();

    // Start server
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
