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

mod config;
mod db;
mod auth;
mod api;
mod training;
mod inference;
mod email;

use api::{create_router, AppState};
use auth::JwtAuth;
use clap::Parser;
use config::Config;
use db::{schema::Schema, Database};
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::{info, error};
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

    // Validate configuration and show warnings
    for warning in config.validate_warnings() {
        tracing::warn!("{}", warning);
    }

    // Strict validation in production (when not using default secret)
    if std::env::var("AXONML_STRICT_CONFIG").is_ok() {
        config.validate()?;
    }

    info!("Data directory: {:?}", config.data_dir());
    info!("Models directory: {:?}", config.models_dir());
    info!("Runs directory: {:?}", config.runs_dir());

    // Connect to Aegis-DB
    info!("Connecting to Aegis-DB at {}", config.aegis_url());
    let db = match Database::new(&config.aegis).await {
        Ok(db) => {
            info!("Connected to Aegis-DB");
            db
        }
        Err(e) => {
            error!("Failed to connect to Aegis-DB: {}", e);
            error!("Make sure Aegis-DB is running on {}:{}", config.aegis.host, config.aegis.port);
            error!("Start it with: aegis start");
            return Err(e.into());
        }
    };

    // Initialize database schema
    if let Err(e) = Schema::init(&db).await {
        error!("Failed to initialize database schema: {}", e);
        // Continue anyway - tables might already exist
    }

    // Create default admin user if not exists
    let default_password_hash = auth::hash_password("admin")?;
    if let Err(e) = Schema::create_default_admin(&db, &default_password_hash).await {
        // This is expected if admin already exists
        tracing::debug!("Admin user creation: {}", e);
    }

    // Initialize JWT authentication
    let jwt = JwtAuth::new(&config.auth.jwt_secret, config.auth.jwt_expiry_hours);

    // Initialize email service
    let email_api_key = std::env::var("RESEND_API_KEY")
        .unwrap_or_else(|_| "re_cQM9wxDs_4ELeERKQ4yAGDEHc9wiTqHUp".to_string());
    let email = email::EmailService::new(email_api_key);

    // Initialize inference server
    let inference = inference::server::InferenceServer::new(inference::server::InferenceConfig::default());

    // Initialize training tracker for real-time metrics broadcasting
    let db_arc = Arc::new(db);
    let tracker = training::tracker::TrainingTracker::new(db_arc.clone());

    // Initialize model pool for managing loaded model instances (max 100 models, 5 minute idle timeout)
    let model_pool = inference::pool::ModelPool::new(100, 300);

    // Initialize inference metrics collector
    let inference_metrics = inference::metrics::InferenceMetrics::new();

    // Create application state
    let state = AppState {
        db: db_arc,
        jwt: Arc::new(jwt),
        config: Arc::new(config.clone()),
        email: Arc::new(email),
        inference: Arc::new(inference),
        tracker: Arc::new(tracker),
        model_pool: Arc::new(model_pool),
        inference_metrics: Arc::new(inference_metrics),
    };

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
    println!("║              AxonML Server v{}                       ║", env!("CARGO_PKG_VERSION"));
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
