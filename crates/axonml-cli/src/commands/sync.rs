//! Server Sync Commands — `login`, `logout`, and `sync` Subcommands
//!
//! Implements the three `server-sync`-feature CLI subcommands that link a
//! local `axonml` CLI session to the AxonML server/webapp:
//! - `login(args)` — reads credentials from `LoginArgs` (or stdin prompts),
//!   pings the server via `ApiClient::is_server_available`, and stores
//!   credentials on success.
//! - `logout()` — clears the locally persisted credentials with
//!   `ApiClient::logout`.
//! - `sync(args)` — loads credentials, checks server reachability, and
//!   either prints status (`--status`) or performs a full sync of
//!   training runs, models, and datasets via the client list endpoints.
//!
//! All output uses `colored` for status glyphs/colors, and every function
//! returns `CliResult<()>` so errors propagate up to `main::run`.
//!
//! # File
//! `crates/axonml-cli/src/commands/sync.rs`
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

// =============================================================================
// Imports
// =============================================================================

use crate::api_client::ApiClient;
use crate::cli::{LoginArgs, SyncArgs};
use crate::error::CliResult;
use colored::Colorize;
use std::io::{self, Write};

// =============================================================================
// Login Command
// =============================================================================

/// Execute the login command
pub async fn login(args: &LoginArgs) -> CliResult<()> {
    let server_url = args.server.as_deref();
    let mut client = ApiClient::new(server_url);

    // Check if server is available
    if !client.is_server_available().await {
        let url = server_url.unwrap_or("http://localhost:3021");
        eprintln!(
            "{} Cannot connect to server at {}",
            "Error:".red().bold(),
            url
        );
        eprintln!("Make sure the AxonML server is running (axonml start)");
        return Ok(());
    }

    // -------------------------------------------------------------------------
    // Credential Prompting
    // -------------------------------------------------------------------------

    // Get username
    let username = if let Some(ref u) = args.username {
        u.clone()
    } else {
        print!("Username: ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        input.trim().to_string()
    };

    // Get password
    let password = if let Some(ref p) = args.password {
        p.clone()
    } else {
        print!("Password: ");
        io::stdout().flush()?;
        // In a real implementation, use rpassword for hidden input
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        input.trim().to_string()
    };

    // -------------------------------------------------------------------------
    // Login Attempt
    // -------------------------------------------------------------------------

    // Attempt login
    println!("{}", "Logging in...".dimmed());
    match client.login(&username, &password).await {
        Ok(user) => {
            println!(
                "{} Logged in as {} ({})",
                "✓".green().bold(),
                user.name.green(),
                user.email.dimmed()
            );
            println!("  Role: {}", user.role);
            println!();
            println!("CLI is now synced with the webapp.");
            println!("Training runs, models, and datasets will appear in both.");
        }
        Err(e) => {
            eprintln!("{} Login failed: {}", "✗".red().bold(), e);
        }
    }

    Ok(())
}

// =============================================================================
// Logout Command
// =============================================================================

/// Execute the logout command
#[allow(clippy::unused_async)]
pub async fn logout() -> CliResult<()> {
    match ApiClient::logout() {
        Ok(()) => {
            println!("{} Logged out successfully", "✓".green().bold());
            println!("CLI is no longer synced with the webapp.");
        }
        Err(e) => {
            eprintln!("{} Logout failed: {}", "✗".red().bold(), e);
        }
    }
    Ok(())
}

// =============================================================================
// Sync Command
// =============================================================================

/// Execute the sync command
pub async fn sync(args: &SyncArgs) -> CliResult<()> {
    // -------------------------------------------------------------------------
    // Credential and Server Checks
    // -------------------------------------------------------------------------

    // Try to load credentials
    let client = match ApiClient::load_credentials() {
        Ok(c) => c,
        Err(_) => {
            println!("{} Not logged in", "Status:".yellow().bold());
            println!();
            println!("Run {} to connect to the webapp", "axonml login".cyan());
            return Ok(());
        }
    };

    // Check server availability
    if !client.is_server_available().await {
        println!(
            "{} Logged in but server is not available",
            "Status:".yellow().bold()
        );
        if let Some(user) = client.current_user() {
            println!("  User: {} ({})", user.name, user.email);
        }
        println!();
        println!("Make sure the AxonML server is running (axonml start)");
        return Ok(());
    }

    // -------------------------------------------------------------------------
    // Status-Only or Full Sync
    // -------------------------------------------------------------------------

    if args.status {
        // Just show status
        println!("{} Connected and synced", "Status:".green().bold());
        if let Some(user) = client.current_user() {
            println!("  User: {} ({})", user.name.green(), user.email);
            println!("  Role: {}", user.role);
        }
        println!();

        // Show summary of synced data
        println!("{}", "Synced Data:".bold());

        match client.list_training_runs().await {
            Ok(runs) => println!("  Training runs: {}", runs.len()),
            Err(_) => println!("  Training runs: (error fetching)"),
        }

        match client.list_models().await {
            Ok(models) => println!("  Models: {}", models.len()),
            Err(_) => println!("  Models: (error fetching)"),
        }

        match client.list_datasets().await {
            Ok(datasets) => println!("  Datasets: {}", datasets.len()),
            Err(_) => println!("  Datasets: (error fetching)"),
        }
    } else {
        // Full sync
        println!("{}", "Syncing with server...".dimmed());

        let mut synced = 0;

        // Sync training runs
        match client.list_training_runs().await {
            Ok(runs) => {
                println!("  {} {} training runs", "✓".green(), runs.len());
                synced += runs.len();
            }
            Err(e) => println!("  {} Training runs: {}", "✗".red(), e),
        }

        // Sync models
        match client.list_models().await {
            Ok(models) => {
                println!("  {} {} models", "✓".green(), models.len());
                synced += models.len();
            }
            Err(e) => println!("  {} Models: {}", "✗".red(), e),
        }

        // Sync datasets
        match client.list_datasets().await {
            Ok(datasets) => {
                println!("  {} {} datasets", "✓".green(), datasets.len());
                synced += datasets.len();
            }
            Err(e) => println!("  {} Datasets: {}", "✗".red(), e),
        }

        println!();
        println!("{} Synced {} items", "✓".green().bold(), synced);
    }

    Ok(())
}
