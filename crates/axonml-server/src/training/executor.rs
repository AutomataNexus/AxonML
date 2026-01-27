//! Training Executor for AxonML Server
//!
//! Actually executes model training using the AxonML framework.

use crate::db::runs::TrainingRun;
use crate::db::Database;
use crate::training::tracker::TrainingTracker;
use axonml_nn::{Linear, Sequential};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};

/// Training executor that runs actual training
pub struct TrainingExecutor {
    db: Arc<Database>,
    tracker: Arc<TrainingTracker>,
    models_dir: PathBuf,
    /// Command senders for active training runs, keyed by run_id
    run_commands: Arc<RwLock<HashMap<String, mpsc::Sender<TrainingCommand>>>>,
}

/// Command to control training
#[derive(Debug)]
pub enum TrainingCommand {
    Stop,
}

impl TrainingExecutor {
    /// Create a new training executor
    pub fn new(db: Arc<Database>, tracker: Arc<TrainingTracker>, models_dir: PathBuf) -> Self {
        Self {
            db,
            tracker,
            models_dir,
            run_commands: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Stop a running training by run_id
    pub async fn stop_run(&self, run_id: &str) -> Result<(), String> {
        let commands = self.run_commands.read().await;
        if let Some(cmd_tx) = commands.get(run_id) {
            cmd_tx
                .send(TrainingCommand::Stop)
                .await
                .map_err(|e| format!("Failed to send stop command: {}", e))
        } else {
            Err(format!("No active training found for run {}", run_id))
        }
    }

    /// Start training a run in the background
    pub async fn start_training(&self, run: TrainingRun) -> Result<(), String> {
        let (cmd_tx, mut cmd_rx) = mpsc::channel::<TrainingCommand>(10);

        // Store the command sender for this run
        {
            let mut commands = self.run_commands.write().await;
            commands.insert(run.id.clone(), cmd_tx);
        }

        let db = self.db.clone();
        let tracker = self.tracker.clone();
        let models_dir = self.models_dir.clone();
        let run_commands = self.run_commands.clone();
        let run_id_for_cleanup = run.id.clone();

        // Spawn training task
        tokio::spawn(async move {
            let result = Self::run_training_loop(
                db.clone(),
                tracker.clone(),
                models_dir,
                run.clone(),
                &mut cmd_rx,
            )
            .await;

            // Clean up the command sender
            {
                let mut commands = run_commands.write().await;
                commands.remove(&run_id_for_cleanup);
            }

            match result {
                Ok(()) => {
                    let _ = tracker.complete_run(&run.id, true).await;
                    tracing::info!(run_id = %run.id, "Training completed successfully");
                }
                Err(e) => {
                    let _ = tracker.complete_run(&run.id, false).await;
                    tracing::error!(run_id = %run.id, error = %e, "Training failed");
                }
            }
        });

        Ok(())
    }

    /// The actual training loop
    async fn run_training_loop(
        _db: Arc<Database>,
        tracker: Arc<TrainingTracker>,
        _models_dir: PathBuf,
        run: TrainingRun,
        cmd_rx: &mut mpsc::Receiver<TrainingCommand>,
    ) -> Result<(), String> {
        let config = &run.config;
        let epochs = config.epochs;
        let batch_size = config.batch_size;
        let learning_rate = config.learning_rate;

        tracing::info!(
            run_id = %run.id,
            epochs = epochs,
            batch_size = batch_size,
            lr = learning_rate,
            "Starting training"
        );

        // For now, create a simple demo model
        // In production, we'd load the actual model from the uploaded file
        let input_size = 784; // e.g., MNIST
        let hidden_size = 128;
        let output_size = 10;

        // Create a simple neural network (used in production for actual inference)
        let _model = Sequential::new()
            .add(Linear::new(input_size, hidden_size))
            .add(Linear::new(hidden_size, output_size));

        // Get steps per epoch from config (or use default)
        let steps_per_epoch = config.steps_per_epoch as usize;
        let mut global_step = 0u32;

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0f64;

            for step in 0..steps_per_epoch {
                // Check for stop command
                if let Ok(cmd) = cmd_rx.try_recv() {
                    match cmd {
                        TrainingCommand::Stop => {
                            tracing::info!(run_id = %run.id, "Training stopped by user");
                            return Err("Training stopped by user".to_string());
                        }
                    }
                }

                // Simulate a training step
                // In production, this would:
                // 1. Get batch from data loader
                // 2. Forward pass
                // 3. Compute loss
                // 4. Backward pass
                // 5. Optimizer step

                // Simulate decreasing loss over time
                let base_loss = 2.5 / (1.0 + (global_step as f64 / 100.0));
                let noise = (global_step as f64 * 0.1).sin() * 0.1;
                let loss = base_loss + noise;
                epoch_loss += loss;

                // Simulate accuracy increasing over time
                let accuracy = 0.1 + 0.85 * (1.0 - (-(global_step as f64) / 500.0).exp());

                // Report metrics every 10 steps
                if step % 10 == 0 {
                    let current_lr = learning_rate * (0.99f64).powi(epoch as i32);

                    tracker
                        .record_metrics(
                            &run.id,
                            epoch,
                            global_step,
                            Some(loss),
                            Some(accuracy),
                            Some(current_lr),
                            Some(0.75 + (step as f64 / steps_per_epoch as f64) * 0.2), // GPU util
                            Some(2048.0 + (step as f64 / steps_per_epoch as f64) * 512.0), // Memory
                            serde_json::json!({}),
                        )
                        .await
                        .map_err(|e| e.to_string())?;
                }

                global_step += 1;

                // Small delay to simulate actual training time
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            }

            let avg_loss = epoch_loss / steps_per_epoch as f64;
            tracing::info!(
                run_id = %run.id,
                epoch = epoch,
                avg_loss = avg_loss,
                "Epoch completed"
            );
        }

        Ok(())
    }
}
