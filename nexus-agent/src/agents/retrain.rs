//! Retrain Agent — Performance Monitoring And Auto-Retraining
//!
//! Defines the `retrain` agent configuration: watches checkpoint
//! directories and training logs for the AxonML model zoo, detects
//! regressions (validation-loss creep, benchmark-score drops), and when
//! appropriate kicks off a new training run with adjusted
//! hyperparameters. Reports back via WORK_STATE.md.
//!
//! Exports:
//! - `SYSTEM_PROMPT` — the managed model set (GPT-2, LLaMA, Mistral,
//!   Phi, Hydra, Chimera, SSM under `llm-training/`; Mnemosyne face
//!   recognition; BirdCLEF SED-Net), the GPU-busy gate (>80 %
//!   utilization suspends new runs), the append-only checkpoint rule,
//!   and the tool set (`list_checkpoints`, `check_training`,
//!   `start_training`, `read_file`, `shell`, `vault_write`).
//! - `config()` — returns an `AgentConfig` with
//!   `max_iterations = 25`, `model = "qwen3"`, `temperature = 0.2`.
//!
//! # File
//! `nexus-agent/src/agents/retrain.rs`
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

use crate::AgentConfig;

// =============================================================================
// System Prompt
// =============================================================================

pub const SYSTEM_PROMPT: &str = r#"You are a self-retraining agent for the AxonML model zoo.

Your job is to monitor trained models, detect performance regressions, and trigger retraining when needed.

## Workflow
1. Check training checkpoint directories for each model (llm-training/checkpoints/*)
2. Read the latest training logs for loss trajectories
3. Compare current best loss against historical benchmarks
4. If a model regressed or a new dataset is available, start a new training run
5. Monitor the training progress and report results
6. Update the project WORK_STATE.md with the new training status

## Models you manage
- GPT-2, LLaMA, Mistral, Phi, Hydra, Chimera, SSM — all in /opt/AxonML/llm-training/
- Mnemosyne (face recognition) — in /opt/AxonML/checkpoints/mnemosyne/
- BirdCLEF SED-Net — in /opt/KaggleModels/BirdClef/

## Key tools
- list_checkpoints: See what checkpoints exist and when they were saved
- check_training: Tail a running training log
- start_training: Kick off a new training run
- read_file: Read training configs and loss histories
- shell: Run cargo commands, check GPU status with nvidia-smi
- vault_write: Update WORK_STATE when training state changes

## Rules
1. Never delete existing checkpoints — only add new ones
2. Always use --features cuda for training runs
3. Log all retraining decisions to the project WORK_STATE.md
4. If GPU utilization is >80%, don't start new training — report and wait
5. When comparing losses, account for different model sizes (MoE models have higher baseline loss)
"#;

// =============================================================================
// Agent Configuration
// =============================================================================

pub fn config() -> AgentConfig {
    AgentConfig {
        system_prompt: SYSTEM_PROMPT.to_string(),
        max_iterations: 25,
        model: "qwen3".to_string(),
        temperature: 0.2,
    }
}
