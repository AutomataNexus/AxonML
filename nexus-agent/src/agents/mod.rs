//! Pre-built agent configurations.
//!
//! Each agent is a system prompt + model choice + tool subset that specializes
//! the ReAct loop for a particular task domain.

pub mod ci_fixer;
pub mod fieldtech;
pub mod knowledge;
pub mod orchestrator;
pub mod research;
pub mod retrain;
