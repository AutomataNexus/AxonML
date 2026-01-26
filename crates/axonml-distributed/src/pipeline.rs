//! Pipeline Parallelism
//!
//! Implements pipeline parallelism for training large models across multiple devices.
//! Splits the model into stages, with each stage running on a different device.
//!
//! # Example
//! ```rust,ignore
//! use axonml_distributed::pipeline::{Pipeline, PipelineSchedule};
//!
//! // Split model into 4 stages across 4 GPUs
//! let pipeline = Pipeline::new(stages, process_group)
//!     .schedule(PipelineSchedule::GPipe)
//!     .num_microbatches(4);
//!
//! let output = pipeline.forward(&input);
//! ```
//!
//! @version 0.1.0

use crate::process_group::ProcessGroup;
use axonml_autograd::Variable;
use axonml_nn::{Module, Parameter};
use axonml_tensor::Tensor;

// =============================================================================
// Pipeline Schedule
// =============================================================================

/// Pipeline execution schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineSchedule {
    /// GPipe: Fill-drain schedule with synchronized updates
    GPipe,
    /// 1F1B: One forward, one backward schedule for memory efficiency
    OneFOneBSchedule,
    /// Interleaved 1F1B for better efficiency
    InterleavedOneFOneB,
}

impl Default for PipelineSchedule {
    fn default() -> Self {
        Self::OneFOneBSchedule
    }
}

// =============================================================================
// Pipeline Stage
// =============================================================================

/// A stage in the pipeline.
pub struct PipelineStage<M: Module> {
    /// The module for this stage
    module: M,
    /// Stage index (0 = first stage)
    stage_id: usize,
    /// Device/rank this stage runs on
    device_rank: usize,
}

impl<M: Module> PipelineStage<M> {
    /// Creates a new pipeline stage.
    pub fn new(module: M, stage_id: usize, device_rank: usize) -> Self {
        Self {
            module,
            stage_id,
            device_rank,
        }
    }

    /// Returns the stage ID.
    pub fn stage_id(&self) -> usize {
        self.stage_id
    }

    /// Returns the device rank.
    pub fn device_rank(&self) -> usize {
        self.device_rank
    }

    /// Forward pass for this stage.
    pub fn forward(&self, input: &Variable) -> Variable {
        self.module.forward(input)
    }
}

impl<M: Module> Module for PipelineStage<M> {
    fn forward(&self, input: &Variable) -> Variable {
        self.module.forward(input)
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.module.parameters()
    }

    fn train(&mut self) {
        self.module.train();
    }

    fn eval(&mut self) {
        self.module.eval();
    }

    fn is_training(&self) -> bool {
        self.module.is_training()
    }
}

// =============================================================================
// Pipeline
// =============================================================================

/// Pipeline parallel wrapper for distributed training.
///
/// Splits model computation across multiple stages, with each stage
/// potentially running on a different device/rank.
pub struct Pipeline<M: Module> {
    /// Pipeline stages
    stages: Vec<PipelineStage<M>>,
    /// Process group for communication
    process_group: ProcessGroup,
    /// Pipeline schedule
    schedule: PipelineSchedule,
    /// Number of microbatches
    num_microbatches: usize,
    /// Current rank's stage index (for future use with multi-GPU)
    #[allow(dead_code)]
    local_stage: usize,
}

impl<M: Module + Clone> Pipeline<M> {
    /// Creates a new pipeline from a list of modules.
    pub fn from_modules(modules: Vec<M>, process_group: ProcessGroup) -> Self {
        let world_size = process_group.world_size();
        let rank = process_group.rank();

        let stages: Vec<PipelineStage<M>> = modules
            .into_iter()
            .enumerate()
            .map(|(i, m)| PipelineStage::new(m, i, i % world_size))
            .collect();

        let local_stage = stages
            .iter()
            .position(|s| s.device_rank == rank)
            .unwrap_or(0);

        Self {
            stages,
            process_group,
            schedule: PipelineSchedule::default(),
            num_microbatches: 1,
            local_stage,
        }
    }

    /// Builder: set pipeline schedule.
    pub fn schedule(mut self, schedule: PipelineSchedule) -> Self {
        self.schedule = schedule;
        self
    }

    /// Builder: set number of microbatches.
    pub fn num_microbatches(mut self, num: usize) -> Self {
        self.num_microbatches = num.max(1);
        self
    }

    /// Returns number of stages.
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }

    /// Returns the current schedule.
    pub fn get_schedule(&self) -> PipelineSchedule {
        self.schedule
    }

    /// Forward pass through the pipeline.
    ///
    /// For the first stage, takes input and forwards to next stage.
    /// For intermediate stages, receives from previous, forwards to next.
    /// For the last stage, receives from previous, returns output.
    pub fn forward(&self, input: &Variable) -> Variable {
        match self.schedule {
            PipelineSchedule::GPipe => self.forward_gpipe(input),
            PipelineSchedule::OneFOneBSchedule => self.forward_1f1b(input),
            PipelineSchedule::InterleavedOneFOneB => self.forward_interleaved(input),
        }
    }

    /// GPipe schedule: fill-drain with all forwards then all backwards.
    fn forward_gpipe(&self, input: &Variable) -> Variable {
        let rank = self.process_group.rank();
        let num_stages = self.stages.len();

        // Split input into microbatches
        let microbatches = self.split_microbatches(input);

        // Process all microbatches through pipeline
        let mut outputs = Vec::new();

        for microbatch in microbatches {
            let mut activation = microbatch;

            // Forward through all stages
            for (stage_idx, stage) in self.stages.iter().enumerate() {
                if stage.device_rank == rank {
                    activation = stage.forward(&activation);
                }

                // Send to next stage if not last
                if stage_idx < num_stages - 1 {
                    let next_rank = self.stages[stage_idx + 1].device_rank;
                    if stage.device_rank == rank {
                        // Send activation to next stage
                        self.send_activation(&activation, next_rank);
                    } else if next_rank == rank {
                        // Receive activation from previous stage
                        activation = self.recv_activation(stage.device_rank, activation.shape());
                    }
                }
            }

            // Last stage collects output
            if self.stages.last().map(|s| s.device_rank) == Some(rank) {
                outputs.push(activation);
            }
        }

        // Combine outputs
        self.combine_microbatches(&outputs)
    }

    /// 1F1B schedule: memory-efficient interleaved forward/backward.
    fn forward_1f1b(&self, input: &Variable) -> Variable {
        // For simplicity, fall back to GPipe in this implementation
        // Full 1F1B requires careful scheduling of forward/backward passes
        self.forward_gpipe(input)
    }

    /// Interleaved 1F1B for virtual pipeline parallelism.
    fn forward_interleaved(&self, input: &Variable) -> Variable {
        // For simplicity, fall back to GPipe
        self.forward_gpipe(input)
    }

    /// Splits input into microbatches.
    fn split_microbatches(&self, input: &Variable) -> Vec<Variable> {
        let data = input.data();
        let batch_size = data.shape()[0];
        let microbatch_size = (batch_size + self.num_microbatches - 1) / self.num_microbatches;

        let mut microbatches = Vec::new();
        let flat_data = data.to_vec();
        let elements_per_sample: usize = data.shape()[1..].iter().product();

        for i in 0..self.num_microbatches {
            let start = i * microbatch_size;
            let end = ((i + 1) * microbatch_size).min(batch_size);

            if start >= batch_size {
                break;
            }

            let mb_size = end - start;
            let start_idx = start * elements_per_sample;
            let end_idx = end * elements_per_sample;
            let mb_data: Vec<f32> = flat_data[start_idx..end_idx].to_vec();

            let mut shape = data.shape().to_vec();
            shape[0] = mb_size;
            let tensor = Tensor::from_vec(mb_data, &shape).unwrap();
            microbatches.push(Variable::new(tensor, input.requires_grad()));
        }

        microbatches
    }

    /// Combines microbatch outputs.
    fn combine_microbatches(&self, outputs: &[Variable]) -> Variable {
        if outputs.is_empty() {
            return Variable::new(Tensor::zeros(&[0]), false);
        }

        if outputs.len() == 1 {
            return outputs[0].clone();
        }

        // Concatenate along batch dimension
        let mut all_data = Vec::new();
        let mut total_batch = 0;
        let shape = outputs[0].data().shape().to_vec();

        for output in outputs {
            all_data.extend(output.data().to_vec());
            total_batch += output.data().shape()[0];
        }

        let mut new_shape = shape;
        new_shape[0] = total_batch;
        let tensor = Tensor::from_vec(all_data, &new_shape).unwrap();
        Variable::new(tensor, outputs[0].requires_grad())
    }

    /// Sends activation to another rank.
    fn send_activation(&self, activation: &Variable, dest_rank: usize) {
        let mut tensor = activation.data().clone();
        self.process_group.send_tensor(&mut tensor, dest_rank);
    }

    /// Receives activation from another rank.
    fn recv_activation(&self, src_rank: usize, shape: Vec<usize>) -> Variable {
        let tensor = self.process_group.recv_tensor(src_rank, &shape);
        Variable::new(tensor, true)
    }
}

impl<M: Module + Clone> Module for Pipeline<M> {
    fn forward(&self, input: &Variable) -> Variable {
        Pipeline::forward(self, input)
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.stages
            .iter()
            .flat_map(|s| s.parameters())
            .collect()
    }

    fn train(&mut self) {
        for stage in &mut self.stages {
            stage.train();
        }
    }

    fn eval(&mut self) {
        for stage in &mut self.stages {
            stage.eval();
        }
    }

    fn is_training(&self) -> bool {
        self.stages.first().map(|s| s.is_training()).unwrap_or(false)
    }
}

// =============================================================================
// Pipeline Memory Stats
// =============================================================================

/// Memory statistics for pipeline parallelism.
#[derive(Debug, Clone)]
pub struct PipelineMemoryStats {
    /// Number of stages
    pub num_stages: usize,
    /// Number of microbatches
    pub num_microbatches: usize,
    /// Peak activations stored (per stage)
    pub peak_activations_per_stage: usize,
    /// Schedule used
    pub schedule: PipelineSchedule,
}

impl PipelineMemoryStats {
    /// Estimates peak activation memory for GPipe.
    pub fn gpipe_peak_activations(num_stages: usize, num_microbatches: usize) -> usize {
        // GPipe stores all microbatch activations
        num_stages * num_microbatches
    }

    /// Estimates peak activation memory for 1F1B.
    pub fn one_f_one_b_peak_activations(num_stages: usize, num_microbatches: usize) -> usize {
        // 1F1B stores at most num_stages activations in steady state
        num_stages.min(num_microbatches)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axonml_nn::Linear;

    /// Simple identity module for testing pipelines (Linear doesn't impl Clone)
    #[derive(Clone)]
    struct IdentityModule {
        size: usize,
        training: bool,
    }

    impl IdentityModule {
        fn new(size: usize) -> Self {
            Self { size, training: true }
        }
    }

    impl Module for IdentityModule {
        fn forward(&self, input: &Variable) -> Variable {
            input.clone()
        }

        fn parameters(&self) -> Vec<Parameter> {
            Vec::new()
        }

        fn train(&mut self) {
            self.training = true;
        }

        fn eval(&mut self) {
            self.training = false;
        }

        fn is_training(&self) -> bool {
            self.training
        }
    }

    #[test]
    fn test_pipeline_schedule_default() {
        assert_eq!(PipelineSchedule::default(), PipelineSchedule::OneFOneBSchedule);
    }

    #[test]
    fn test_pipeline_stage_creation() {
        let module = Linear::new(10, 5);
        let stage = PipelineStage::new(module, 0, 0);

        assert_eq!(stage.stage_id(), 0);
        assert_eq!(stage.device_rank(), 0);
    }

    #[test]
    fn test_pipeline_creation() {
        let modules = vec![
            IdentityModule::new(10),
            IdentityModule::new(8),
            IdentityModule::new(6),
        ];
        let pg = ProcessGroup::mock();
        let pipeline = Pipeline::from_modules(modules, pg)
            .schedule(PipelineSchedule::GPipe)
            .num_microbatches(2);

        assert_eq!(pipeline.num_stages(), 3);
        assert_eq!(pipeline.get_schedule(), PipelineSchedule::GPipe);
    }

    #[test]
    fn test_pipeline_forward() {
        let modules = vec![
            IdentityModule::new(4),
        ];
        let pg = ProcessGroup::mock();
        let pipeline = Pipeline::from_modules(modules, pg);

        let input = Variable::new(Tensor::randn(&[2, 4]), false);
        let output = pipeline.forward(&input);

        assert_eq!(output.data().shape(), &[2, 4]);
    }

    #[test]
    fn test_pipeline_memory_stats() {
        let gpipe = PipelineMemoryStats::gpipe_peak_activations(4, 8);
        let one_f_one_b = PipelineMemoryStats::one_f_one_b_peak_activations(4, 8);

        assert_eq!(gpipe, 32); // 4 * 8
        assert_eq!(one_f_one_b, 4); // min(4, 8)
    }

    #[test]
    fn test_split_microbatches() {
        let modules = vec![IdentityModule::new(4)];
        let pg = ProcessGroup::mock();
        let pipeline = Pipeline::from_modules(modules, pg).num_microbatches(2);

        let input = Variable::new(Tensor::randn(&[4, 4]), false);
        let microbatches = pipeline.split_microbatches(&input);

        assert_eq!(microbatches.len(), 2);
        assert_eq!(microbatches[0].data().shape()[0], 2);
        assert_eq!(microbatches[1].data().shape()[0], 2);
    }
}
