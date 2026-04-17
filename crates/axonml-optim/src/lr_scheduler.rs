//! Learning rate schedulers — seven strategies for LR annealing.
//!
//! `StepLR` (decay every N epochs), `MultiStepLR` (decay at milestones),
//! `ExponentialLR` (multiplicative decay per epoch), `CosineAnnealingLR`
//! (cosine anneal to eta_min), `OneCycleLR` (1-cycle super-convergence
//! with warmup/annealing), `WarmupLR` (linear warmup then constant),
//! `ReduceLROnPlateau` (reduce on metric stall, with cooldown + min_lr).
//! Each `.step()` updates the optimizer's LR.
//!
//! # File
//! `crates/axonml-optim/src/lr_scheduler.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 14, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use crate::optimizer::Optimizer;

// =============================================================================
// LRScheduler Trait
// =============================================================================

/// Trait for learning rate schedulers.
pub trait LRScheduler {
    /// Updates the learning rate (epoch-based schedulers).
    fn step<O: Optimizer>(&mut self, optimizer: &mut O);

    /// Updates the learning rate based on a metric value.
    ///
    /// Default implementation delegates to `step()`, ignoring the metric.
    /// Override for metric-based schedulers like `ReduceLROnPlateau`.
    fn step_with_metric<O: Optimizer>(&mut self, optimizer: &mut O, _metric: f32) {
        self.step(optimizer);
    }

    /// Returns the current learning rate.
    fn get_last_lr(&self) -> f32;

    /// Returns the current epoch/step count.
    fn get_step(&self) -> usize;
}

// =============================================================================
// StepLR
// =============================================================================

/// Decays learning rate by gamma every `step_size` epochs.
///
/// lr = `initial_lr` * gamma^(epoch // `step_size`)
pub struct StepLR {
    initial_lr: f32,
    step_size: usize,
    gamma: f32,
    current_step: usize,
    last_lr: f32,
}

impl StepLR {
    /// Creates a new `StepLR` scheduler.
    pub fn new<O: Optimizer>(optimizer: &O, step_size: usize, gamma: f32) -> Self {
        let initial_lr = optimizer.get_lr();
        Self {
            initial_lr,
            step_size,
            gamma,
            current_step: 0,
            last_lr: initial_lr,
        }
    }
}

impl LRScheduler for StepLR {
    fn step<O: Optimizer>(&mut self, optimizer: &mut O) {
        self.current_step += 1;
        let num_decays = self.current_step / self.step_size;
        let new_lr = self.initial_lr * self.gamma.powi(num_decays as i32);
        optimizer.set_lr(new_lr);
        self.last_lr = new_lr;
    }

    fn get_last_lr(&self) -> f32 {
        self.last_lr
    }

    fn get_step(&self) -> usize {
        self.current_step
    }
}

// =============================================================================
// MultiStepLR
// =============================================================================

/// Decays learning rate by gamma at each milestone.
pub struct MultiStepLR {
    initial_lr: f32,
    milestones: Vec<usize>,
    gamma: f32,
    current_step: usize,
    last_lr: f32,
    milestone_idx: usize,
}

impl MultiStepLR {
    /// Creates a new `MultiStepLR` scheduler.
    pub fn new<O: Optimizer>(optimizer: &O, mut milestones: Vec<usize>, gamma: f32) -> Self {
        let initial_lr = optimizer.get_lr();
        milestones.sort_unstable();
        Self {
            initial_lr,
            milestones,
            gamma,
            current_step: 0,
            last_lr: initial_lr,
            milestone_idx: 0,
        }
    }
}

impl LRScheduler for MultiStepLR {
    fn step<O: Optimizer>(&mut self, optimizer: &mut O) {
        self.current_step += 1;

        // Check if we've passed any milestones
        while self.milestone_idx < self.milestones.len()
            && self.current_step >= self.milestones[self.milestone_idx]
        {
            self.milestone_idx += 1;
        }

        let new_lr = self.initial_lr * self.gamma.powi(self.milestone_idx as i32);
        optimizer.set_lr(new_lr);
        self.last_lr = new_lr;
    }

    fn get_last_lr(&self) -> f32 {
        self.last_lr
    }

    fn get_step(&self) -> usize {
        self.current_step
    }
}

// =============================================================================
// ExponentialLR
// =============================================================================

/// Decays learning rate by gamma every epoch.
///
/// lr = `initial_lr` * gamma^epoch
pub struct ExponentialLR {
    initial_lr: f32,
    gamma: f32,
    current_step: usize,
    last_lr: f32,
}

impl ExponentialLR {
    /// Creates a new `ExponentialLR` scheduler.
    pub fn new<O: Optimizer>(optimizer: &O, gamma: f32) -> Self {
        let initial_lr = optimizer.get_lr();
        Self {
            initial_lr,
            gamma,
            current_step: 0,
            last_lr: initial_lr,
        }
    }
}

impl LRScheduler for ExponentialLR {
    fn step<O: Optimizer>(&mut self, optimizer: &mut O) {
        self.current_step += 1;
        let new_lr = self.initial_lr * self.gamma.powi(self.current_step as i32);
        optimizer.set_lr(new_lr);
        self.last_lr = new_lr;
    }

    fn get_last_lr(&self) -> f32 {
        self.last_lr
    }

    fn get_step(&self) -> usize {
        self.current_step
    }
}

// =============================================================================
// CosineAnnealingLR
// =============================================================================

/// Cosine annealing learning rate scheduler.
///
/// lr = `eta_min` + (`initial_lr` - `eta_min`) * (1 + cos(pi * epoch / `T_max`)) / 2
pub struct CosineAnnealingLR {
    initial_lr: f32,
    t_max: usize,
    eta_min: f32,
    current_step: usize,
    last_lr: f32,
}

impl CosineAnnealingLR {
    /// Creates a new `CosineAnnealingLR` scheduler.
    pub fn new<O: Optimizer>(optimizer: &O, t_max: usize) -> Self {
        Self::with_eta_min(optimizer, t_max, 0.0)
    }

    /// Creates a `CosineAnnealingLR` with minimum learning rate.
    pub fn with_eta_min<O: Optimizer>(optimizer: &O, t_max: usize, eta_min: f32) -> Self {
        let initial_lr = optimizer.get_lr();
        Self {
            initial_lr,
            t_max,
            eta_min,
            current_step: 0,
            last_lr: initial_lr,
        }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn step<O: Optimizer>(&mut self, optimizer: &mut O) {
        self.current_step += 1;

        let progress = self.current_step as f32 / self.t_max as f32;
        let new_lr = self.eta_min
            + (self.initial_lr - self.eta_min) * (1.0 + (std::f32::consts::PI * progress).cos())
                / 2.0;

        optimizer.set_lr(new_lr);
        self.last_lr = new_lr;
    }

    fn get_last_lr(&self) -> f32 {
        self.last_lr
    }

    fn get_step(&self) -> usize {
        self.current_step
    }
}

// =============================================================================
// ReduceLROnPlateau
// =============================================================================

/// Reduces learning rate when a metric has stopped improving.
pub struct ReduceLROnPlateau {
    mode: String,
    factor: f32,
    patience: usize,
    threshold: f32,
    cooldown: usize,
    min_lr: f32,
    best: f32,
    num_bad_epochs: usize,
    cooldown_counter: usize,
    current_step: usize,
    last_lr: f32,
}

impl ReduceLROnPlateau {
    /// Creates a new `ReduceLROnPlateau` scheduler for minimizing.
    pub fn new<O: Optimizer>(optimizer: &O) -> Self {
        Self::with_options(optimizer, "min", 0.1, 10, 1e-4, 0, 0.0)
    }

    /// Creates a `ReduceLROnPlateau` with options.
    pub fn with_options<O: Optimizer>(
        optimizer: &O,
        mode: &str,
        factor: f32,
        patience: usize,
        threshold: f32,
        cooldown: usize,
        min_lr: f32,
    ) -> Self {
        let initial_lr = optimizer.get_lr();
        let best = if mode == "min" {
            f32::INFINITY
        } else {
            f32::NEG_INFINITY
        };
        Self {
            mode: mode.to_string(),
            factor,
            patience,
            threshold,
            cooldown,
            min_lr,
            best,
            num_bad_epochs: 0,
            cooldown_counter: 0,
            current_step: 0,
            last_lr: initial_lr,
        }
    }

    /// Internal: steps the scheduler based on a metric value.
    fn step_metric_impl<O: Optimizer>(&mut self, optimizer: &mut O, metric: f32) {
        self.current_step += 1;

        // Check if we're in cooldown
        if self.cooldown_counter > 0 {
            self.cooldown_counter -= 1;
            return;
        }

        // Check if metric improved
        let improved = if self.mode == "min" {
            metric < self.best * (1.0 - self.threshold)
        } else {
            metric > self.best * (1.0 + self.threshold)
        };

        if improved {
            self.best = metric;
            self.num_bad_epochs = 0;
        } else {
            self.num_bad_epochs += 1;
        }

        // Reduce learning rate if patience exceeded
        if self.num_bad_epochs > self.patience {
            let current_lr = optimizer.get_lr();
            let new_lr = (current_lr * self.factor).max(self.min_lr);
            optimizer.set_lr(new_lr);
            self.last_lr = new_lr;
            self.cooldown_counter = self.cooldown;
            self.num_bad_epochs = 0;
        }
    }
}

impl LRScheduler for ReduceLROnPlateau {
    fn step<O: Optimizer>(&mut self, _optimizer: &mut O) {
        // No-op: this scheduler requires a metric. Use step_with_metric().
        self.current_step += 1;
    }

    fn step_with_metric<O: Optimizer>(&mut self, optimizer: &mut O, metric: f32) {
        self.step_metric_impl(optimizer, metric);
    }

    fn get_last_lr(&self) -> f32 {
        self.last_lr
    }

    fn get_step(&self) -> usize {
        self.current_step
    }
}

// =============================================================================
// OneCycleLR
// =============================================================================

/// One-cycle learning rate scheduler.
///
/// Implements the 1cycle policy from "Super-Convergence" paper.
pub struct OneCycleLR {
    max_lr: f32,
    total_steps: usize,
    pct_start: f32,
    div_factor: f32,
    final_div_factor: f32,
    current_step: usize,
    last_lr: f32,
}

impl OneCycleLR {
    /// Creates a new `OneCycleLR` scheduler.
    pub fn new<O: Optimizer>(optimizer: &O, max_lr: f32, total_steps: usize) -> Self {
        Self::with_options(optimizer, max_lr, total_steps, 0.3, 25.0, 1e4)
    }

    /// Creates `OneCycleLR` with options.
    pub fn with_options<O: Optimizer>(
        _optimizer: &O,
        max_lr: f32,
        total_steps: usize,
        pct_start: f32,
        div_factor: f32,
        final_div_factor: f32,
    ) -> Self {
        let initial_lr = max_lr / div_factor;
        Self {
            max_lr,
            total_steps,
            pct_start,
            div_factor,
            final_div_factor,
            current_step: 0,
            last_lr: initial_lr,
        }
    }
}

impl LRScheduler for OneCycleLR {
    fn step<O: Optimizer>(&mut self, optimizer: &mut O) {
        self.current_step += 1;

        let step_ratio = self.current_step as f32 / self.total_steps as f32;
        let initial_lr = self.max_lr / self.div_factor;
        let min_lr = self.max_lr / self.final_div_factor;

        let new_lr = if step_ratio <= self.pct_start {
            // Warmup phase: linear increase from initial_lr to max_lr
            let phase_ratio = step_ratio / self.pct_start;
            initial_lr + (self.max_lr - initial_lr) * phase_ratio
        } else {
            // Annealing phase: cosine decrease from max_lr to min_lr
            let phase_ratio = (step_ratio - self.pct_start) / (1.0 - self.pct_start);
            min_lr
                + (self.max_lr - min_lr) * (1.0 + (std::f32::consts::PI * phase_ratio).cos()) / 2.0
        };

        optimizer.set_lr(new_lr);
        self.last_lr = new_lr;
    }

    fn get_last_lr(&self) -> f32 {
        self.last_lr
    }

    fn get_step(&self) -> usize {
        self.current_step
    }
}

// =============================================================================
// WarmupLR
// =============================================================================

/// Linear warmup scheduler.
///
/// Linearly increases learning rate from 0 to `initial_lr` over `warmup_steps`.
pub struct WarmupLR {
    initial_lr: f32,
    warmup_steps: usize,
    current_step: usize,
    last_lr: f32,
}

impl WarmupLR {
    /// Creates a new `WarmupLR` scheduler.
    pub fn new<O: Optimizer>(optimizer: &O, warmup_steps: usize) -> Self {
        let initial_lr = optimizer.get_lr();
        Self {
            initial_lr,
            warmup_steps,
            current_step: 0,
            last_lr: 0.0,
        }
    }
}

impl LRScheduler for WarmupLR {
    fn step<O: Optimizer>(&mut self, optimizer: &mut O) {
        self.current_step += 1;

        let new_lr = if self.current_step <= self.warmup_steps {
            self.initial_lr * (self.current_step as f32 / self.warmup_steps as f32)
        } else {
            self.initial_lr
        };

        optimizer.set_lr(new_lr);
        self.last_lr = new_lr;
    }

    fn get_last_lr(&self) -> f32 {
        self.last_lr
    }

    fn get_step(&self) -> usize {
        self.current_step
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SGD;
    use axonml_autograd::Variable;
    use axonml_nn::Parameter;
    use axonml_tensor::Tensor;

    fn create_test_optimizer() -> SGD {
        let var = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), true);
        let param = Parameter::from_variable(var);
        SGD::new(vec![param], 0.1)
    }

    #[test]
    fn test_step_lr() {
        let mut optimizer = create_test_optimizer();
        let mut scheduler = StepLR::new(&optimizer, 10, 0.1);

        assert!((optimizer.get_lr() - 0.1).abs() < 1e-6);

        for _ in 0..10 {
            scheduler.step(&mut optimizer);
        }

        assert!((optimizer.get_lr() - 0.01).abs() < 1e-6);

        for _ in 0..10 {
            scheduler.step(&mut optimizer);
        }

        assert!((optimizer.get_lr() - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_multi_step_lr() {
        let mut optimizer = create_test_optimizer();
        let mut scheduler = MultiStepLR::new(&optimizer, vec![5, 15], 0.1);

        assert!((optimizer.get_lr() - 0.1).abs() < 1e-6);

        for _ in 0..5 {
            scheduler.step(&mut optimizer);
        }
        assert!((optimizer.get_lr() - 0.01).abs() < 1e-6);

        for _ in 0..10 {
            scheduler.step(&mut optimizer);
        }
        assert!((optimizer.get_lr() - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_exponential_lr() {
        let mut optimizer = create_test_optimizer();
        let mut scheduler = ExponentialLR::new(&optimizer, 0.9);

        scheduler.step(&mut optimizer);
        assert!((optimizer.get_lr() - 0.09).abs() < 1e-6);

        scheduler.step(&mut optimizer);
        assert!((optimizer.get_lr() - 0.081).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_annealing_lr() {
        let mut optimizer = create_test_optimizer();
        let mut scheduler = CosineAnnealingLR::new(&optimizer, 100);

        // At step 50 (halfway), should be at eta_min + (initial - eta_min) * 0.5
        for _ in 0..50 {
            scheduler.step(&mut optimizer);
        }
        assert!((optimizer.get_lr() - 0.05).abs() < 0.01);

        // At step 100 (end), should be at eta_min
        for _ in 0..50 {
            scheduler.step(&mut optimizer);
        }
        assert!(optimizer.get_lr() < 0.01);
    }

    #[test]
    fn test_warmup_lr() {
        let mut optimizer = create_test_optimizer();
        let mut scheduler = WarmupLR::new(&optimizer, 10);

        scheduler.step(&mut optimizer);
        assert!((optimizer.get_lr() - 0.01).abs() < 1e-6);

        for _ in 0..9 {
            scheduler.step(&mut optimizer);
        }
        assert!((optimizer.get_lr() - 0.1).abs() < 1e-6);

        // After warmup, should stay at initial_lr
        scheduler.step(&mut optimizer);
        assert!((optimizer.get_lr() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_one_cycle_lr() {
        let mut optimizer = create_test_optimizer();
        let mut scheduler = OneCycleLR::new(&optimizer, 0.1, 100);

        // At start, should be at initial_lr = max_lr / div_factor
        assert!((scheduler.get_last_lr() - 0.004).abs() < 0.001);

        // Step through warmup phase
        for _ in 0..30 {
            scheduler.step(&mut optimizer);
        }

        // Should be at or near max_lr
        assert!(optimizer.get_lr() > 0.08);
    }

    #[test]
    fn test_reduce_lr_on_plateau() {
        let mut optimizer = create_test_optimizer();
        let mut scheduler = ReduceLROnPlateau::with_options(&optimizer, "min", 0.5, 2, 0.0, 0, 0.0);

        let initial_lr = optimizer.get_lr();

        // Simulate improving metric
        scheduler.step_with_metric(&mut optimizer, 1.0);
        scheduler.step_with_metric(&mut optimizer, 0.9);
        assert!((optimizer.get_lr() - initial_lr).abs() < 1e-6);

        // Simulate plateau
        scheduler.step_with_metric(&mut optimizer, 0.91);
        scheduler.step_with_metric(&mut optimizer, 0.91);
        scheduler.step_with_metric(&mut optimizer, 0.91);

        // LR should have been reduced
        assert!(optimizer.get_lr() < initial_lr);
    }

    // =========================================================================
    // ReduceLROnPlateau — Comprehensive
    // =========================================================================

    #[test]
    fn test_reduce_lr_on_plateau_max_mode() {
        let mut optimizer = create_test_optimizer();
        let mut scheduler = ReduceLROnPlateau::with_options(&optimizer, "max", 0.5, 2, 0.0, 0, 0.0);

        let initial_lr = optimizer.get_lr();

        // Improving metric (higher is better)
        scheduler.step_with_metric(&mut optimizer, 0.8);
        scheduler.step_with_metric(&mut optimizer, 0.9);
        assert!((optimizer.get_lr() - initial_lr).abs() < 1e-6);

        // Plateau (metric not improving)
        scheduler.step_with_metric(&mut optimizer, 0.85);
        scheduler.step_with_metric(&mut optimizer, 0.85);
        scheduler.step_with_metric(&mut optimizer, 0.85);

        assert!(
            optimizer.get_lr() < initial_lr,
            "LR should reduce on plateau in max mode"
        );
    }

    #[test]
    fn test_reduce_lr_on_plateau_min_lr_floor() {
        let mut optimizer = create_test_optimizer();
        let mut scheduler =
            ReduceLROnPlateau::with_options(&optimizer, "min", 0.1, 0, 0.0, 0, 0.001);

        // Force many reductions
        for _ in 0..50 {
            scheduler.step_with_metric(&mut optimizer, 999.0); // never improves
        }

        assert!(
            optimizer.get_lr() >= 0.001,
            "LR should not go below min_lr, got {}",
            optimizer.get_lr()
        );
    }

    #[test]
    fn test_reduce_lr_cooldown() {
        let mut optimizer = create_test_optimizer();
        let mut scheduler = ReduceLROnPlateau::with_options(&optimizer, "min", 0.5, 0, 0.0, 3, 0.0);

        let initial_lr = optimizer.get_lr();

        // Trigger reduction
        scheduler.step_with_metric(&mut optimizer, 999.0);
        scheduler.step_with_metric(&mut optimizer, 999.0);
        let lr_after_first_reduce = optimizer.get_lr();
        assert!(lr_after_first_reduce < initial_lr);

        // During cooldown (3 steps), LR should not change again
        scheduler.step_with_metric(&mut optimizer, 999.0);
        scheduler.step_with_metric(&mut optimizer, 999.0);
        scheduler.step_with_metric(&mut optimizer, 999.0);
        assert!(
            (optimizer.get_lr() - lr_after_first_reduce).abs() < 1e-8,
            "LR should not change during cooldown"
        );
    }

    // =========================================================================
    // OneCycleLR — Comprehensive
    // =========================================================================

    #[test]
    fn test_one_cycle_lr_full_cycle() {
        let mut optimizer = create_test_optimizer();
        let mut scheduler = OneCycleLR::new(&optimizer, 0.1, 100);

        let mut lrs = Vec::new();
        for _ in 0..100 {
            scheduler.step(&mut optimizer);
            lrs.push(optimizer.get_lr());
        }

        // Should start low, peak around 30%, end very low
        let max_lr = lrs.iter().copied().fold(f32::MIN, f32::max);
        let final_lr = *lrs.last().unwrap();

        assert!(
            max_lr > 0.08,
            "Peak should be near max_lr=0.1, got {}",
            max_lr
        );
        assert!(
            final_lr < 0.001,
            "Final LR should be very small, got {}",
            final_lr
        );

        // Peak should occur around 30% of total steps
        let peak_idx = lrs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert!(
            (25..=35).contains(&peak_idx),
            "Peak should be around step 30, was at step {}",
            peak_idx
        );
    }

    #[test]
    fn test_one_cycle_lr_monotonic_phases() {
        let mut optimizer = create_test_optimizer();
        let mut scheduler = OneCycleLR::new(&optimizer, 0.1, 100);

        let mut lrs = Vec::new();
        for _ in 0..100 {
            scheduler.step(&mut optimizer);
            lrs.push(optimizer.get_lr());
        }

        // Warmup phase (steps 1-30): should be monotonically increasing
        for i in 1..29 {
            assert!(
                lrs[i] >= lrs[i - 1] - 1e-6,
                "Warmup should increase: step {} lr={} < step {} lr={}",
                i,
                lrs[i],
                i - 1,
                lrs[i - 1]
            );
        }

        // Annealing phase (steps 31-100): should be monotonically decreasing
        for i in 32..99 {
            assert!(
                lrs[i] <= lrs[i - 1] + 1e-6,
                "Annealing should decrease: step {} lr={} > step {} lr={}",
                i,
                lrs[i],
                i - 1,
                lrs[i - 1]
            );
        }
    }

    // =========================================================================
    // CosineAnnealingLR — Comprehensive
    // =========================================================================

    #[test]
    fn test_cosine_annealing_with_eta_min() {
        let mut optimizer = create_test_optimizer();
        let mut scheduler = CosineAnnealingLR::with_eta_min(&optimizer, 100, 0.001);

        for _ in 0..100 {
            scheduler.step(&mut optimizer);
        }

        // At end should be at eta_min
        assert!(
            (optimizer.get_lr() - 0.001).abs() < 0.002,
            "Should reach eta_min at end, got {}",
            optimizer.get_lr()
        );
    }

    #[test]
    fn test_cosine_annealing_monotonic_decrease() {
        let mut optimizer = create_test_optimizer();
        let mut scheduler = CosineAnnealingLR::new(&optimizer, 100);

        let mut lrs = Vec::new();
        for _ in 0..100 {
            scheduler.step(&mut optimizer);
            lrs.push(optimizer.get_lr());
        }

        // Cosine annealing should monotonically decrease
        for i in 1..lrs.len() {
            assert!(
                lrs[i] <= lrs[i - 1] + 1e-6,
                "Cosine should decrease: step {} lr={} > step {} lr={}",
                i + 1,
                lrs[i],
                i,
                lrs[i - 1]
            );
        }

        // All LRs should be non-negative
        assert!(
            lrs.iter().all(|lr| *lr >= 0.0),
            "LRs should be non-negative"
        );
    }

    // =========================================================================
    // WarmupLR — Edge Cases
    // =========================================================================

    #[test]
    fn test_warmup_lr_stays_constant_after() {
        let mut optimizer = create_test_optimizer();
        let mut scheduler = WarmupLR::new(&optimizer, 5);

        for _ in 0..5 {
            scheduler.step(&mut optimizer);
        }
        let target = optimizer.get_lr();

        // Should stay constant for many more steps
        for _ in 0..100 {
            scheduler.step(&mut optimizer);
            assert!(
                (optimizer.get_lr() - target).abs() < 1e-8,
                "LR should stay at {} after warmup, got {}",
                target,
                optimizer.get_lr()
            );
        }
    }
}
