//! Graph Fusion Optimizer
//!
//! Optimizes computational graphs by detecting and applying fusion patterns.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use crate::patterns::{FusionPattern, OpType, detect_patterns};
use crate::error::{FusionError, FusionResult};

// =============================================================================
// Optimizer Configuration
// =============================================================================

/// Configuration for the fusion optimizer.
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Enable elementwise fusion.
    pub fuse_elementwise: bool,
    /// Enable linear (matmul+bias+act) fusion.
    pub fuse_linear: bool,
    /// Enable convolution fusion.
    pub fuse_conv: bool,
    /// Minimum number of ops to fuse for elementwise chains.
    pub min_elementwise_chain: usize,
    /// Enable aggressive fusion (may increase compilation time).
    pub aggressive: bool,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            fuse_elementwise: true,
            fuse_linear: true,
            fuse_conv: true,
            min_elementwise_chain: 2,
            aggressive: false,
        }
    }
}

impl FusionConfig {
    /// Creates a configuration with all fusion enabled.
    pub fn all_enabled() -> Self {
        Self {
            fuse_elementwise: true,
            fuse_linear: true,
            fuse_conv: true,
            min_elementwise_chain: 2,
            aggressive: true,
        }
    }

    /// Creates a configuration with only safe fusions.
    pub fn conservative() -> Self {
        Self {
            fuse_elementwise: true,
            fuse_linear: true,
            fuse_conv: false,
            min_elementwise_chain: 3,
            aggressive: false,
        }
    }
}

// =============================================================================
// Fusion Optimizer
// =============================================================================

/// Graph fusion optimizer.
#[derive(Debug)]
pub struct FusionOptimizer {
    /// Configuration.
    config: FusionConfig,
    /// Statistics about optimizations applied.
    stats: OptimizationStats,
}

/// Statistics about optimization.
#[derive(Debug, Default, Clone)]
pub struct OptimizationStats {
    /// Number of fusion patterns applied.
    pub fusions_applied: usize,
    /// Number of operations eliminated.
    pub ops_eliminated: usize,
    /// Estimated speedup.
    pub estimated_speedup: f32,
    /// Patterns applied (pattern, count).
    pub patterns: Vec<(FusionPattern, usize)>,
}

impl OptimizationStats {
    /// Adds a fusion to the statistics.
    pub fn add_fusion(&mut self, pattern: FusionPattern) {
        self.fusions_applied += 1;
        self.ops_eliminated += pattern.num_ops() - 1;
        self.estimated_speedup *= pattern.estimated_speedup();

        // Update pattern counts
        if let Some(entry) = self.patterns.iter_mut().find(|(p, _)| *p == pattern) {
            entry.1 += 1;
        } else {
            self.patterns.push((pattern, 1));
        }
    }
}

impl FusionOptimizer {
    /// Creates a new optimizer with default configuration.
    pub fn new() -> Self {
        Self::with_config(FusionConfig::default())
    }

    /// Creates a new optimizer with the given configuration.
    pub fn with_config(config: FusionConfig) -> Self {
        Self {
            config,
            stats: OptimizationStats {
                estimated_speedup: 1.0,
                ..Default::default()
            },
        }
    }

    /// Returns the configuration.
    pub fn config(&self) -> &FusionConfig {
        &self.config
    }

    /// Returns optimization statistics.
    pub fn stats(&self) -> &OptimizationStats {
        &self.stats
    }

    /// Analyzes a sequence of operations and returns applicable fusions.
    pub fn analyze(&self, ops: &[OpType]) -> Vec<(FusionPattern, usize, usize)> {
        let patterns = detect_patterns(ops);

        // Filter based on config
        patterns.into_iter().filter(|(pattern, _, _)| {
            match pattern {
                FusionPattern::MatMulBias | FusionPattern::MatMulBiasRelu |
                FusionPattern::MatMulBiasGelu => self.config.fuse_linear,

                FusionPattern::ConvBatchNorm | FusionPattern::ConvBatchNormRelu => {
                    self.config.fuse_conv
                }

                FusionPattern::ElementwiseChain => self.config.fuse_elementwise,

                _ => true,
            }
        }).collect()
    }

    /// Applies detected fusions and updates statistics.
    pub fn apply_fusions(&mut self, patterns: &[(FusionPattern, usize, usize)]) {
        for (pattern, _, _) in patterns {
            self.stats.add_fusion(*pattern);
        }
    }

    /// Resets optimization statistics.
    pub fn reset_stats(&mut self) {
        self.stats = OptimizationStats {
            estimated_speedup: 1.0,
            ..Default::default()
        };
    }
}

impl Default for FusionOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Graph Optimization
// =============================================================================

/// Optimizes a computational graph represented as a sequence of operations.
///
/// # Arguments
/// * `ops` - The operations in the graph
/// * `config` - Fusion configuration
///
/// # Returns
/// Optimization results including detected patterns and statistics
pub fn optimize_graph(
    ops: &[OpType],
    config: Option<FusionConfig>,
) -> FusionResult<(Vec<(FusionPattern, usize, usize)>, OptimizationStats)> {
    let mut optimizer = FusionOptimizer::with_config(config.unwrap_or_default());

    let patterns = optimizer.analyze(ops);
    optimizer.apply_fusions(&patterns);

    Ok((patterns, optimizer.stats().clone()))
}

/// Estimates the potential speedup from optimizing a graph.
pub fn estimate_speedup(ops: &[OpType]) -> f32 {
    let (patterns, _) = optimize_graph(ops, None).unwrap_or_default();

    let mut speedup = 1.0f32;
    for (pattern, _, _) in patterns {
        speedup *= pattern.estimated_speedup();
    }
    speedup
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_creation() {
        let opt = FusionOptimizer::new();
        assert!(opt.config().fuse_elementwise);
        assert!(opt.config().fuse_linear);
    }

    #[test]
    fn test_analyze_patterns() {
        let opt = FusionOptimizer::new();
        let ops = vec![OpType::MatMul, OpType::Add, OpType::Relu];
        let patterns = opt.analyze(&ops);

        assert!(!patterns.is_empty());
    }

    #[test]
    fn test_optimization_stats() {
        let mut opt = FusionOptimizer::new();
        let ops = vec![OpType::MatMul, OpType::Add, OpType::Relu];
        let patterns = opt.analyze(&ops);
        opt.apply_fusions(&patterns);

        assert!(opt.stats().fusions_applied > 0);
        assert!(opt.stats().estimated_speedup > 1.0);
    }

    #[test]
    fn test_disabled_fusion() {
        let config = FusionConfig {
            fuse_linear: false,
            ..Default::default()
        };
        let opt = FusionOptimizer::with_config(config);
        let ops = vec![OpType::MatMul, OpType::Add, OpType::Relu];
        let patterns = opt.analyze(&ops);

        // Should not detect matmul patterns when disabled
        let has_matmul_pattern = patterns.iter().any(|(p, _, _)| {
            matches!(p, FusionPattern::MatMulBias | FusionPattern::MatMulBiasRelu)
        });
        assert!(!has_matmul_pattern);
    }

    #[test]
    fn test_optimize_graph() {
        let ops = vec![OpType::MatMul, OpType::Add, OpType::Relu, OpType::Add, OpType::Mul];
        let (patterns, stats) = optimize_graph(&ops, None).unwrap();

        assert!(!patterns.is_empty());
        assert!(stats.estimated_speedup >= 1.0);
    }

    #[test]
    fn test_estimate_speedup() {
        let ops = vec![OpType::MatMul, OpType::Add, OpType::Relu];
        let speedup = estimate_speedup(&ops);
        assert!(speedup >= 1.0);
    }

    #[test]
    fn test_conservative_config() {
        let config = FusionConfig::conservative();
        assert!(!config.aggressive);
        assert!(!config.fuse_conv);
        assert_eq!(config.min_elementwise_chain, 3);
    }
}
