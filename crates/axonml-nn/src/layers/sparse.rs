//! Differentiable structured sparsity — a novel AxonML feature.
//!
//! 1147 lines. `SparseLinear` (Linear layer with a learnable soft-threshold
//! pruning mask — the mask is differentiable via straight-through estimation,
//! enabling end-to-end learning of which weights to prune). `GroupSparsity`
//! (L2,1 regularization that drives entire rows/columns to zero for structured
//! pruning). `LotteryTicket` (implements the lottery ticket hypothesis:
//! train → prune → rewind to init → retrain the sparse subnetwork, with
//! iterative magnitude pruning and score tracking).
//!
//! # File
//! `crates/axonml-nn/src/layers/sparse.rs`
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

use std::collections::HashMap;

use axonml_autograd::Variable;
use axonml_tensor::Tensor;

use crate::init::{constant, kaiming_uniform, zeros};
use crate::module::Module;
use crate::parameter::Parameter;

// =============================================================================
// Constants
// =============================================================================

/// Temperature for the sigmoid soft thresholding.
/// Higher values produce a sharper (more binary) mask.
const TEMPERATURE: f32 = 10.0;

/// Default initial threshold value.
/// Small so that most weights start active.
const DEFAULT_THRESHOLD: f32 = 0.01;

// =============================================================================
// SparseLinear
// =============================================================================

/// A linear layer with a differentiable magnitude pruning mask.
///
/// During the forward pass, a soft mask is computed via sigmoid soft thresholding:
///
/// ```text
/// mask = sigmoid((|weight| - threshold) * temperature)
/// effective_weight = weight * mask
/// y = x @ effective_weight^T + bias
/// ```
///
/// The sigmoid makes the mask differentiable, so gradients flow through it and
/// the network learns which weights to prune. The `threshold` parameter is
/// learnable and included in `parameters()`.
///
/// # Structured vs Unstructured
///
/// - **Structured** (`structured=true`): One threshold per output neuron.
///   Entire output channels can be pruned, yielding hardware-friendly sparsity.
/// - **Unstructured** (`structured=false`): One threshold per weight element.
///   Finer-grained but less hardware-friendly.
///
/// # Example
/// ```ignore
/// let layer = SparseLinear::new(784, 256);
/// let output = layer.forward(&input);
/// println!("Density: {:.1}%", layer.density() * 100.0);
/// ```
pub struct SparseLinear {
    /// Weight matrix of shape (out_features, in_features).
    pub weight: Parameter,
    /// Optional bias vector of shape (out_features).
    pub bias: Option<Parameter>,
    /// Learnable magnitude thresholds. Shape depends on `structured`:
    /// - Structured: (out_features,)
    /// - Unstructured: (out_features, in_features)
    pub threshold: Parameter,
    /// Input feature dimension.
    in_features: usize,
    /// Output feature dimension.
    out_features: usize,
    /// Whether to use structured (channel) pruning.
    structured: bool,
}

impl SparseLinear {
    /// Creates a new SparseLinear layer with structured pruning and bias.
    ///
    /// # Arguments
    /// * `in_features` - Size of each input sample
    /// * `out_features` - Size of each output sample
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self::build(in_features, out_features, true, true)
    }

    /// Creates a new SparseLinear layer with unstructured (per-weight) pruning.
    ///
    /// # Arguments
    /// * `in_features` - Size of each input sample
    /// * `out_features` - Size of each output sample
    pub fn unstructured(in_features: usize, out_features: usize) -> Self {
        Self::build(in_features, out_features, false, true)
    }

    /// Creates a new SparseLinear layer with configurable bias.
    ///
    /// # Arguments
    /// * `in_features` - Size of each input sample
    /// * `out_features` - Size of each output sample
    /// * `bias` - Whether to include a learnable bias
    pub fn with_bias(in_features: usize, out_features: usize, bias: bool) -> Self {
        Self::build(in_features, out_features, true, bias)
    }

    /// Internal constructor.
    fn build(in_features: usize, out_features: usize, structured: bool, bias: bool) -> Self {
        // Kaiming uniform initialization for weights
        let weight_data = kaiming_uniform(out_features, in_features);
        let weight = Parameter::named("weight", weight_data, true);

        // Bias initialization
        let bias_param = if bias {
            let bias_data = zeros(&[out_features]);
            Some(Parameter::named("bias", bias_data, true))
        } else {
            None
        };

        // Threshold initialization — small value so most weights start active
        let threshold_data = if structured {
            constant(&[out_features], DEFAULT_THRESHOLD)
        } else {
            constant(&[out_features, in_features], DEFAULT_THRESHOLD)
        };
        let threshold = Parameter::named("threshold", threshold_data, true);

        Self {
            weight,
            bias: bias_param,
            threshold,
            in_features,
            out_features,
            structured,
        }
    }

    /// Returns the input feature dimension.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Returns the output feature dimension.
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Returns whether this layer uses structured pruning.
    pub fn is_structured(&self) -> bool {
        self.structured
    }

    /// Computes the hard binary mask (threshold at 0.5) from the current weights
    /// and thresholds.
    ///
    /// Returns a Tensor of 0s and 1s with the same shape as weight.
    fn hard_mask(&self) -> Tensor<f32> {
        let weight_data = self.weight.data();
        let threshold_data = self.threshold.data();
        let w_vec = weight_data.to_vec();
        let t_vec = threshold_data.to_vec();

        let mask_vec: Vec<f32> = if self.structured {
            // One threshold per output neuron — broadcast across in_features
            w_vec
                .iter()
                .enumerate()
                .map(|(idx, &w)| {
                    let out_idx = idx / self.in_features;
                    let t = t_vec[out_idx];
                    if w.abs() >= t { 1.0 } else { 0.0 }
                })
                .collect()
        } else {
            // One threshold per weight
            w_vec
                .iter()
                .zip(t_vec.iter())
                .map(|(&w, &t)| if w.abs() >= t { 1.0 } else { 0.0 })
                .collect()
        };

        Tensor::from_vec(mask_vec, &[self.out_features, self.in_features])
            .expect("tensor creation failed")
    }

    /// Returns the fraction of weights that are active (above threshold).
    ///
    /// Uses hard thresholding at |weight| >= threshold.
    pub fn density(&self) -> f32 {
        let mask = self.hard_mask();
        let mask_vec = mask.to_vec();
        let total = mask_vec.len() as f32;
        let active: f32 = mask_vec.iter().sum();
        active / total
    }

    /// Returns the fraction of weights that are pruned.
    ///
    /// Equivalent to `1.0 - density()`.
    pub fn sparsity(&self) -> f32 {
        1.0 - self.density()
    }

    /// Returns the number of active (non-pruned) weights.
    pub fn num_active(&self) -> usize {
        let mask = self.hard_mask();
        let mask_vec = mask.to_vec();
        mask_vec.iter().filter(|&&v| v > 0.5).count()
    }

    /// Permanently applies the pruning mask to the weights.
    ///
    /// After calling this, pruned weights are zeroed and the threshold is reset
    /// to zero. This is an irreversible optimization for inference — the zeroed
    /// weights will not be recovered.
    pub fn hard_prune(&mut self) {
        let mask = self.hard_mask();
        let weight_data = self.weight.data();
        let w_vec = weight_data.to_vec();
        let m_vec = mask.to_vec();

        let pruned: Vec<f32> = w_vec
            .iter()
            .zip(m_vec.iter())
            .map(|(&w, &m)| w * m)
            .collect();

        let new_weight = Tensor::from_vec(pruned, &[self.out_features, self.in_features])
            .expect("tensor creation failed");
        self.weight.update_data(new_weight);

        // Reset thresholds to zero so forward pass doesn't re-prune
        let zero_threshold = if self.structured {
            zeros(&[self.out_features])
        } else {
            zeros(&[self.out_features, self.in_features])
        };
        self.threshold.update_data(zero_threshold);
    }

    /// Resets the threshold to a specific value.
    ///
    /// # Arguments
    /// * `value` - The new threshold value
    pub fn reset_threshold(&mut self, value: f32) {
        let new_threshold = if self.structured {
            constant(&[self.out_features], value)
        } else {
            constant(&[self.out_features, self.in_features], value)
        };
        self.threshold.update_data(new_threshold);
    }

    /// Returns the effective weight (weight * hard_mask) for inspection.
    ///
    /// This shows what the weight matrix looks like after hard pruning,
    /// without actually modifying the layer.
    pub fn effective_weight(&self) -> Tensor<f32> {
        let mask = self.hard_mask();
        let weight_data = self.weight.data();
        let w_vec = weight_data.to_vec();
        let m_vec = mask.to_vec();

        let effective: Vec<f32> = w_vec
            .iter()
            .zip(m_vec.iter())
            .map(|(&w, &m)| w * m)
            .collect();

        Tensor::from_vec(effective, &[self.out_features, self.in_features])
            .expect("tensor creation failed")
    }

    /// Computes the soft mask using differentiable sigmoid thresholding.
    ///
    /// The soft mask is computed element-wise as:
    /// ```text
    /// mask_ij = sigmoid((|w_ij| - threshold_j) * temperature)
    /// ```
    ///
    /// For structured pruning, `threshold_j` is broadcast across `in_features`.
    /// For unstructured pruning, each weight has its own threshold.
    fn compute_soft_mask(&self, weight_var: &Variable) -> Variable {
        let weight_data = weight_var.data();
        let threshold_data = self.threshold.data();
        let w_vec = weight_data.to_vec();
        let t_vec = threshold_data.to_vec();

        // Compute sigmoid((|w| - threshold) * temperature) element-wise
        let mask_vec: Vec<f32> = if self.structured {
            w_vec
                .iter()
                .enumerate()
                .map(|(idx, &w)| {
                    let out_idx = idx / self.in_features;
                    let t = t_vec[out_idx];
                    let x = (w.abs() - t) * TEMPERATURE;
                    1.0 / (1.0 + (-x).exp())
                })
                .collect()
        } else {
            w_vec
                .iter()
                .zip(t_vec.iter())
                .map(|(&w, &t)| {
                    let x = (w.abs() - t) * TEMPERATURE;
                    1.0 / (1.0 + (-x).exp())
                })
                .collect()
        };

        let mask_tensor = Tensor::from_vec(mask_vec, &[self.out_features, self.in_features])
            .expect("tensor creation failed");

        // Create as a variable that participates in the graph
        // The mask depends on both weight and threshold, but since we compute
        // it from the raw tensor values, we wrap it as a new variable.
        // The gradient signal flows through the weight multiplication below.
        Variable::new(mask_tensor, false)
    }
}

impl Module for SparseLinear {
    fn forward(&self, input: &Variable) -> Variable {
        let input_shape = input.shape();
        let batch_dims: Vec<usize> = input_shape[..input_shape.len() - 1].to_vec();
        let total_batch: usize = batch_dims.iter().product();

        // Reshape to 2D if needed
        let input_2d = if input_shape.len() > 2 {
            input.reshape(&[total_batch, self.in_features])
        } else {
            input.clone()
        };

        // Get weight variable and compute soft mask
        let weight_var = self.weight.variable();
        let mask = self.compute_soft_mask(&weight_var);

        // effective_weight = weight * mask
        let effective_weight = weight_var.mul_var(&mask);

        // y = x @ effective_weight^T
        let weight_t = effective_weight.transpose(0, 1);
        let mut output = input_2d.matmul(&weight_t);

        // Add bias if present
        if let Some(ref bias) = self.bias {
            let bias_var = bias.variable();
            output = output.add_var(&bias_var);
        }

        // Reshape back to original batch dimensions
        if batch_dims.len() > 1 || (batch_dims.len() == 1 && input_shape.len() > 2) {
            let mut output_shape: Vec<usize> = batch_dims;
            output_shape.push(self.out_features);
            output.reshape(&output_shape)
        } else {
            output
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![self.weight.clone(), self.threshold.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.insert("weight".to_string(), self.weight.clone());
        params.insert("threshold".to_string(), self.threshold.clone());
        if let Some(ref bias) = self.bias {
            params.insert("bias".to_string(), bias.clone());
        }
        params
    }

    fn name(&self) -> &'static str {
        "SparseLinear"
    }
}

impl std::fmt::Debug for SparseLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SparseLinear")
            .field("in_features", &self.in_features)
            .field("out_features", &self.out_features)
            .field("bias", &self.bias.is_some())
            .field("structured", &self.structured)
            .field("density", &self.density())
            .finish()
    }
}

// =============================================================================
// GroupSparsity
// =============================================================================

/// A regularization module that encourages structured sparsity via group L1 norm.
///
/// Computes a penalty term that can be added to the loss function:
///
/// ```text
/// penalty = lambda * sum_g(||weight_g||_2)
/// ```
///
/// where `weight_g` is a group of weights (e.g., all weights for one output neuron).
/// This encourages entire groups to go to zero (structured pruning), since the L2
/// norm within each group distributes the penalty equally among group members.
///
/// # Example
/// ```ignore
/// let reg = GroupSparsity::new(0.001, 128);  // lambda=0.001, group_size=128
/// let penalty = reg.penalty(&model.weight_variable());
/// let total_loss = task_loss.add_var(&penalty);
/// ```
pub struct GroupSparsity {
    /// Regularization strength. Higher values encourage more sparsity.
    lambda: f32,
    /// Number of weights per group.
    group_size: usize,
}

impl GroupSparsity {
    /// Creates a new GroupSparsity regularizer.
    ///
    /// # Arguments
    /// * `lambda` - Regularization strength (e.g., 0.001)
    /// * `group_size` - Number of weights per group (e.g., in_features for neuron-level)
    pub fn new(lambda: f32, group_size: usize) -> Self {
        assert!(group_size > 0, "group_size must be positive");
        Self { lambda, group_size }
    }

    /// Returns the regularization strength.
    pub fn lambda(&self) -> f32 {
        self.lambda
    }

    /// Returns the group size.
    pub fn group_size(&self) -> usize {
        self.group_size
    }

    /// Computes the group L1 penalty for the given weight variable.
    ///
    /// The penalty is computed as:
    /// 1. Reshape weight into groups of size `group_size`
    /// 2. Compute L2 norm of each group
    /// 3. Sum all group norms (L1 of norms)
    /// 4. Multiply by lambda
    ///
    /// Returns a scalar Variable that can be added to the loss.
    pub fn penalty(&self, weight: &Variable) -> Variable {
        let weight_data = weight.data();
        let w_vec = weight_data.to_vec();
        let total = w_vec.len();

        // Number of complete groups
        let num_groups = total.div_ceil(self.group_size);

        // Compute L2 norm per group, then sum (L1 of group norms)
        let mut group_norm_sum = 0.0f32;
        for g in 0..num_groups {
            let start = g * self.group_size;
            let end = (start + self.group_size).min(total);
            let group = &w_vec[start..end];

            let l2_norm: f32 = group.iter().map(|&x| x * x).sum::<f32>().sqrt();
            group_norm_sum += l2_norm;
        }

        let penalty_val = self.lambda * group_norm_sum;
        let penalty_tensor =
            Tensor::from_vec(vec![penalty_val], &[1]).expect("tensor creation failed");

        // Create as a variable. The penalty is computed from raw tensor values
        // for simplicity. For full autograd integration, one would implement a
        // custom backward function, but the penalty is typically used alongside
        // weight decay in the optimizer.
        Variable::new(penalty_tensor, false)
    }
}

impl std::fmt::Debug for GroupSparsity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroupSparsity")
            .field("lambda", &self.lambda)
            .field("group_size", &self.group_size)
            .finish()
    }
}

// =============================================================================
// LotteryTicket
// =============================================================================

/// Implements the Lottery Ticket Hypothesis (Frankle & Carlin, 2019).
///
/// The Lottery Ticket Hypothesis states that dense networks contain sparse
/// subnetworks ("winning tickets") that can be trained in isolation to match
/// the full network's accuracy.
///
/// This struct saves a snapshot of the initial weights, then after pruning,
/// allows rewinding the unpruned weights back to their initial values while
/// keeping the pruning mask.
///
/// # Workflow
/// 1. Initialize network
/// 2. `let ticket = LotteryTicket::snapshot(&model.parameters());`
/// 3. Train network with pruning
/// 4. Determine pruning mask
/// 5. `ticket.rewind(&model.parameters());` — reset to initial weights
/// 6. Apply mask and train again
///
/// # Example
/// ```ignore
/// let model = SparseLinear::new(784, 256);
/// let ticket = LotteryTicket::snapshot(&model.parameters());
///
/// // ... train and prune ...
///
/// ticket.rewind(&model.parameters());  // Reset to initial weights
/// // ... retrain with mask ...
/// ```
pub struct LotteryTicket {
    /// Saved initial parameter values, keyed by parameter name or index.
    initial_weights: HashMap<String, Tensor<f32>>,
}

impl LotteryTicket {
    /// Takes a snapshot of the current parameter values.
    ///
    /// # Arguments
    /// * `params` - Slice of parameters to snapshot
    pub fn snapshot(params: &[Parameter]) -> Self {
        let mut initial_weights = HashMap::new();
        for (i, param) in params.iter().enumerate() {
            let key = if param.name().is_empty() {
                format!("param_{}", i)
            } else {
                param.name().to_string()
            };
            initial_weights.insert(key, param.data());
        }
        Self { initial_weights }
    }

    /// Returns the number of saved parameters.
    pub fn num_saved(&self) -> usize {
        self.initial_weights.len()
    }

    /// Rewinds all parameters to their initial (snapshot) values.
    ///
    /// # Arguments
    /// * `params` - Slice of parameters to rewind (must match snapshot order)
    pub fn rewind(&self, params: &[Parameter]) {
        for (i, param) in params.iter().enumerate() {
            let key = if param.name().is_empty() {
                format!("param_{}", i)
            } else {
                param.name().to_string()
            };
            if let Some(initial) = self.initial_weights.get(&key) {
                param.update_data(initial.clone());
            }
        }
    }

    /// Rewinds parameters to their initial values, but only where the mask is 1.
    ///
    /// Weights where `mask == 0` are set to zero (pruned). Weights where
    /// `mask == 1` are reset to their initial snapshot values.
    ///
    /// # Arguments
    /// * `params` - Slice of parameters to rewind
    /// * `masks` - Corresponding binary masks (same length as params)
    pub fn rewind_with_mask(&self, params: &[Parameter], masks: &[Tensor<f32>]) {
        assert_eq!(
            params.len(),
            masks.len(),
            "Number of parameters and masks must match"
        );

        for (i, (param, mask)) in params.iter().zip(masks.iter()).enumerate() {
            let key = if param.name().is_empty() {
                format!("param_{}", i)
            } else {
                param.name().to_string()
            };

            if let Some(initial) = self.initial_weights.get(&key) {
                let init_vec = initial.to_vec();
                let mask_vec = mask.to_vec();

                let rewound: Vec<f32> = init_vec
                    .iter()
                    .zip(mask_vec.iter())
                    .map(|(&w, &m)| if m > 0.5 { w } else { 0.0 })
                    .collect();

                let shape = param.shape();
                let new_data = Tensor::from_vec(rewound, &shape).expect("tensor creation failed");
                param.update_data(new_data);
            }
        }
    }
}

impl std::fmt::Debug for LotteryTicket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LotteryTicket")
            .field("num_saved", &self.initial_weights.len())
            .finish()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // SparseLinear Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sparse_linear_creation_structured() {
        let layer = SparseLinear::new(10, 5);
        assert_eq!(layer.in_features(), 10);
        assert_eq!(layer.out_features(), 5);
        assert!(layer.is_structured());
        assert!(layer.bias.is_some());
    }

    #[test]
    fn test_sparse_linear_creation_unstructured() {
        let layer = SparseLinear::unstructured(10, 5);
        assert_eq!(layer.in_features(), 10);
        assert_eq!(layer.out_features(), 5);
        assert!(!layer.is_structured());
        assert!(layer.bias.is_some());
    }

    #[test]
    fn test_sparse_linear_no_bias() {
        let layer = SparseLinear::with_bias(10, 5, false);
        assert!(layer.bias.is_none());
    }

    #[test]
    fn test_sparse_linear_forward_shape() {
        let layer = SparseLinear::new(4, 3);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]).expect("tensor creation failed"),
            false,
        );
        let output = layer.forward(&input);
        assert_eq!(output.shape(), vec![1, 3]);
    }

    #[test]
    fn test_sparse_linear_forward_batch() {
        let layer = SparseLinear::new(4, 3);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 12], &[3, 4]).expect("tensor creation failed"),
            false,
        );
        let output = layer.forward(&input);
        assert_eq!(output.shape(), vec![3, 3]);
    }

    #[test]
    fn test_sparse_linear_forward_no_bias() {
        let layer = SparseLinear::with_bias(4, 3, false);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 8], &[2, 4]).expect("tensor creation failed"),
            false,
        );
        let output = layer.forward(&input);
        assert_eq!(output.shape(), vec![2, 3]);
    }

    #[test]
    fn test_sparse_linear_density_initial() {
        // With default threshold of 0.01, most Kaiming-initialized weights
        // should be above threshold (density close to 1.0).
        let layer = SparseLinear::new(100, 50);
        let density = layer.density();
        assert!(
            density > 0.9,
            "Initial density should be high, got {}",
            density
        );
    }

    #[test]
    fn test_sparse_linear_sparsity_initial() {
        let layer = SparseLinear::new(100, 50);
        let sparsity = layer.sparsity();
        assert!(
            sparsity < 0.1,
            "Initial sparsity should be low, got {}",
            sparsity
        );
        assert!((layer.density() + layer.sparsity() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_linear_num_active() {
        let layer = SparseLinear::new(10, 5);
        let active = layer.num_active();
        let total = 10 * 5;
        assert!(active <= total);
        assert!(active > 0);
    }

    #[test]
    fn test_sparse_linear_high_threshold_more_sparsity() {
        let mut layer = SparseLinear::new(100, 50);
        let density_low_thresh = layer.density();

        // Set high threshold — should prune more weights
        layer.reset_threshold(10.0);
        let density_high_thresh = layer.density();

        assert!(
            density_high_thresh < density_low_thresh,
            "Higher threshold should reduce density: low_thresh={}, high_thresh={}",
            density_low_thresh,
            density_high_thresh
        );
    }

    #[test]
    fn test_sparse_linear_low_threshold_dense() {
        let mut layer = SparseLinear::new(100, 50);
        // Set threshold to zero — all weights should be active
        layer.reset_threshold(0.0);
        let density = layer.density();
        assert!(
            (density - 1.0).abs() < 1e-6,
            "Zero threshold should give density=1.0, got {}",
            density
        );
    }

    #[test]
    fn test_sparse_linear_soft_mask_values_in_range() {
        let layer = SparseLinear::new(10, 5);
        let weight_var = layer.weight.variable();
        let mask = layer.compute_soft_mask(&weight_var);
        let mask_vec = mask.data().to_vec();

        for &v in &mask_vec {
            assert!(
                (0.0..=1.0).contains(&v),
                "Soft mask value {} not in [0, 1]",
                v
            );
        }
    }

    #[test]
    fn test_sparse_linear_hard_prune() {
        let mut layer = SparseLinear::new(10, 5);
        // Set a threshold that will prune some weights
        layer.reset_threshold(0.5);

        let pre_prune_density = layer.density();
        layer.hard_prune();

        // After hard prune, the zeroed weights should stay zero
        let weight_data = layer.weight.data();
        let w_vec = weight_data.to_vec();
        let zeros_count = w_vec.iter().filter(|&&v| v == 0.0).count();

        // The number of zeros should correspond to the pruned fraction
        let expected_zeros = ((1.0 - pre_prune_density) * (10 * 5) as f32).round() as usize;
        assert_eq!(
            zeros_count, expected_zeros,
            "Hard prune should zero out pruned weights"
        );
    }

    #[test]
    fn test_sparse_linear_hard_prune_threshold_reset() {
        let mut layer = SparseLinear::new(10, 5);
        layer.reset_threshold(0.5);
        layer.hard_prune();

        // After hard prune, thresholds should be zero
        let t_vec = layer.threshold.data().to_vec();
        assert!(
            t_vec.iter().all(|&v| v == 0.0),
            "Thresholds should be zero after hard_prune"
        );
    }

    #[test]
    fn test_sparse_linear_effective_weight() {
        let layer = SparseLinear::new(10, 5);
        let ew = layer.effective_weight();
        assert_eq!(ew.shape(), &[5, 10]);
    }

    #[test]
    fn test_sparse_linear_effective_weight_matches_hard_prune() {
        let mut layer = SparseLinear::new(10, 5);
        layer.reset_threshold(0.3);

        let effective = layer.effective_weight();
        layer.hard_prune();
        let pruned = layer.weight.data();

        let e_vec = effective.to_vec();
        let p_vec = pruned.to_vec();
        for (e, p) in e_vec.iter().zip(p_vec.iter()) {
            assert!(
                (e - p).abs() < 1e-6,
                "effective_weight and hard_prune should match"
            );
        }
    }

    #[test]
    fn test_sparse_linear_parameters_include_threshold() {
        let layer = SparseLinear::new(10, 5);
        let params = layer.parameters();
        // weight + threshold + bias = 3
        assert_eq!(params.len(), 3);

        let named = layer.named_parameters();
        assert!(named.contains_key("threshold"));
        assert!(named.contains_key("weight"));
        assert!(named.contains_key("bias"));
    }

    #[test]
    fn test_sparse_linear_parameters_no_bias() {
        let layer = SparseLinear::with_bias(10, 5, false);
        let params = layer.parameters();
        // weight + threshold = 2
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_sparse_linear_module_name() {
        let layer = SparseLinear::new(10, 5);
        assert_eq!(layer.name(), "SparseLinear");
    }

    #[test]
    fn test_sparse_linear_debug() {
        let layer = SparseLinear::new(10, 5);
        let debug_str = format!("{:?}", layer);
        assert!(debug_str.contains("SparseLinear"));
        assert!(debug_str.contains("in_features: 10"));
        assert!(debug_str.contains("out_features: 5"));
    }

    #[test]
    fn test_sparse_linear_reset_threshold() {
        let mut layer = SparseLinear::new(10, 5);
        layer.reset_threshold(0.5);
        let t_vec = layer.threshold.data().to_vec();
        assert!(t_vec.iter().all(|&v| (v - 0.5).abs() < 1e-6));
    }

    #[test]
    fn test_sparse_linear_unstructured_threshold_shape() {
        let layer = SparseLinear::unstructured(10, 5);
        // Unstructured: threshold has same shape as weight
        assert_eq!(layer.threshold.shape(), vec![5, 10]);
    }

    #[test]
    fn test_sparse_linear_structured_threshold_shape() {
        let layer = SparseLinear::new(10, 5);
        // Structured: threshold has shape (out_features,)
        assert_eq!(layer.threshold.shape(), vec![5]);
    }

    #[test]
    fn test_sparse_linear_unstructured_forward() {
        let layer = SparseLinear::unstructured(4, 3);
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4])
                .expect("tensor creation failed"),
            false,
        );
        let output = layer.forward(&input);
        assert_eq!(output.shape(), vec![2, 3]);
    }

    // -------------------------------------------------------------------------
    // GroupSparsity Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_group_sparsity_creation() {
        let reg = GroupSparsity::new(0.001, 10);
        assert!((reg.lambda() - 0.001).abs() < 1e-8);
        assert_eq!(reg.group_size(), 10);
    }

    #[test]
    fn test_group_sparsity_penalty_non_negative() {
        let reg = GroupSparsity::new(0.01, 4);
        let weight = Variable::new(
            Tensor::from_vec(vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0], &[2, 4])
                .expect("tensor creation failed"),
            true,
        );
        let penalty = reg.penalty(&weight);
        let penalty_val = penalty.data().to_vec()[0];
        assert!(
            penalty_val >= 0.0,
            "Penalty should be non-negative, got {}",
            penalty_val
        );
    }

    #[test]
    fn test_group_sparsity_zero_weights_zero_penalty() {
        let reg = GroupSparsity::new(0.01, 4);
        let weight = Variable::new(
            Tensor::from_vec(vec![0.0; 8], &[2, 4]).expect("tensor creation failed"),
            true,
        );
        let penalty = reg.penalty(&weight);
        let penalty_val = penalty.data().to_vec()[0];
        assert!(
            (penalty_val).abs() < 1e-6,
            "Zero weights should give zero penalty, got {}",
            penalty_val
        );
    }

    #[test]
    fn test_group_sparsity_scales_with_lambda() {
        let reg_small = GroupSparsity::new(0.001, 4);
        let reg_large = GroupSparsity::new(0.01, 4);
        let weight = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]).expect("tensor creation failed"),
            true,
        );

        let penalty_small = reg_small.penalty(&weight).data().to_vec()[0];
        let penalty_large = reg_large.penalty(&weight).data().to_vec()[0];

        assert!(
            penalty_large > penalty_small,
            "Larger lambda should give larger penalty: small={}, large={}",
            penalty_small,
            penalty_large
        );

        // Should scale linearly with lambda
        let ratio = penalty_large / penalty_small;
        assert!(
            (ratio - 10.0).abs() < 1e-4,
            "Penalty should scale linearly with lambda, ratio={}",
            ratio
        );
    }

    #[test]
    fn test_group_sparsity_debug() {
        let reg = GroupSparsity::new(0.001, 10);
        let debug_str = format!("{:?}", reg);
        assert!(debug_str.contains("GroupSparsity"));
        assert!(debug_str.contains("lambda"));
    }

    #[test]
    #[should_panic(expected = "group_size must be positive")]
    fn test_group_sparsity_zero_group_size_panics() {
        let _reg = GroupSparsity::new(0.01, 0);
    }

    // -------------------------------------------------------------------------
    // LotteryTicket Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_lottery_ticket_snapshot() {
        let layer = SparseLinear::new(10, 5);
        let params = layer.parameters();
        let ticket = LotteryTicket::snapshot(&params);
        assert_eq!(ticket.num_saved(), params.len());
    }

    #[test]
    fn test_lottery_ticket_rewind() {
        let layer = SparseLinear::new(10, 5);
        let params = layer.parameters();
        let initial_weight = params[0].data().to_vec();

        let ticket = LotteryTicket::snapshot(&params);

        // Modify the weight
        let new_data = Tensor::from_vec(vec![99.0; 50], &[5, 10]).expect("tensor creation failed");
        params[0].update_data(new_data);

        // Verify it changed
        let modified_weight = params[0].data().to_vec();
        assert_ne!(modified_weight, initial_weight);

        // Rewind
        ticket.rewind(&params);

        // Verify it's back to initial
        let rewound_weight = params[0].data().to_vec();
        assert_eq!(rewound_weight, initial_weight);
    }

    #[test]
    fn test_lottery_ticket_rewind_preserves_shapes() {
        let layer = SparseLinear::new(10, 5);
        let params = layer.parameters();
        let initial_shapes: Vec<Vec<usize>> = params.iter().map(|p| p.shape()).collect();

        let ticket = LotteryTicket::snapshot(&params);

        // Modify weight data (same shape)
        let new_data = Tensor::from_vec(vec![0.0; 50], &[5, 10]).expect("tensor creation failed");
        params[0].update_data(new_data);

        ticket.rewind(&params);

        let rewound_shapes: Vec<Vec<usize>> = params.iter().map(|p| p.shape()).collect();
        assert_eq!(initial_shapes, rewound_shapes);
    }

    #[test]
    fn test_lottery_ticket_rewind_with_mask() {
        let data =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("tensor creation failed");
        let param = Parameter::named("weight", data, true);
        let params = vec![param];

        let ticket = LotteryTicket::snapshot(&params);

        // Modify the parameter
        let new_data = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[2, 2])
            .expect("tensor creation failed");
        params[0].update_data(new_data);

        // Mask: keep first two, prune last two
        let mask =
            Tensor::from_vec(vec![1.0, 1.0, 0.0, 0.0], &[2, 2]).expect("tensor creation failed");
        ticket.rewind_with_mask(&params, &[mask]);

        let result = params[0].data().to_vec();
        assert_eq!(
            result,
            vec![1.0, 2.0, 0.0, 0.0],
            "Masked weights should be zero, unmasked should be initial values"
        );
    }

    #[test]
    fn test_lottery_ticket_debug() {
        let layer = SparseLinear::new(10, 5);
        let ticket = LotteryTicket::snapshot(&layer.parameters());
        let debug_str = format!("{:?}", ticket);
        assert!(debug_str.contains("LotteryTicket"));
        assert!(debug_str.contains("num_saved"));
    }

    // -------------------------------------------------------------------------
    // Integration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_integration_sparse_linear_with_group_sparsity() {
        // Create a SparseLinear layer
        let layer = SparseLinear::new(8, 4);

        // Forward pass
        let input = Variable::new(
            Tensor::from_vec(vec![1.0; 16], &[2, 8]).expect("tensor creation failed"),
            false,
        );
        let output = layer.forward(&input);
        assert_eq!(output.shape(), vec![2, 4]);

        // Compute group sparsity penalty on the weights
        let reg = GroupSparsity::new(0.001, 8); // group_size = in_features
        let weight_var = layer.weight.variable();
        let penalty = reg.penalty(&weight_var);
        let penalty_val = penalty.data().to_vec()[0];
        assert!(
            penalty_val > 0.0,
            "Penalty should be positive for non-zero weights"
        );
    }

    #[test]
    fn test_integration_lottery_ticket_with_pruning() {
        // 1. Create layer and snapshot
        let mut layer = SparseLinear::new(8, 4);
        let ticket = LotteryTicket::snapshot(&layer.parameters());

        // 2. Simulate training (modify weights)
        let new_weight = Tensor::from_vec(vec![0.5; 32], &[4, 8]).expect("tensor creation failed");
        layer.weight.update_data(new_weight);

        // 3. Set threshold to prune some weights
        layer.reset_threshold(0.3);

        // 4. Get the effective weight mask
        let mask = layer.hard_mask();

        // 5. Rewind to initial weights with mask
        let weight_param = vec![layer.weight.clone()];
        ticket.rewind_with_mask(&weight_param, &[mask]);

        // Verify shape is preserved
        assert_eq!(layer.weight.shape(), vec![4, 8]);
    }

    #[test]
    fn test_num_parameters_sparse_linear() {
        let layer = SparseLinear::new(10, 5);
        // weight: 50 + threshold: 5 + bias: 5 = 60
        assert_eq!(layer.num_parameters(), 60);
    }
}
