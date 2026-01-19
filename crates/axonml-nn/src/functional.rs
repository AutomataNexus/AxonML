//! Functional API - Stateless Neural Network Operations
//!
//! Provides functional versions of neural network operations
//! that don't require module state.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use axonml_autograd::Variable;
use axonml_tensor::Tensor;

// =============================================================================
// Activation Functions
// =============================================================================

/// ReLU activation function.
pub fn relu(input: &Variable) -> Variable {
    input.relu()
}

/// Leaky ReLU activation function.
pub fn leaky_relu(input: &Variable, negative_slope: f32) -> Variable {
    let data = input.data();
    let result: Vec<f32> = data
        .to_vec()
        .iter()
        .map(|&x| if x > 0.0 { x } else { x * negative_slope })
        .collect();
    Variable::new(
        Tensor::from_vec(result, data.shape()).unwrap(),
        input.requires_grad(),
    )
}

/// Sigmoid activation function.
pub fn sigmoid(input: &Variable) -> Variable {
    input.sigmoid()
}

/// Tanh activation function.
pub fn tanh(input: &Variable) -> Variable {
    input.tanh()
}

/// GELU activation function.
pub fn gelu(input: &Variable) -> Variable {
    let data = input.data();
    let data_vec = data.to_vec();
    // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
    let result: Vec<f32> = data_vec
        .iter()
        .map(|&x| {
            let inner = sqrt_2_over_pi * (x + 0.044715 * x.powi(3));
            0.5 * x * (1.0 + inner.tanh())
        })
        .collect();
    Variable::new(
        Tensor::from_vec(result, data.shape()).unwrap(),
        input.requires_grad(),
    )
}

/// SiLU (Swish) activation function.
pub fn silu(input: &Variable) -> Variable {
    let sigmoid = input.sigmoid();
    input.mul_var(&sigmoid)
}

/// ELU activation function.
pub fn elu(input: &Variable, alpha: f32) -> Variable {
    let data = input.data();
    let result: Vec<f32> = data
        .to_vec()
        .iter()
        .map(|&x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) })
        .collect();
    Variable::new(
        Tensor::from_vec(result, data.shape()).unwrap(),
        input.requires_grad(),
    )
}

/// Softmax along a dimension.
pub fn softmax(input: &Variable, dim: i64) -> Variable {
    let data = input.data();
    let shape = data.shape().to_vec();
    let data_vec = data.to_vec();

    let ndim = shape.len();
    let dim_idx = if dim < 0 {
        (ndim as i64 + dim) as usize
    } else {
        dim as usize
    };

    let outer_size: usize = shape[..dim_idx].iter().product();
    let dim_size = shape[dim_idx];
    let inner_size: usize = shape[dim_idx + 1..].iter().product();

    let mut result = vec![0.0f32; data_vec.len()];

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let mut max_val = f32::NEG_INFINITY;
            for d in 0..dim_size {
                let idx = outer * dim_size * inner_size.max(1) + d * inner_size.max(1) + inner;
                if idx < data_vec.len() {
                    max_val = max_val.max(data_vec[idx]);
                }
            }

            let mut sum = 0.0f32;
            for d in 0..dim_size {
                let idx = outer * dim_size * inner_size.max(1) + d * inner_size.max(1) + inner;
                if idx < data_vec.len() {
                    let exp_val = (data_vec[idx] - max_val).exp();
                    result[idx] = exp_val;
                    sum += exp_val;
                }
            }

            for d in 0..dim_size {
                let idx = outer * dim_size * inner_size.max(1) + d * inner_size.max(1) + inner;
                if idx < result.len() {
                    result[idx] /= sum;
                }
            }
        }
    }

    Variable::new(
        Tensor::from_vec(result, &shape).unwrap(),
        input.requires_grad(),
    )
}

/// Log softmax along a dimension.
pub fn log_softmax(input: &Variable, dim: i64) -> Variable {
    let sm = softmax(input, dim);
    let sm_vec = sm.data().to_vec();
    let result: Vec<f32> = sm_vec.iter().map(|&x| x.ln()).collect();
    Variable::new(
        Tensor::from_vec(result, sm.data().shape()).unwrap(),
        input.requires_grad(),
    )
}

// =============================================================================
// Linear Operations
// =============================================================================

/// Linear transformation: y = xA^T + b
pub fn linear(input: &Variable, weight: &Variable, bias: Option<&Variable>) -> Variable {
    let weight_t = Variable::new(weight.data().t().unwrap(), weight.requires_grad());
    let mut output = input.matmul(&weight_t);
    if let Some(b) = bias {
        output = output.add_var(b);
    }
    output
}

// =============================================================================
// Normalization
// =============================================================================

/// Layer normalization.
pub fn layer_norm(
    input: &Variable,
    normalized_shape: &[usize],
    weight: Option<&Variable>,
    bias: Option<&Variable>,
    eps: f32,
) -> Variable {
    let data = input.data();
    let shape = data.shape().to_vec();
    let data_vec = data.to_vec();

    let norm_size: usize = normalized_shape.iter().product();
    let batch_size = data_vec.len() / norm_size;

    let mut result = vec![0.0f32; data_vec.len()];

    for b in 0..batch_size {
        let start = b * norm_size;
        let end = start + norm_size;
        let slice = &data_vec[start..end];

        let mean: f32 = slice.iter().sum::<f32>() / norm_size as f32;
        let var: f32 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / norm_size as f32;

        for i in 0..norm_size {
            let normalized = (slice[i] - mean) / (var + eps).sqrt();
            result[start + i] = normalized;
        }
    }

    let mut output = Variable::new(
        Tensor::from_vec(result, &shape).unwrap(),
        input.requires_grad(),
    );

    // Apply affine transform if provided
    if let Some(w) = weight {
        output = output.mul_var(w);
    }
    if let Some(b) = bias {
        output = output.add_var(b);
    }

    output
}

// =============================================================================
// Dropout
// =============================================================================

/// Dropout during training.
pub fn dropout(input: &Variable, p: f32, training: bool) -> Variable {
    if !training || p == 0.0 {
        return input.clone();
    }

    let data = input.data();
    let data_vec = data.to_vec();
    let scale = 1.0 / (1.0 - p);

    let mut rng = rand::thread_rng();
    use rand::Rng;

    let result: Vec<f32> = data_vec
        .iter()
        .map(|&x| if rng.gen::<f32>() < p { 0.0 } else { x * scale })
        .collect();

    Variable::new(
        Tensor::from_vec(result, data.shape()).unwrap(),
        input.requires_grad(),
    )
}

// =============================================================================
// Loss Functions
// =============================================================================

/// Mean squared error loss.
pub fn mse_loss(input: &Variable, target: &Variable) -> Variable {
    input.mse_loss(target)
}

/// Cross entropy loss.
pub fn cross_entropy(input: &Variable, target: &Variable) -> Variable {
    let input_data = input.data();
    let target_data = target.data();
    let shape = input_data.shape().to_vec();
    let batch_size = shape[0];
    let num_classes = shape[1];

    let input_vec = input_data.to_vec();
    let target_vec = target_data.to_vec();

    let mut total_loss = 0.0f32;

    for b in 0..batch_size {
        let offset = b * num_classes;
        let max_val = (0..num_classes)
            .map(|c| input_vec[offset + c])
            .fold(f32::NEG_INFINITY, f32::max);

        let mut log_sum_exp = 0.0f32;
        for c in 0..num_classes {
            log_sum_exp += (input_vec[offset + c] - max_val).exp();
        }
        log_sum_exp = max_val + log_sum_exp.ln();

        let target_class = target_vec[b] as usize;
        total_loss += log_sum_exp - input_vec[offset + target_class];
    }

    Variable::new(
        Tensor::scalar(total_loss / batch_size as f32),
        input.requires_grad(),
    )
}

/// Binary cross entropy loss.
pub fn binary_cross_entropy(input: &Variable, target: &Variable) -> Variable {
    input.binary_cross_entropy(target)
}

// =============================================================================
// Pooling
// =============================================================================

/// Adaptive average pooling to output size.
pub fn adaptive_avg_pool2d(input: &Variable, output_size: (usize, usize)) -> Variable {
    let shape = input.shape();
    let batch = shape[0];
    let channels = shape[1];
    let in_h = shape[2];
    let in_w = shape[3];
    let (out_h, out_w) = output_size;

    let input_vec = input.data().to_vec();
    let mut output_data = vec![0.0f32; batch * channels * out_h * out_w];

    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let ih_start = (oh * in_h) / out_h;
                    let ih_end = ((oh + 1) * in_h) / out_h;
                    let iw_start = (ow * in_w) / out_w;
                    let iw_end = ((ow + 1) * in_w) / out_w;

                    let mut sum = 0.0f32;
                    let mut count = 0;

                    for ih in ih_start..ih_end {
                        for iw in iw_start..iw_end {
                            let idx = b * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw;
                            sum += input_vec[idx];
                            count += 1;
                        }
                    }

                    let out_idx =
                        b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                    output_data[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                }
            }
        }
    }

    Variable::new(
        Tensor::from_vec(output_data, &[batch, channels, out_h, out_w]).unwrap(),
        input.requires_grad(),
    )
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_functional() {
        let input = Variable::new(Tensor::from_vec(vec![-1.0, 0.0, 1.0], &[3]).unwrap(), false);
        let output = relu(&input);
        assert_eq!(output.data().to_vec(), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_softmax_functional() {
        let input = Variable::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap(),
            false,
        );
        let output = softmax(&input, -1);
        let sum: f32 = output.data().to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_dropout_functional() {
        let input = Variable::new(Tensor::from_vec(vec![1.0; 100], &[100]).unwrap(), false);
        let output = dropout(&input, 0.5, true);
        let output_vec = output.data().to_vec();
        let num_zeros = output_vec.iter().filter(|&&x| x == 0.0).count();
        assert!(num_zeros > 30 && num_zeros < 70);
    }

    #[test]
    fn test_mse_loss_functional() {
        let input = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), false);
        let target = Variable::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), false);
        let loss = mse_loss(&input, &target);
        assert!((loss.data().to_vec()[0] - 0.0).abs() < 1e-6);
    }
}
