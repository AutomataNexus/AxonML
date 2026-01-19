//! Backward Pass - Gradient Computation
//!
//! Implements the backward pass (backpropagation) algorithm for computing
//! gradients through the computational graph using reverse-mode autodiff.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::collections::{HashMap, HashSet};

use axonml_tensor::Tensor;

use crate::grad_fn::{GradFn, GradFnId};
use crate::variable::Variable;

// =============================================================================
// Backward Function
// =============================================================================

/// Computes gradients for all leaf variables in the graph.
///
/// Traverses the computational graph in reverse topological order,
/// computing and accumulating gradients for each variable.
///
/// # Arguments
/// * `output` - The output variable (typically the loss)
/// * `grad_output` - The gradient of the loss with respect to output (typically 1.0)
pub fn backward(output: &Variable, grad_output: &Tensor<f32>) {
    // Get the gradient function for the output
    let grad_fn = if let Some(gf) = output.grad_fn() { gf.clone() } else {
        // Leaf variable - accumulate gradient directly
        if output.is_leaf() && output.requires_grad() {
            output.accumulate_grad(grad_output);
        }
        return;
    };

    // Build the topological order of nodes
    let mut topo_order: Vec<GradFn> = Vec::new();
    let mut visited: HashSet<GradFnId> = HashSet::new();
    build_topo_order(&grad_fn, &mut topo_order, &mut visited);

    // Initialize gradient map with the output gradient
    // Use stable IDs based on Arc pointer, not struct address
    let mut grad_map: HashMap<GradFnId, Tensor<f32>> = HashMap::new();
    let output_id = grad_fn.id();
    grad_map.insert(output_id, grad_output.clone());

    // Process nodes in reverse topological order
    for node in topo_order.iter().rev() {
        let node_id = node.id();

        // Get the accumulated gradient for this node
        let grad = match grad_map.get(&node_id) {
            Some(g) => g.clone(),
            None => continue, // No gradient to propagate
        };

        // Apply the gradient function to get input gradients
        let input_grads = node.apply(&grad);

        // Propagate gradients to input nodes
        let next_fns = node.next_functions();
        for (i, maybe_next) in next_fns.iter().enumerate() {
            if let Some(next_fn) = maybe_next {
                if let Some(input_grad) = input_grads.get(i).and_then(std::clone::Clone::clone) {
                    let next_id = next_fn.id();

                    // Accumulate gradient
                    grad_map
                        .entry(next_id)
                        .and_modify(|existing| {
                            *existing = existing.add(&input_grad).unwrap();
                        })
                        .or_insert(input_grad);
                }
            }
        }
    }
}

/// Builds the topological order of nodes in the graph using DFS.
fn build_topo_order(node: &GradFn, order: &mut Vec<GradFn>, visited: &mut HashSet<GradFnId>) {
    let node_id = node.id();

    if visited.contains(&node_id) {
        return;
    }
    visited.insert(node_id);

    // Visit all input nodes first
    for next in node.next_functions().iter().flatten() {
        build_topo_order(next, order, visited);
    }

    // Add this node after its inputs
    order.push(node.clone());
}

// =============================================================================
// Gradient Checking
// =============================================================================

/// Numerically checks the gradient computation using finite differences.
///
/// This is useful for verifying that analytical gradients are correct.
///
/// # Arguments
/// * `func` - Function to differentiate
/// * `input` - Input variable
/// * `eps` - Epsilon for finite differences (default: 1e-5)
///
/// # Returns
/// Numerical gradient approximation
pub fn numerical_gradient<F>(func: F, input: &Variable, eps: f32) -> Tensor<f32>
where
    F: Fn(&Variable) -> Variable,
{
    let input_data = input.data();
    let mut grad_data = vec![0.0f32; input_data.numel()];

    for i in 0..input_data.numel() {
        // f(x + eps)
        let mut plus_data = input_data.to_vec();
        plus_data[i] += eps;
        let plus_input =
            Variable::from_tensor(Tensor::from_vec(plus_data, input_data.shape()).unwrap());
        let plus_output = func(&plus_input);
        let plus_val = plus_output.data().to_vec()[0];

        // f(x - eps)
        let mut minus_data = input_data.to_vec();
        minus_data[i] -= eps;
        let minus_input =
            Variable::from_tensor(Tensor::from_vec(minus_data, input_data.shape()).unwrap());
        let minus_output = func(&minus_input);
        let minus_val = minus_output.data().to_vec()[0];

        // Central difference
        grad_data[i] = (plus_val - minus_val) / (2.0 * eps);
    }

    Tensor::from_vec(grad_data, input_data.shape()).unwrap()
}

/// Checks if analytical and numerical gradients match.
///
/// # Arguments
/// * `analytical` - Analytically computed gradient
/// * `numerical` - Numerically computed gradient
/// * `rtol` - Relative tolerance
/// * `atol` - Absolute tolerance
///
/// # Returns
/// True if gradients match within tolerance
#[must_use] pub fn gradcheck(analytical: &Tensor<f32>, numerical: &Tensor<f32>, rtol: f32, atol: f32) -> bool {
    if analytical.shape() != numerical.shape() {
        return false;
    }

    let a = analytical.to_vec();
    let n = numerical.to_vec();

    for (&av, &nv) in a.iter().zip(n.iter()) {
        let diff = (av - nv).abs();
        let tol = atol + rtol * nv.abs();
        if diff > tol {
            return false;
        }
    }

    true
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_backward() {
        // y = x^2, dy/dx = 2x
        let x = Variable::new(Tensor::from_vec(vec![3.0], &[1]).unwrap(), true);
        let y = x.pow(2.0);

        y.backward();

        // dy/dx at x=3 should be 6
        let grad = x.grad().unwrap();
        assert!((grad.to_vec()[0] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_chain_backward() {
        // y = (x^2)^2 = x^4, dy/dx = 4x^3
        let x = Variable::new(Tensor::from_vec(vec![2.0], &[1]).unwrap(), true);
        let y = x.pow(2.0).pow(2.0);

        y.backward();

        // dy/dx at x=2 should be 4 * 8 = 32
        let grad = x.grad().unwrap();
        assert!((grad.to_vec()[0] - 32.0).abs() < 1e-4);
    }

    #[test]
    fn test_add_backward() {
        let a = Variable::new(Tensor::from_vec(vec![2.0], &[1]).unwrap(), true);
        let b = Variable::new(Tensor::from_vec(vec![3.0], &[1]).unwrap(), true);
        let c = &a + &b;
        let loss = c.sum();

        loss.backward();

        // d(a+b)/da = 1, d(a+b)/db = 1
        assert!((a.grad().unwrap().to_vec()[0] - 1.0).abs() < 1e-5);
        assert!((b.grad().unwrap().to_vec()[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_mul_backward() {
        let a = Variable::new(Tensor::from_vec(vec![2.0], &[1]).unwrap(), true);
        let b = Variable::new(Tensor::from_vec(vec![3.0], &[1]).unwrap(), true);
        let c = &a * &b;
        let loss = c.sum();

        loss.backward();

        // d(a*b)/da = b = 3, d(a*b)/db = a = 2
        assert!((a.grad().unwrap().to_vec()[0] - 3.0).abs() < 1e-5);
        assert!((b.grad().unwrap().to_vec()[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_numerical_gradient() {
        let x = Variable::new(Tensor::from_vec(vec![2.0, 3.0], &[2]).unwrap(), false);

        let numerical = numerical_gradient(|v| v.pow(2.0).sum(), &x, 1e-5);

        // d(x^2)/dx = 2x, so [4.0, 6.0]
        // Use 1e-2 tolerance due to floating point precision in f32 pow operations
        let expected = [4.0, 6.0];
        for (i, &n) in numerical.to_vec().iter().enumerate() {
            assert!(
                (n - expected[i]).abs() < 1e-2,
                "Numerical gradient check failed: got {}, expected {}",
                n,
                expected[i]
            );
        }
    }

    #[test]
    fn test_gradcheck() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::from_vec(vec![1.001, 2.001, 3.001], &[3]).unwrap();

        assert!(gradcheck(&a, &b, 0.01, 0.01));
        assert!(!gradcheck(&a, &b, 0.0001, 0.0001));
    }
}
