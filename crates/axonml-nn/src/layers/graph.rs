//! Graph Neural Network Layers
//!
//! Provides graph convolution layers for learning on graph-structured data.
//! Includes GCN (Graph Convolutional Network) and GAT (Graph Attention Network).
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use std::collections::HashMap;

use axonml_autograd::Variable;
use axonml_tensor::Tensor;

use crate::module::Module;
use crate::parameter::Parameter;


// =============================================================================
// GCNConv
// =============================================================================

/// Graph Convolutional Network layer (Kipf & Welling, 2017).
///
/// Performs the graph convolution: `output = adj @ x @ weight + bias`
/// where `adj` is the (possibly learned) adjacency matrix.
///
/// # Arguments
/// * `in_features` - Number of input features per node
/// * `out_features` - Number of output features per node
///
/// # Forward Signature
/// Uses `forward_graph(x, adj)` since it requires an adjacency matrix.
/// The standard `forward()` from Module is provided but panics—use `forward_graph()`.
///
/// # Example
/// ```ignore
/// use axonml_nn::layers::GCNConv;
///
/// let gcn = GCNConv::new(72, 128);
/// let x = Variable::new(Tensor::randn(&[2, 7, 72]), true);     // (batch, nodes, features)
/// let adj = Variable::new(Tensor::ones(&[7, 7]), false);         // (nodes, nodes)
/// let output = gcn.forward_graph(&x, &adj);
/// // output shape: (2, 7, 128)
/// ```
pub struct GCNConv {
    weight: Parameter,
    bias: Option<Parameter>,
    in_features: usize,
    out_features: usize,
}

impl GCNConv {
    /// Creates a new GCN convolution layer with bias.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        // Xavier initialization
        let scale = (2.0 / (in_features + out_features) as f32).sqrt();
        let weight_data: Vec<f32> = (0..in_features * out_features)
            .map(|i| {
                // Simple deterministic-ish init for reproducibility
                let x = ((i as f32 * 0.6180339887) % 1.0) * 2.0 - 1.0;
                x * scale
            })
            .collect();

        let weight = Parameter::named(
            "weight",
            Tensor::from_vec(weight_data, &[in_features, out_features]).unwrap(),
            true,
        );

        let bias_data = vec![0.0; out_features];
        let bias = Some(Parameter::named(
            "bias",
            Tensor::from_vec(bias_data, &[out_features]).unwrap(),
            true,
        ));

        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    /// Creates a GCN layer without bias.
    pub fn without_bias(in_features: usize, out_features: usize) -> Self {
        let scale = (2.0 / (in_features + out_features) as f32).sqrt();
        let weight_data: Vec<f32> = (0..in_features * out_features)
            .map(|i| {
                let x = ((i as f32 * 0.6180339887) % 1.0) * 2.0 - 1.0;
                x * scale
            })
            .collect();

        let weight = Parameter::named(
            "weight",
            Tensor::from_vec(weight_data, &[in_features, out_features]).unwrap(),
            true,
        );

        Self {
            weight,
            bias: None,
            in_features,
            out_features,
        }
    }

    /// Graph convolution forward pass.
    ///
    /// # Arguments
    /// * `x` - Node features: `(batch, num_nodes, in_features)`
    /// * `adj` - Adjacency matrix: `(num_nodes, num_nodes)` or `(batch, num_nodes, num_nodes)`
    ///
    /// # Returns
    /// Output features: `(batch, num_nodes, out_features)`
    pub fn forward_graph(&self, x: &Variable, adj: &Variable) -> Variable {
        let shape = x.shape();
        assert!(shape.len() == 3, "GCNConv expects input shape (batch, nodes, features), got {:?}", shape);
        assert_eq!(shape[2], self.in_features, "Input features mismatch");

        let batch = shape[0];
        let nodes = shape[1];
        let adj_shape = adj.shape();

        let x_data = x.data().to_vec();
        let adj_data = adj.data().to_vec();
        let w_data = self.weight.data().to_vec();

        let mut output = vec![0.0f32; batch * nodes * self.out_features];

        for b in 0..batch {
            // Get adjacency for this batch
            let adj_offset = if adj_shape.len() == 3 {
                b * nodes * nodes
            } else {
                0 // shared adjacency
            };

            // Step 1: message = adj @ x  → (nodes, in_features)
            // Step 2: output = message @ weight → (nodes, out_features)
            for i in 0..nodes {
                // Aggregate neighbor features: message_i = sum_j adj[i,j] * x[b,j,:]
                let mut message = vec![0.0f32; self.in_features];
                for j in 0..nodes {
                    let a_ij = adj_data[adj_offset + i * nodes + j];
                    if a_ij != 0.0 {
                        let x_offset = (b * nodes + j) * self.in_features;
                        for f in 0..self.in_features {
                            message[f] += a_ij * x_data[x_offset + f];
                        }
                    }
                }

                // Transform: out_i = message_i @ weight
                let out_offset = (b * nodes + i) * self.out_features;
                for o in 0..self.out_features {
                    let mut val = 0.0;
                    for f in 0..self.in_features {
                        val += message[f] * w_data[f * self.out_features + o];
                    }
                    output[out_offset + o] = val;
                }
            }
        }

        // Add bias
        if let Some(bias) = &self.bias {
            let bias_data = bias.data().to_vec();
            for b in 0..batch {
                for i in 0..nodes {
                    let offset = (b * nodes + i) * self.out_features;
                    for o in 0..self.out_features {
                        output[offset + o] += bias_data[o];
                    }
                }
            }
        }

        Variable::new(
            Tensor::from_vec(output, &[batch, nodes, self.out_features]).unwrap(),
            x.requires_grad() || adj.requires_grad(),
        )
    }

    /// Returns the input feature dimension.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Returns the output feature dimension.
    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

impl Module for GCNConv {
    fn forward(&self, _input: &Variable) -> Variable {
        panic!("GCNConv requires an adjacency matrix. Use forward_graph(x, adj) instead.")
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![self.weight.clone()];
        if let Some(bias) = &self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.insert("weight".to_string(), self.weight.clone());
        if let Some(bias) = &self.bias {
            params.insert("bias".to_string(), bias.clone());
        }
        params
    }

    fn name(&self) -> &'static str {
        "GCNConv"
    }
}

// =============================================================================
// GATConv
// =============================================================================

/// Graph Attention Network layer (Veličković et al., 2018).
///
/// Computes attention-weighted graph convolution where attention coefficients
/// are learned based on node features, masked by the adjacency matrix.
///
/// # Example
/// ```ignore
/// use axonml_nn::layers::GATConv;
///
/// let gat = GATConv::new(72, 32, 4); // 4 attention heads
/// let x = Variable::new(Tensor::randn(&[2, 7, 72]), true);
/// let adj = Variable::new(Tensor::ones(&[7, 7]), false);
/// let output = gat.forward_graph(&x, &adj);
/// // output shape: (2, 7, 128)  — 32 * 4 heads
/// ```
pub struct GATConv {
    w: Parameter,
    attn_src: Parameter,
    attn_dst: Parameter,
    bias: Option<Parameter>,
    in_features: usize,
    out_features: usize,
    num_heads: usize,
    negative_slope: f32,
}

impl GATConv {
    /// Creates a new GAT convolution layer.
    ///
    /// # Arguments
    /// * `in_features` - Input feature dimension per node
    /// * `out_features` - Output feature dimension per head
    /// * `num_heads` - Number of attention heads (output = out_features * num_heads)
    pub fn new(in_features: usize, out_features: usize, num_heads: usize) -> Self {
        let total_out = out_features * num_heads;
        let scale = (2.0 / (in_features + total_out) as f32).sqrt();

        let w_data: Vec<f32> = (0..in_features * total_out)
            .map(|i| {
                let x = ((i as f32 * 0.6180339887) % 1.0) * 2.0 - 1.0;
                x * scale
            })
            .collect();

        let w = Parameter::named(
            "w",
            Tensor::from_vec(w_data, &[in_features, total_out]).unwrap(),
            true,
        );

        // Attention vectors: one per head for source and destination
        let attn_scale = (1.0 / out_features as f32).sqrt();
        let attn_src_data: Vec<f32> = (0..total_out)
            .map(|i| {
                let x = ((i as f32 * 0.7236067977) % 1.0) * 2.0 - 1.0;
                x * attn_scale
            })
            .collect();
        let attn_dst_data: Vec<f32> = (0..total_out)
            .map(|i| {
                let x = ((i as f32 * 0.3819660113) % 1.0) * 2.0 - 1.0;
                x * attn_scale
            })
            .collect();

        let attn_src = Parameter::named(
            "attn_src",
            Tensor::from_vec(attn_src_data, &[num_heads, out_features]).unwrap(),
            true,
        );

        let attn_dst = Parameter::named(
            "attn_dst",
            Tensor::from_vec(attn_dst_data, &[num_heads, out_features]).unwrap(),
            true,
        );

        let bias_data = vec![0.0; total_out];
        let bias = Some(Parameter::named(
            "bias",
            Tensor::from_vec(bias_data, &[total_out]).unwrap(),
            true,
        ));

        Self {
            w,
            attn_src,
            attn_dst,
            bias,
            in_features,
            out_features,
            num_heads,
            negative_slope: 0.2,
        }
    }

    /// Graph attention forward pass.
    ///
    /// # Arguments
    /// * `x` - Node features: `(batch, num_nodes, in_features)`
    /// * `adj` - Adjacency mask: `(num_nodes, num_nodes)` — non-zero entries allow attention
    ///
    /// # Returns
    /// Output features: `(batch, num_nodes, out_features * num_heads)`
    pub fn forward_graph(&self, x: &Variable, adj: &Variable) -> Variable {
        let shape = x.shape();
        assert!(shape.len() == 3, "GATConv expects (batch, nodes, features), got {:?}", shape);

        let batch = shape[0];
        let nodes = shape[1];
        let total_out = self.out_features * self.num_heads;

        let x_data = x.data().to_vec();
        let adj_data = adj.data().to_vec();
        let w_data = self.w.data().to_vec();
        let attn_src_data = self.attn_src.data().to_vec();
        let attn_dst_data = self.attn_dst.data().to_vec();

        let adj_nodes = if adj.shape().len() == 3 { adj.shape()[1] } else { adj.shape()[0] };
        assert_eq!(adj_nodes, nodes, "Adjacency matrix size mismatch");

        let mut output = vec![0.0f32; batch * nodes * total_out];

        for b in 0..batch {
            // Step 1: Project all nodes: h = x @ w → (nodes, total_out)
            let mut h = vec![0.0f32; nodes * total_out];
            for i in 0..nodes {
                let x_off = (b * nodes + i) * self.in_features;
                for o in 0..total_out {
                    let mut val = 0.0;
                    for f in 0..self.in_features {
                        val += x_data[x_off + f] * w_data[f * total_out + o];
                    }
                    h[i * total_out + o] = val;
                }
            }

            // Step 2: Compute attention per head
            let adj_off = if adj.shape().len() == 3 { b * nodes * nodes } else { 0 };

            for head in 0..self.num_heads {
                let head_off = head * self.out_features;

                // Compute attention scores for each edge
                // e_ij = LeakyReLU(attn_src · h_i + attn_dst · h_j)
                let mut attn_scores = vec![f32::NEG_INFINITY; nodes * nodes];

                for i in 0..nodes {
                    // src score for node i
                    let mut src_score = 0.0;
                    for f in 0..self.out_features {
                        src_score += h[i * total_out + head_off + f] * attn_src_data[head * self.out_features + f];
                    }

                    for j in 0..nodes {
                        let a_ij = adj_data[adj_off + i * nodes + j];
                        if a_ij != 0.0 {
                            let mut dst_score = 0.0;
                            for f in 0..self.out_features {
                                dst_score += h[j * total_out + head_off + f] * attn_dst_data[head * self.out_features + f];
                            }

                            let e = src_score + dst_score;
                            // LeakyReLU
                            let e = if e > 0.0 { e } else { e * self.negative_slope };
                            attn_scores[i * nodes + j] = e;
                        }
                    }
                }

                // Softmax per row (per destination node)
                for i in 0..nodes {
                    let row_start = i * nodes;
                    let row_end = row_start + nodes;
                    let row = &attn_scores[row_start..row_end];

                    let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    if max_val == f32::NEG_INFINITY {
                        continue; // No neighbors
                    }

                    let mut sum_exp = 0.0f32;
                    let mut exps = vec![0.0; nodes];
                    for j in 0..nodes {
                        if row[j] > f32::NEG_INFINITY {
                            exps[j] = (row[j] - max_val).exp();
                            sum_exp += exps[j];
                        }
                    }

                    // Weighted sum of neighbor features
                    let out_off = (b * nodes + i) * total_out + head_off;
                    for j in 0..nodes {
                        if exps[j] > 0.0 {
                            let alpha = exps[j] / sum_exp;
                            for f in 0..self.out_features {
                                output[out_off + f] += alpha * h[j * total_out + head_off + f];
                            }
                        }
                    }
                }
            }
        }

        // Add bias
        if let Some(bias) = &self.bias {
            let bias_data = bias.data().to_vec();
            for b in 0..batch {
                for i in 0..nodes {
                    let offset = (b * nodes + i) * total_out;
                    for o in 0..total_out {
                        output[offset + o] += bias_data[o];
                    }
                }
            }
        }

        Variable::new(
            Tensor::from_vec(output, &[batch, nodes, total_out]).unwrap(),
            x.requires_grad(),
        )
    }

    /// Total output dimension (out_features * num_heads).
    pub fn total_out_features(&self) -> usize {
        self.out_features * self.num_heads
    }
}

impl Module for GATConv {
    fn forward(&self, _input: &Variable) -> Variable {
        panic!("GATConv requires an adjacency matrix. Use forward_graph(x, adj) instead.")
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![self.w.clone(), self.attn_src.clone(), self.attn_dst.clone()];
        if let Some(bias) = &self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.insert("w".to_string(), self.w.clone());
        params.insert("attn_src".to_string(), self.attn_src.clone());
        params.insert("attn_dst".to_string(), self.attn_dst.clone());
        if let Some(bias) = &self.bias {
            params.insert("bias".to_string(), bias.clone());
        }
        params
    }

    fn name(&self) -> &'static str {
        "GATConv"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcn_conv_shape() {
        let gcn = GCNConv::new(72, 128);
        let x = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * 7 * 72], &[2, 7, 72]).unwrap(),
            false,
        );
        let adj = Variable::new(
            Tensor::from_vec(vec![1.0; 7 * 7], &[7, 7]).unwrap(),
            false,
        );
        let output = gcn.forward_graph(&x, &adj);
        assert_eq!(output.shape(), vec![2, 7, 128]);
    }

    #[test]
    fn test_gcn_conv_identity_adjacency() {
        // With identity adjacency, each node only sees itself
        let gcn = GCNConv::new(4, 8);
        let x = Variable::new(
            Tensor::from_vec(vec![1.0; 1 * 3 * 4], &[1, 3, 4]).unwrap(),
            false,
        );

        // Identity adjacency
        let mut adj_data = vec![0.0; 9];
        adj_data[0] = 1.0; // (0,0)
        adj_data[4] = 1.0; // (1,1)
        adj_data[8] = 1.0; // (2,2)
        let adj = Variable::new(
            Tensor::from_vec(adj_data, &[3, 3]).unwrap(),
            false,
        );

        let output = gcn.forward_graph(&x, &adj);
        assert_eq!(output.shape(), vec![1, 3, 8]);

        // All nodes have same input, so all should produce same output
        let data = output.data().to_vec();
        for i in 0..3 {
            for f in 0..8 {
                assert!((data[i * 8 + f] - data[f]).abs() < 1e-6,
                    "Node outputs should be identical with identity adj and same input");
            }
        }
    }

    #[test]
    fn test_gcn_conv_parameters() {
        let gcn = GCNConv::new(16, 32);
        let params = gcn.parameters();
        assert_eq!(params.len(), 2); // weight + bias

        let total_params: usize = params.iter().map(|p| p.numel()).sum();
        assert_eq!(total_params, 16 * 32 + 32); // weight + bias
    }

    #[test]
    fn test_gcn_conv_no_bias() {
        let gcn = GCNConv::without_bias(16, 32);
        let params = gcn.parameters();
        assert_eq!(params.len(), 1); // weight only
    }

    #[test]
    fn test_gcn_conv_named_parameters() {
        let gcn = GCNConv::new(16, 32);
        let params = gcn.named_parameters();
        assert!(params.contains_key("weight"));
        assert!(params.contains_key("bias"));
    }

    #[test]
    fn test_gat_conv_shape() {
        let gat = GATConv::new(72, 32, 4); // 4 heads
        let x = Variable::new(
            Tensor::from_vec(vec![1.0; 2 * 7 * 72], &[2, 7, 72]).unwrap(),
            false,
        );
        let adj = Variable::new(
            Tensor::from_vec(vec![1.0; 7 * 7], &[7, 7]).unwrap(),
            false,
        );
        let output = gat.forward_graph(&x, &adj);
        assert_eq!(output.shape(), vec![2, 7, 128]); // 32 * 4 = 128
    }

    #[test]
    fn test_gat_conv_single_head() {
        let gat = GATConv::new(16, 8, 1);
        let x = Variable::new(
            Tensor::from_vec(vec![1.0; 1 * 5 * 16], &[1, 5, 16]).unwrap(),
            false,
        );
        let adj = Variable::new(
            Tensor::from_vec(vec![1.0; 5 * 5], &[5, 5]).unwrap(),
            false,
        );
        let output = gat.forward_graph(&x, &adj);
        assert_eq!(output.shape(), vec![1, 5, 8]);
    }

    #[test]
    fn test_gat_conv_parameters() {
        let gat = GATConv::new(16, 8, 4);
        let params = gat.parameters();
        assert_eq!(params.len(), 4); // w, attn_src, attn_dst, bias

        let named = gat.named_parameters();
        assert!(named.contains_key("w"));
        assert!(named.contains_key("attn_src"));
        assert!(named.contains_key("attn_dst"));
        assert!(named.contains_key("bias"));
    }

    #[test]
    fn test_gat_conv_total_output() {
        let gat = GATConv::new(16, 32, 4);
        assert_eq!(gat.total_out_features(), 128);
    }

    #[test]
    fn test_gcn_zero_adjacency() {
        // Zero adjacency should produce only bias in output
        let gcn = GCNConv::new(4, 4);
        let x = Variable::new(
            Tensor::from_vec(vec![99.0; 1 * 3 * 4], &[1, 3, 4]).unwrap(),
            false,
        );
        let adj = Variable::new(
            Tensor::from_vec(vec![0.0; 9], &[3, 3]).unwrap(),
            false,
        );
        let output = gcn.forward_graph(&x, &adj);

        // With zero adjacency, output should be just bias (all zeros initially)
        let data = output.data().to_vec();
        for val in &data {
            assert!(val.abs() < 1e-6, "Zero adjacency should zero out message passing");
        }
    }
}
