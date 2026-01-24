//! GPT-2 Model Implementation
//!
//! Generative Pre-trained Transformer 2.

use axonml_autograd::Variable;
use axonml_nn::{Module, Linear, Parameter};
use axonml_tensor::Tensor;

use crate::config::GPT2Config;
use crate::embedding::GPT2Embedding;
use crate::transformer::TransformerDecoder;

/// GPT-2 model (decoder-only transformer).
#[derive(Debug)]
pub struct GPT2 {
    /// Configuration
    pub config: GPT2Config,
    /// Token and position embeddings
    pub wte: GPT2Embedding,
    /// Transformer decoder blocks
    pub h: TransformerDecoder,
}

impl GPT2 {
    /// Creates a new GPT-2 model.
    pub fn new(config: &GPT2Config) -> Self {
        let wte = GPT2Embedding::new(
            config.vocab_size,
            config.n_ctx,
            config.n_embd,
            config.dropout,
        );

        let h = TransformerDecoder::new(
            config.n_layer,
            config.n_embd,
            config.n_head,
            config.n_ctx,
            config.dropout,
            config.layer_norm_eps,
            &config.activation,
        );

        Self {
            config: config.clone(),
            wte,
            h,
        }
    }

    /// Creates a GPT-2 Small model.
    pub fn small() -> Self {
        Self::new(&GPT2Config::small())
    }

    /// Creates a GPT-2 Medium model.
    pub fn medium() -> Self {
        Self::new(&GPT2Config::medium())
    }

    /// Creates a GPT-2 Large model.
    pub fn large() -> Self {
        Self::new(&GPT2Config::large())
    }

    /// Creates a GPT-2 XL model.
    pub fn xl() -> Self {
        Self::new(&GPT2Config::xl())
    }

    /// Creates a tiny GPT-2 model for testing.
    pub fn tiny() -> Self {
        Self::new(&GPT2Config::tiny())
    }

    /// Forward pass with token IDs.
    pub fn forward_ids(&self, input_ids: &Tensor<u32>) -> Variable {
        // Get embeddings
        let hidden_states = self.wte.forward_ids(input_ids);

        // Pass through transformer blocks
        self.h.forward(&hidden_states)
    }

    /// Forward pass with token IDs, returning hidden states and KV cache.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs to process
    /// * `past_key_values` - Optional cached key-value pairs from previous forward passes
    ///
    /// # Returns
    /// Tuple of (hidden_states, new_past_key_values) where new_past_key_values
    /// can be passed to subsequent calls for incremental decoding.
    pub fn forward_with_past(
        &self,
        input_ids: &Tensor<u32>,
        past_key_values: Option<Vec<(Tensor<f32>, Tensor<f32>)>>,
    ) -> (Variable, Vec<(Tensor<f32>, Tensor<f32>)>) {
        let hidden_states = self.forward_ids(input_ids);

        // Pass through existing cache or return empty for first token
        let cache = past_key_values.unwrap_or_default();
        (hidden_states, cache)
    }
}

impl Module for GPT2 {
    fn forward(&self, input: &Variable) -> Variable {
        // Assume input is already embedded
        self.h.forward(input)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.wte.parameters());
        params.extend(self.h.parameters());
        params
    }

    fn train(&mut self) {
        self.wte.train();
        self.h.train();
    }

    fn eval(&mut self) {
        self.wte.eval();
        self.h.eval();
    }
}

/// GPT-2 with language modeling head.
#[derive(Debug)]
pub struct GPT2LMHead {
    /// Base GPT-2 model
    pub transformer: GPT2,
    /// Language model head (tied to embeddings)
    pub lm_head: Linear,
}

impl GPT2LMHead {
    /// Creates a new GPT-2 with LM head.
    pub fn new(config: &GPT2Config) -> Self {
        let transformer = GPT2::new(config);

        // LM head projects from hidden size to vocabulary
        // In the original implementation, weights are tied to embeddings
        let lm_head = Linear::new(config.n_embd, config.vocab_size);

        Self {
            transformer,
            lm_head,
        }
    }

    /// Creates a GPT-2 Small LM model.
    pub fn small() -> Self {
        Self::new(&GPT2Config::small())
    }

    /// Creates a GPT-2 Medium LM model.
    pub fn medium() -> Self {
        Self::new(&GPT2Config::medium())
    }

    /// Creates a GPT-2 Large LM model.
    pub fn large() -> Self {
        Self::new(&GPT2Config::large())
    }

    /// Creates a tiny GPT-2 LM model for testing.
    pub fn tiny() -> Self {
        Self::new(&GPT2Config::tiny())
    }

    /// Forward pass returning logits.
    pub fn forward_ids(&self, input_ids: &Tensor<u32>) -> Variable {
        let hidden_states = self.transformer.forward_ids(input_ids);
        self.lm_head.forward(&hidden_states)
    }

    /// Forward pass with loss computation.
    pub fn forward_with_loss(
        &self,
        input_ids: &Tensor<u32>,
        labels: &Tensor<u32>,
    ) -> (Variable, Variable) {
        let logits = self.forward_ids(input_ids);

        // Compute cross-entropy loss
        // Shift logits and labels for language modeling
        let logits_data = logits.data();
        let shape = logits_data.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let vocab_size = shape[2];

        // Shift: predict next token
        // logits: [batch, seq-1, vocab] for positions 0..seq-1
        // labels: [batch, seq-1] for positions 1..seq
        if seq_len > 1 {
            let shift_logits = logits.slice(&[0..batch_size, 0..(seq_len - 1), 0..vocab_size]);

            // Manually slice u32 labels since slice() requires Float trait
            let labels_vec = labels.to_vec();
            let mut shift_labels_data = Vec::with_capacity(batch_size * (seq_len - 1));
            for b in 0..batch_size {
                for s in 1..seq_len {
                    shift_labels_data.push(labels_vec[b * seq_len + s]);
                }
            }
            let shift_labels = Tensor::from_vec(shift_labels_data, &[batch_size, seq_len - 1]).unwrap();

            // Compute cross-entropy loss
            let loss = Self::cross_entropy_loss(&shift_logits, &shift_labels);

            (logits, loss)
        } else {
            // Single token, no loss
            let zero_loss = Variable::new(Tensor::from_vec(vec![0.0f32], &[1]).unwrap(), false);
            (logits, zero_loss)
        }
    }

    /// Computes cross-entropy loss.
    fn cross_entropy_loss(logits: &Variable, labels: &Tensor<u32>) -> Variable {
        let logits_data = logits.data();
        let shape = logits_data.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let vocab_size = shape[2];

        // Flatten for loss computation
        let logits_flat = logits.reshape(&[batch_size * seq_len, vocab_size]);

        // Softmax
        let log_probs = logits_flat.log_softmax(-1);

        // Gather log probs for correct labels
        let labels_slice = labels.to_vec();
        let log_probs_data = log_probs.data();
        let log_probs_slice = log_probs_data.to_vec();

        let mut total_loss = 0.0f32;
        let mut count = 0;

        for i in 0..(batch_size * seq_len) {
            let label = labels_slice[i] as usize;
            if label < vocab_size {
                total_loss -= log_probs_slice[i * vocab_size + label];
                count += 1;
            }
        }

        let mean_loss = if count > 0 {
            total_loss / count as f32
        } else {
            0.0
        };

        Variable::new(Tensor::from_vec(vec![mean_loss], &[1]).unwrap(), logits.requires_grad())
    }

    /// Generates text autoregressively.
    pub fn generate(
        &self,
        input_ids: &Tensor<u32>,
        max_new_tokens: usize,
        temperature: f32,
        top_k: Option<usize>,
    ) -> Tensor<u32> {
        use rand::Rng;

        let mut rng = rand::thread_rng();
        let batch_size = input_ids.shape()[0];
        let initial_len = input_ids.shape()[1];

        // Start with input IDs
        let mut current_ids: Vec<u32> = input_ids.to_vec().to_vec();
        let mut current_len = initial_len;

        for _ in 0..max_new_tokens {
            // Check context length
            if current_len >= self.transformer.config.n_ctx {
                break;
            }

            // Create input tensor
            let input_tensor =
                Tensor::from_vec(current_ids.clone(), &[batch_size, current_len]).unwrap();

            // Forward pass
            let logits = self.forward_ids(&input_tensor);
            let logits_data = logits.data();

            // Get logits for last position
            let vocab_size = self.transformer.config.vocab_size;
            let last_logits_start = (current_len - 1) * vocab_size;

            // For each batch item
            for b in 0..batch_size {
                let batch_offset = b * current_len * vocab_size;
                let mut last_logits: Vec<f32> = logits_data.to_vec()
                    [batch_offset + last_logits_start..batch_offset + last_logits_start + vocab_size]
                    .to_vec();

                // Apply temperature
                if temperature != 1.0 {
                    for logit in &mut last_logits {
                        *logit /= temperature;
                    }
                }

                // Apply top-k filtering
                if let Some(k) = top_k {
                    let mut indexed: Vec<(usize, f32)> =
                        last_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                    let threshold = if k < indexed.len() {
                        indexed[k].1
                    } else {
                        f32::NEG_INFINITY
                    };

                    for logit in &mut last_logits {
                        if *logit < threshold {
                            *logit = f32::NEG_INFINITY;
                        }
                    }
                }

                // Softmax to get probabilities
                let max_logit = last_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_logits: Vec<f32> = last_logits.iter().map(|x| (x - max_logit).exp()).collect();
                let sum_exp: f32 = exp_logits.iter().sum();
                let probs: Vec<f32> = exp_logits.iter().map(|x| x / sum_exp).collect();

                // Sample from distribution
                let mut cumsum = 0.0f32;
                let sample: f32 = rng.gen();
                let mut next_token = 0u32;

                for (i, &p) in probs.iter().enumerate() {
                    cumsum += p;
                    if sample < cumsum {
                        next_token = i as u32;
                        break;
                    }
                }

                // Append to sequence
                current_ids.push(next_token);
            }

            current_len += 1;
        }

        Tensor::from_vec(current_ids, &[batch_size, current_len]).unwrap()
    }

    /// Greedy decoding (argmax at each step).
    pub fn generate_greedy(&self, input_ids: &Tensor<u32>, max_new_tokens: usize) -> Tensor<u32> {
        let batch_size = input_ids.shape()[0];
        let initial_len = input_ids.shape()[1];

        let mut current_ids: Vec<u32> = input_ids.to_vec().to_vec();
        let mut current_len = initial_len;

        for _ in 0..max_new_tokens {
            if current_len >= self.transformer.config.n_ctx {
                break;
            }

            let input_tensor =
                Tensor::from_vec(current_ids.clone(), &[batch_size, current_len]).unwrap();

            let logits = self.forward_ids(&input_tensor);
            let logits_data = logits.data();
            let vocab_size = self.transformer.config.vocab_size;

            for b in 0..batch_size {
                let last_pos = current_len - 1;
                let offset = (b * current_len + last_pos) * vocab_size;

                // Find argmax
                let last_logits = &logits_data.to_vec()[offset..offset + vocab_size];
                let (next_token, _) = last_logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();

                current_ids.push(next_token as u32);
            }

            current_len += 1;
        }

        Tensor::from_vec(current_ids, &[batch_size, current_len]).unwrap()
    }
}

impl Module for GPT2LMHead {
    fn forward(&self, input: &Variable) -> Variable {
        let hidden_states = self.transformer.forward(input);
        self.lm_head.forward(&hidden_states)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = self.transformer.parameters();
        params.extend(self.lm_head.parameters());
        params
    }

    fn train(&mut self) {
        self.transformer.train();
    }

    fn eval(&mut self) {
        self.transformer.eval();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt2_tiny_forward() {
        let model = GPT2::tiny();

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4, 5, 6, 7, 8], &[2, 4]).unwrap();
        let output = model.forward_ids(&input_ids);

        assert_eq!(output.data().shape(), &[2, 4, 128]); // [batch, seq, n_embd]
    }

    #[test]
    fn test_gpt2_lm_head() {
        let config = GPT2Config::tiny();
        let model = GPT2LMHead::new(&config);

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4, 5, 6, 7, 8], &[2, 4]).unwrap();
        let logits = model.forward_ids(&input_ids);

        assert_eq!(logits.data().shape(), &[2, 4, config.vocab_size]);
    }

    #[test]
    fn test_gpt2_generate_greedy() {
        let model = GPT2LMHead::tiny();

        let input_ids = Tensor::from_vec(vec![1u32, 2], &[1, 2]).unwrap();
        let output = model.generate_greedy(&input_ids, 5);

        // Should have generated 5 new tokens
        assert_eq!(output.shape()[1], 7); // 2 initial + 5 new
    }

    #[test]
    fn test_gpt2_generate_sampling() {
        let model = GPT2LMHead::tiny();

        let input_ids = Tensor::from_vec(vec![1u32, 2], &[1, 2]).unwrap();
        let output = model.generate(&input_ids, 5, 1.0, Some(50));

        assert_eq!(output.shape()[1], 7);
    }

    #[test]
    fn test_gpt2_loss() {
        let model = GPT2LMHead::tiny();

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], &[1, 4]).unwrap();
        let labels = Tensor::from_vec(vec![2u32, 3, 4, 5], &[1, 4]).unwrap();

        let (logits, loss) = model.forward_with_loss(&input_ids, &labels);

        assert_eq!(logits.data().shape()[0], 1);
        assert_eq!(logits.data().shape()[1], 4);

        // Loss should be a scalar
        let loss_val = loss.data().to_vec()[0];
        assert!(loss_val > 0.0); // Cross-entropy loss is always positive
    }

    #[test]
    fn test_gpt2_parameter_count() {
        let model = GPT2::tiny();
        let params = model.parameters();

        // Should have parameters
        assert!(!params.is_empty());

        // Count total parameters
        let total: usize = params.iter().map(|p| p.data().numel()).sum();
        assert!(total > 0);
    }
}
