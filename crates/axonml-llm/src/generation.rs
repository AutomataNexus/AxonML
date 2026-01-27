//! Text Generation Utilities
//!
//! Sampling strategies and generation configuration for language models.

use axonml_tensor::Tensor;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Configuration for text generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate
    pub max_new_tokens: usize,
    /// Temperature for sampling (1.0 = no change, <1.0 = more deterministic, >1.0 = more random)
    pub temperature: f32,
    /// Top-k sampling: only sample from top k tokens
    pub top_k: Option<usize>,
    /// Top-p (nucleus) sampling: sample from tokens with cumulative probability >= p
    pub top_p: Option<f32>,
    /// Repetition penalty (1.0 = no penalty, >1.0 = penalize repetition)
    pub repetition_penalty: f32,
    /// Stop token IDs
    pub eos_token_ids: Vec<u32>,
    /// Pad token ID
    pub pad_token_id: Option<u32>,
    /// Whether to do greedy decoding
    pub do_sample: bool,
    /// Number of beams for beam search (1 = no beam search)
    pub num_beams: usize,
    /// Length penalty for beam search
    pub length_penalty: f32,
    /// Early stopping for beam search
    pub early_stopping: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 50,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
            eos_token_ids: vec![],
            pad_token_id: None,
            do_sample: true,
            num_beams: 1,
            length_penalty: 1.0,
            early_stopping: false,
        }
    }
}

impl GenerationConfig {
    /// Creates a config for greedy decoding.
    pub fn greedy() -> Self {
        Self {
            do_sample: false,
            temperature: 1.0,
            top_k: None,
            top_p: None,
            ..Default::default()
        }
    }

    /// Creates a config for sampling with temperature.
    pub fn sampling(temperature: f32) -> Self {
        Self {
            do_sample: true,
            temperature,
            ..Default::default()
        }
    }

    /// Creates a config for top-k sampling.
    pub fn top_k_sampling(k: usize, temperature: f32) -> Self {
        Self {
            do_sample: true,
            temperature,
            top_k: Some(k),
            ..Default::default()
        }
    }

    /// Creates a config for nucleus (top-p) sampling.
    pub fn nucleus_sampling(p: f32, temperature: f32) -> Self {
        Self {
            do_sample: true,
            temperature,
            top_p: Some(p),
            ..Default::default()
        }
    }

    /// Creates a config for beam search.
    pub fn beam_search(num_beams: usize) -> Self {
        Self {
            do_sample: false,
            num_beams,
            ..Default::default()
        }
    }

    /// Sets the maximum number of new tokens.
    pub fn with_max_tokens(mut self, max_new_tokens: usize) -> Self {
        self.max_new_tokens = max_new_tokens;
        self
    }

    /// Sets the EOS token ID.
    pub fn with_eos_token(mut self, eos_token_id: u32) -> Self {
        self.eos_token_ids.push(eos_token_id);
        self
    }

    /// Sets the repetition penalty.
    pub fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = penalty;
        self
    }
}

/// Text generator for language models.
pub struct TextGenerator {
    /// Generation configuration
    pub config: GenerationConfig,
}

impl TextGenerator {
    /// Creates a new text generator.
    pub fn new(config: GenerationConfig) -> Self {
        Self { config }
    }

    /// Applies temperature scaling to logits.
    pub fn apply_temperature(&self, logits: &mut [f32]) {
        if self.config.temperature != 1.0 {
            for logit in logits.iter_mut() {
                *logit /= self.config.temperature;
            }
        }
    }

    /// Applies repetition penalty to logits.
    pub fn apply_repetition_penalty(&self, logits: &mut [f32], generated_tokens: &[u32]) {
        if self.config.repetition_penalty != 1.0 {
            for &token in generated_tokens {
                let idx = token as usize;
                if idx < logits.len() {
                    if logits[idx] > 0.0 {
                        logits[idx] /= self.config.repetition_penalty;
                    } else {
                        logits[idx] *= self.config.repetition_penalty;
                    }
                }
            }
        }
    }

    /// Applies top-k filtering to logits.
    pub fn apply_top_k(&self, logits: &mut [f32]) {
        if let Some(k) = self.config.top_k {
            if k < logits.len() {
                // Find indices of top k values
                let mut sorted_indices: Vec<usize> = (0..logits.len()).collect();
                sorted_indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());

                // Create a set of top-k indices
                let top_k_indices: std::collections::HashSet<usize> =
                    sorted_indices[..k].iter().copied().collect();

                // Set all values not in top-k to -inf
                for (i, logit) in logits.iter_mut().enumerate() {
                    if !top_k_indices.contains(&i) {
                        *logit = f32::NEG_INFINITY;
                    }
                }
            }
        }
    }

    /// Applies top-p (nucleus) filtering to logits.
    pub fn apply_top_p(&self, logits: &mut [f32]) {
        if let Some(p) = self.config.top_p {
            // Convert to probabilities
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_logits: Vec<f32> = logits.iter().map(|x| (x - max_logit).exp()).collect();
            let sum_exp: f32 = exp_logits.iter().sum();
            let probs: Vec<f32> = exp_logits.iter().map(|x| x / sum_exp).collect();

            // Sort by probability
            let mut sorted_indices: Vec<usize> = (0..probs.len()).collect();
            sorted_indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

            // Find cutoff
            let mut cumsum = 0.0f32;
            let mut cutoff_idx = sorted_indices.len();

            for (i, &idx) in sorted_indices.iter().enumerate() {
                cumsum += probs[idx];
                if cumsum > p {
                    cutoff_idx = i + 1;
                    break;
                }
            }

            // Set values outside nucleus to -inf
            for (i, logit) in logits.iter_mut().enumerate() {
                if !sorted_indices[..cutoff_idx].contains(&i) {
                    *logit = f32::NEG_INFINITY;
                }
            }
        }
    }

    /// Samples from logits distribution.
    pub fn sample(&self, logits: &[f32]) -> u32 {
        if !self.config.do_sample {
            // Greedy: return argmax
            return self.argmax(logits);
        }

        // Sample from distribution
        let mut rng = rand::thread_rng();

        // Softmax
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = logits.iter().map(|x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits.iter().map(|x| x / sum_exp).collect();

        // Sample
        let mut cumsum = 0.0f32;
        let sample: f32 = rng.gen();

        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if sample < cumsum {
                return i as u32;
            }
        }

        // Fallback to last token
        (logits.len() - 1) as u32
    }

    /// Returns the index of the maximum value.
    pub fn argmax(&self, logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }

    /// Processes logits and returns next token.
    pub fn get_next_token(&self, logits: &[f32], generated_tokens: &[u32]) -> u32 {
        let mut logits = logits.to_vec();

        // Apply modifiers
        self.apply_repetition_penalty(&mut logits, generated_tokens);
        self.apply_temperature(&mut logits);
        self.apply_top_k(&mut logits);
        self.apply_top_p(&mut logits);

        // Sample
        self.sample(&logits)
    }

    /// Checks if generation should stop.
    pub fn should_stop(&self, token: u32) -> bool {
        self.config.eos_token_ids.contains(&token)
    }
}

/// Beam for beam search.
#[derive(Debug, Clone)]
pub struct Beam {
    /// Token sequence
    pub tokens: Vec<u32>,
    /// Log probability score
    pub score: f32,
    /// Whether this beam has finished
    pub finished: bool,
}

impl Beam {
    /// Creates a new beam.
    pub fn new(initial_tokens: Vec<u32>) -> Self {
        Self {
            tokens: initial_tokens,
            score: 0.0,
            finished: false,
        }
    }

    /// Returns the normalized score (for length penalty).
    pub fn normalized_score(&self, length_penalty: f32) -> f32 {
        let length = self.tokens.len() as f32;
        self.score / length.powf(length_penalty)
    }
}

/// Beam search implementation.
pub struct BeamSearch {
    /// Number of beams
    pub num_beams: usize,
    /// Length penalty
    pub length_penalty: f32,
    /// Early stopping
    pub early_stopping: bool,
    /// EOS token IDs
    pub eos_token_ids: Vec<u32>,
}

impl BeamSearch {
    /// Creates a new beam search.
    pub fn new(
        num_beams: usize,
        length_penalty: f32,
        early_stopping: bool,
        eos_token_ids: Vec<u32>,
    ) -> Self {
        Self {
            num_beams,
            length_penalty,
            early_stopping,
            eos_token_ids,
        }
    }

    /// Initializes beams from input tokens.
    pub fn init_beams(&self, input_ids: &Tensor<u32>) -> Vec<Beam> {
        let tokens: Vec<u32> = input_ids.to_vec().to_vec();
        vec![Beam::new(tokens)]
    }

    /// Expands beams with new tokens and scores.
    pub fn expand_beams(&self, beams: &[Beam], next_token_logits: &[Vec<f32>]) -> Vec<Beam> {
        let mut candidates = Vec::new();

        for (beam_idx, beam) in beams.iter().enumerate() {
            if beam.finished {
                candidates.push(beam.clone());
                continue;
            }

            let logits = &next_token_logits[beam_idx];

            // Get top-k tokens for this beam
            let mut indexed: Vec<(usize, f32)> =
                logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            for (token, log_prob) in indexed.into_iter().take(self.num_beams * 2) {
                let mut new_beam = beam.clone();
                new_beam.tokens.push(token as u32);
                new_beam.score += log_prob;

                if self.eos_token_ids.contains(&(token as u32)) {
                    new_beam.finished = true;
                }

                candidates.push(new_beam);
            }
        }

        // Sort by score and keep top beams
        candidates.sort_by(|a, b| {
            b.normalized_score(self.length_penalty)
                .partial_cmp(&a.normalized_score(self.length_penalty))
                .unwrap()
        });

        candidates.into_iter().take(self.num_beams).collect()
    }

    /// Checks if search should stop.
    pub fn should_stop(&self, beams: &[Beam]) -> bool {
        if self.early_stopping {
            beams.iter().all(|b| b.finished)
        } else {
            false
        }
    }

    /// Returns the best completed sequence.
    pub fn best_sequence(&self, beams: &[Beam]) -> Option<Vec<u32>> {
        beams
            .iter()
            .filter(|b| b.finished)
            .max_by(|a, b| {
                a.normalized_score(self.length_penalty)
                    .partial_cmp(&b.normalized_score(self.length_penalty))
                    .unwrap()
            })
            .map(|b| b.tokens.clone())
            .or_else(|| beams.first().map(|b| b.tokens.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_config_defaults() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_new_tokens, 50);
        assert_eq!(config.temperature, 1.0);
        assert!(config.do_sample);
    }

    #[test]
    fn test_greedy_config() {
        let config = GenerationConfig::greedy();
        assert!(!config.do_sample);
    }

    #[test]
    fn test_top_k_filtering() {
        let config = GenerationConfig::top_k_sampling(2, 1.0);
        let generator = TextGenerator::new(config);

        let mut logits = vec![1.0, 5.0, 3.0, 4.0, 2.0];
        generator.apply_top_k(&mut logits);

        // Only top 2 should remain finite
        let finite_count = logits.iter().filter(|x| x.is_finite()).count();
        assert_eq!(finite_count, 2);
    }

    #[test]
    fn test_temperature_scaling() {
        let config = GenerationConfig::sampling(2.0);
        let generator = TextGenerator::new(config);

        let mut logits = vec![2.0, 4.0, 6.0];
        generator.apply_temperature(&mut logits);

        assert_eq!(logits, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_argmax() {
        let config = GenerationConfig::greedy();
        let generator = TextGenerator::new(config);

        let logits = vec![1.0, 5.0, 3.0, 4.0, 2.0];
        let result = generator.argmax(&logits);

        assert_eq!(result, 1);
    }

    #[test]
    fn test_repetition_penalty() {
        let config = GenerationConfig::default().with_repetition_penalty(2.0);
        let generator = TextGenerator::new(config);

        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        let generated = vec![1, 3];
        generator.apply_repetition_penalty(&mut logits, &generated);

        // Tokens 1 and 3 should be penalized
        assert!(logits[1] < 2.0);
        assert!(logits[3] < 4.0);
    }

    #[test]
    fn test_beam_search_init() {
        let beam_search = BeamSearch::new(3, 1.0, false, vec![0]);
        let input_ids = Tensor::from_vec(vec![1u32, 2, 3], &[1, 3]).unwrap();
        let beams = beam_search.init_beams(&input_ids);

        assert_eq!(beams.len(), 1);
        assert_eq!(beams[0].tokens, vec![1, 2, 3]);
    }
}
