//! BERT Model Implementation
//!
//! Bidirectional Encoder Representations from Transformers.

use axonml_autograd::Variable;
use axonml_nn::{Module, Linear, Dropout, Parameter};
use axonml_tensor::Tensor;

use crate::config::BertConfig;
use crate::embedding::BertEmbedding;
use crate::error::{LLMError, LLMResult};
use crate::transformer::{TransformerEncoder, LayerNorm};

/// BERT model (encoder-only transformer).
#[derive(Debug)]
pub struct Bert {
    /// Configuration
    pub config: BertConfig,
    /// Embeddings
    pub embeddings: BertEmbedding,
    /// Transformer encoder
    pub encoder: TransformerEncoder,
    /// Pooler (optional CLS token transformation)
    pub pooler: Option<BertPooler>,
}

/// BERT pooler for sequence classification.
#[derive(Debug)]
pub struct BertPooler {
    /// Dense layer
    pub dense: Linear,
}

impl BertPooler {
    /// Creates a new BERT pooler.
    pub fn new(hidden_size: usize) -> Self {
        Self {
            dense: Linear::new(hidden_size, hidden_size),
        }
    }
}

impl Module for BertPooler {
    fn forward(&self, input: &Variable) -> Variable {
        // Take the first token ([CLS]) representation
        let input_data = input.data();
        let shape = input_data.shape();
        let batch_size = shape[0];
        let hidden_size = shape[2];

        // Extract [CLS] token: input[:, 0, :]
        let cls_output = input.slice(&[0..batch_size, 0..1, 0..hidden_size]);
        let cls_output = cls_output.reshape(&[batch_size, hidden_size]);

        // Apply dense + tanh
        let pooled = self.dense.forward(&cls_output);
        pooled.tanh()
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.dense.parameters()
    }
}

impl Bert {
    /// Creates a new BERT model.
    pub fn new(config: &BertConfig) -> Self {
        Self::with_pooler(config, true)
    }

    /// Creates a new BERT model with optional pooler.
    pub fn with_pooler(config: &BertConfig, add_pooler: bool) -> Self {
        let embeddings = BertEmbedding::new(
            config.vocab_size,
            config.max_position_embeddings,
            config.type_vocab_size,
            config.hidden_size,
            config.layer_norm_eps,
            config.hidden_dropout_prob,
        );

        let encoder = TransformerEncoder::new(
            config.num_hidden_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.intermediate_size,
            config.hidden_dropout_prob,
            config.layer_norm_eps,
            &config.hidden_act,
            false, // post-norm for BERT
        );

        let pooler = if add_pooler {
            Some(BertPooler::new(config.hidden_size))
        } else {
            None
        };

        Self {
            config: config.clone(),
            embeddings,
            encoder,
            pooler,
        }
    }

    /// Forward pass returning both sequence output and pooled output.
    pub fn forward_with_pooling(
        &self,
        input_ids: &Tensor<u32>,
        token_type_ids: Option<&Tensor<u32>>,
        attention_mask: Option<&Tensor<f32>>,
    ) -> (Variable, Option<Variable>) {
        // Get embeddings
        let hidden_states = self.embeddings.forward_with_ids(input_ids, token_type_ids, None);

        // Encode
        let sequence_output = self.encoder.forward_with_mask(&hidden_states, attention_mask);

        // Pool if pooler exists
        let pooled_output = self.pooler.as_ref().map(|p| p.forward(&sequence_output));

        (sequence_output, pooled_output)
    }

    /// Forward pass returning sequence output.
    pub fn forward_ids(&self, input_ids: &Tensor<u32>) -> Variable {
        let (sequence_output, _) = self.forward_with_pooling(input_ids, None, None);
        sequence_output
    }
}

impl Module for Bert {
    fn forward(&self, input: &Variable) -> Variable {
        // Assume input is embeddings if using Module trait
        self.encoder.forward(input)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.embeddings.parameters());
        params.extend(self.encoder.parameters());
        if let Some(ref pooler) = self.pooler {
            params.extend(pooler.parameters());
        }
        params
    }

    fn train(&mut self) {
        self.embeddings.train();
        self.encoder.train();
    }

    fn eval(&mut self) {
        self.embeddings.eval();
        self.encoder.eval();
    }
}

/// BERT for sequence classification.
#[derive(Debug)]
pub struct BertForSequenceClassification {
    /// Base BERT model
    pub bert: Bert,
    /// Dropout
    pub dropout: Dropout,
    /// Classification head
    pub classifier: Linear,
    /// Number of labels
    pub num_labels: usize,
}

impl BertForSequenceClassification {
    /// Creates a new BERT for sequence classification.
    pub fn new(config: &BertConfig, num_labels: usize) -> Self {
        Self {
            bert: Bert::new(config),
            dropout: Dropout::new(config.hidden_dropout_prob),
            classifier: Linear::new(config.hidden_size, num_labels),
            num_labels,
        }
    }

    /// Forward pass for classification.
    ///
    /// # Errors
    /// Returns an error if the BERT model does not have a pooler configured.
    pub fn forward_classification(&self, input_ids: &Tensor<u32>) -> LLMResult<Variable> {
        let (_, pooled_output) = self.bert.forward_with_pooling(input_ids, None, None);

        if let Some(pooled) = pooled_output {
            let pooled = self.dropout.forward(&pooled);
            Ok(self.classifier.forward(&pooled))
        } else {
            Err(LLMError::InvalidConfig(
                "BERT model must have pooler for sequence classification".to_string()
            ))
        }
    }
}

impl Module for BertForSequenceClassification {
    fn forward(&self, input: &Variable) -> Variable {
        let sequence_output = self.bert.forward(input);

        // Get [CLS] token
        let seq_data = sequence_output.data();
        let shape = seq_data.shape();
        let batch_size = shape[0];
        let hidden_size = shape[2];

        let cls_output = sequence_output.slice(&[0..batch_size, 0..1, 0..hidden_size]);
        let cls_output = cls_output.reshape(&[batch_size, hidden_size]);

        let cls_output = self.dropout.forward(&cls_output);
        self.classifier.forward(&cls_output)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = self.bert.parameters();
        params.extend(self.classifier.parameters());
        params
    }

    fn train(&mut self) {
        self.bert.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.bert.eval();
        self.dropout.eval();
    }
}

/// BERT for masked language modeling.
#[derive(Debug)]
pub struct BertForMaskedLM {
    /// Base BERT model
    pub bert: Bert,
    /// MLM head
    pub cls: BertLMPredictionHead,
}

/// BERT LM prediction head.
#[derive(Debug)]
pub struct BertLMPredictionHead {
    /// Transform layer
    pub transform: BertPredictionHeadTransform,
    /// Output projection (tied to embeddings in full implementation)
    pub decoder: Linear,
}

/// Transform layer for BERT prediction head.
#[derive(Debug)]
pub struct BertPredictionHeadTransform {
    /// Dense layer
    pub dense: Linear,
    /// Layer norm
    pub layer_norm: LayerNorm,
    /// Activation
    pub activation: String,
}

impl BertPredictionHeadTransform {
    /// Creates a new prediction head transform.
    pub fn new(hidden_size: usize, layer_norm_eps: f32, activation: &str) -> Self {
        Self {
            dense: Linear::new(hidden_size, hidden_size),
            layer_norm: LayerNorm::new(hidden_size, layer_norm_eps),
            activation: activation.to_string(),
        }
    }
}

impl Module for BertPredictionHeadTransform {
    fn forward(&self, input: &Variable) -> Variable {
        let x = self.dense.forward(input);
        let x = match self.activation.as_str() {
            "gelu" => x.gelu(),
            "relu" => x.relu(),
            _ => x.gelu(),
        };
        self.layer_norm.forward(&x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = self.dense.parameters();
        params.extend(self.layer_norm.parameters());
        params
    }
}

impl BertLMPredictionHead {
    /// Creates a new LM prediction head.
    pub fn new(hidden_size: usize, vocab_size: usize, layer_norm_eps: f32, activation: &str) -> Self {
        Self {
            transform: BertPredictionHeadTransform::new(hidden_size, layer_norm_eps, activation),
            decoder: Linear::new(hidden_size, vocab_size),
        }
    }
}

impl Module for BertLMPredictionHead {
    fn forward(&self, input: &Variable) -> Variable {
        let x = self.transform.forward(input);
        self.decoder.forward(&x)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = self.transform.parameters();
        params.extend(self.decoder.parameters());
        params
    }
}

impl BertForMaskedLM {
    /// Creates a new BERT for masked language modeling.
    pub fn new(config: &BertConfig) -> Self {
        let bert = Bert::with_pooler(config, false); // No pooler needed for MLM
        let cls = BertLMPredictionHead::new(
            config.hidden_size,
            config.vocab_size,
            config.layer_norm_eps,
            &config.hidden_act,
        );

        Self { bert, cls }
    }

    /// Forward pass for MLM.
    pub fn forward_mlm(&self, input_ids: &Tensor<u32>) -> Variable {
        let sequence_output = self.bert.forward_ids(input_ids);
        self.cls.forward(&sequence_output)
    }
}

impl Module for BertForMaskedLM {
    fn forward(&self, input: &Variable) -> Variable {
        let sequence_output = self.bert.forward(input);
        self.cls.forward(&sequence_output)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = self.bert.parameters();
        params.extend(self.cls.parameters());
        params
    }

    fn train(&mut self) {
        self.bert.train();
    }

    fn eval(&mut self) {
        self.bert.eval();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bert_tiny() {
        let config = BertConfig::tiny();
        let model = Bert::new(&config);

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4, 5, 6, 7, 8], &[2, 4]).unwrap();
        let output = model.forward_ids(&input_ids);

        assert_eq!(output.data().shape(), &[2, 4, config.hidden_size]);
    }

    #[test]
    fn test_bert_pooler() {
        let config = BertConfig::tiny();
        let model = Bert::new(&config);

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4, 5, 6, 7, 8], &[2, 4]).unwrap();
        let (seq_out, pooled_out) = model.forward_with_pooling(&input_ids, None, None);

        assert_eq!(seq_out.data().shape(), &[2, 4, config.hidden_size]);
        assert!(pooled_out.is_some());
        assert_eq!(pooled_out.unwrap().data().shape(), &[2, config.hidden_size]);
    }

    #[test]
    fn test_bert_for_classification() {
        let config = BertConfig::tiny();
        let model = BertForSequenceClassification::new(&config, 2);

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4, 5, 6, 7, 8], &[2, 4]).unwrap();
        let logits = model.forward_classification(&input_ids).unwrap();

        assert_eq!(logits.data().shape(), &[2, 2]); // [batch, num_labels]
    }

    #[test]
    fn test_bert_for_mlm() {
        let config = BertConfig::tiny();
        let model = BertForMaskedLM::new(&config);

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4, 5, 6, 7, 8], &[2, 4]).unwrap();
        let logits = model.forward_mlm(&input_ids);

        assert_eq!(logits.data().shape(), &[2, 4, config.vocab_size]);
    }

    #[test]
    fn test_bert_parameter_count() {
        let config = BertConfig::tiny();
        let model = Bert::new(&config);
        let params = model.parameters();

        // Should have many parameters
        assert!(!params.is_empty());
    }
}
