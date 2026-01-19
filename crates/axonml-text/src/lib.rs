//! Axonml Text - Natural Language Processing Utilities
//!
//! This crate provides text processing functionality for the Axonml ML framework:
//!
//! - **Vocabulary**: Token-to-index mapping with special tokens
//! - **Tokenizers**: Various tokenization strategies (whitespace, char, BPE)
//! - **Datasets**: Text classification, language modeling, seq2seq
//!
//! # Example
//!
//! ```ignore
//! use axonml_text::prelude::*;
//!
//! // Build vocabulary from text
//! let vocab = Vocab::from_text("the quick brown fox", 1);
//!
//! // Tokenize text
//! let tokenizer = WhitespaceTokenizer::new();
//! let tokens = tokenizer.tokenize("hello world");
//!
//! // Create a sentiment dataset
//! let dataset = SyntheticSentimentDataset::small();
//! ```
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
// ML/tensor-specific allowances
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::ptr_arg)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::unused_self)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::single_match_else)]
#![allow(clippy::fn_params_excessive_bools)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::format_push_string)]
#![allow(clippy::erasing_op)]
#![allow(clippy::type_repetition_in_bounds)]
#![allow(clippy::iter_without_into_iter)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::use_debug)]
#![allow(clippy::case_sensitive_file_extension_comparisons)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::panic)]
#![allow(clippy::struct_field_names)]
#![allow(clippy::missing_fields_in_debug)]
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::assigning_clones)]
#![allow(clippy::option_if_let_else)]
#![allow(clippy::manual_let_else)]
#![allow(clippy::explicit_iter_loop)]
#![allow(clippy::default_trait_access)]
#![allow(clippy::only_used_in_recursion)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::ref_option)]
#![allow(clippy::multiple_bound_locations)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::manual_assert)]
#![allow(clippy::unnecessary_debug_formatting)]

pub mod datasets;
pub mod tokenizer;
pub mod vocab;

// =============================================================================
// Re-exports
// =============================================================================

pub use vocab::{Vocab, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN, PAD_TOKEN, UNK_TOKEN};

pub use tokenizer::{
    BasicBPETokenizer, CharTokenizer, NGramTokenizer, Tokenizer, UnigramTokenizer,
    WhitespaceTokenizer, WordPunctTokenizer,
};

pub use datasets::{
    LanguageModelDataset, SyntheticSentimentDataset, SyntheticSeq2SeqDataset, TextDataset,
};

// =============================================================================
// Prelude
// =============================================================================

/// Common imports for text processing.
pub mod prelude {
    pub use crate::{
        BasicBPETokenizer,
        CharTokenizer,
        LanguageModelDataset,
        NGramTokenizer,
        SyntheticSentimentDataset,
        SyntheticSeq2SeqDataset,
        // Datasets
        TextDataset,
        // Tokenizers
        Tokenizer,
        UnigramTokenizer,
        // Vocabulary
        Vocab,
        WhitespaceTokenizer,
        WordPunctTokenizer,
        BOS_TOKEN,
        EOS_TOKEN,
        MASK_TOKEN,
        PAD_TOKEN,
        UNK_TOKEN,
    };

    pub use axonml_data::{DataLoader, Dataset};
    pub use axonml_tensor::Tensor;
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axonml_data::Dataset;

    #[test]
    fn test_vocab_and_tokenizer_integration() {
        let text = "the quick brown fox jumps over the lazy dog";
        let vocab = Vocab::from_text(text, 1);
        let tokenizer = WhitespaceTokenizer::new();

        let tokens = tokenizer.tokenize("the fox");
        let indices = tokenizer.encode("the fox", &vocab);

        assert_eq!(tokens.len(), 2);
        assert_eq!(indices.len(), 2);
    }

    #[test]
    fn test_text_dataset_with_tokenizer() {
        let samples = vec![
            ("good movie".to_string(), 1),
            ("bad movie".to_string(), 0),
            ("great film".to_string(), 1),
            ("terrible movie".to_string(), 0),
        ];

        let tokenizer = WhitespaceTokenizer::new();
        let dataset = TextDataset::from_samples(&samples, &tokenizer, 1, 10);

        assert_eq!(dataset.len(), 4);
        assert_eq!(dataset.num_classes(), 2);
    }

    #[test]
    fn test_language_model_pipeline() {
        let text = "one two three four five six seven eight nine ten";
        let dataset = LanguageModelDataset::from_text(text, 3, 1);

        assert!(dataset.len() > 0);

        // Get first sample
        let (input, target) = dataset.get(0).unwrap();
        assert_eq!(input.shape(), &[3]);
        assert_eq!(target.shape(), &[3]);
    }

    #[test]
    fn test_bpe_tokenizer_training() {
        let mut tokenizer = BasicBPETokenizer::new();
        let text = "low lower lowest newer newest";
        tokenizer.train(text, 10);

        let vocab = tokenizer.get_vocab();
        assert!(!vocab.is_empty());

        let tokens = tokenizer.tokenize("low");
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_char_tokenizer_with_vocab() {
        let tokenizer = CharTokenizer::new();
        let mut vocab = Vocab::with_special_tokens();

        // Add characters to vocabulary
        for c in "abcdefghijklmnopqrstuvwxyz ".chars() {
            vocab.add_token(&c.to_string());
        }

        let indices = tokenizer.encode("hello", &vocab);
        assert_eq!(indices.len(), 5);
    }

    #[test]
    fn test_synthetic_datasets_with_dataloader() {
        use axonml_data::DataLoader;

        let dataset = SyntheticSentimentDataset::small();
        let loader = DataLoader::new(dataset, 16);

        let mut batch_count = 0;
        for batch in loader.iter().take(3) {
            assert_eq!(batch.data.shape()[0], 16);
            batch_count += 1;
        }
        assert_eq!(batch_count, 3);
    }

    #[test]
    fn test_ngram_tokenizer() {
        let word_bigrams = NGramTokenizer::word_ngrams(2);
        let tokens = word_bigrams.tokenize("one two three four");

        assert_eq!(tokens.len(), 3);
        assert!(tokens.contains(&"one two".to_string()));

        let char_trigrams = NGramTokenizer::char_ngrams(3);
        let tokens = char_trigrams.tokenize("hello");

        assert_eq!(tokens.len(), 3);
    }

    #[test]
    fn test_seq2seq_reverse_task() {
        let dataset = SyntheticSeq2SeqDataset::copy_task(10, 5, 100);

        let (src, tgt) = dataset.get(0).unwrap();

        // Verify target is reversed source
        let src_vec = src.to_vec();
        let tgt_vec = tgt.to_vec();

        for (i, &val) in src_vec.iter().enumerate() {
            assert_eq!(val, tgt_vec[src_vec.len() - 1 - i]);
        }
    }
}
