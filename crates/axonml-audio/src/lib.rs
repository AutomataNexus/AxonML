//! Audio processing for AxonML.
//!
//! Transforms (MelSpectrogram via rustfft O(n log n), MFCC, Resample,
//! NormalizeAudio, AddNoise, TimeStretch, PitchShift, TrimSilence) and
//! synthetic datasets (SyntheticCommandDataset, SyntheticMusicDataset,
//! SyntheticSpeakerDataset) for command recognition, music genre
//! classification, and speaker identification tasks.
//!
//! # File
//! `crates/axonml-audio/src/lib.rs`
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
pub mod transforms;

// =============================================================================
// Re-exports
// =============================================================================

pub use transforms::{
    AddNoise, MFCC, MelSpectrogram, NormalizeAudio, PitchShift, Resample, TimeStretch, TrimSilence,
};

pub use datasets::{
    AudioClassificationDataset, AudioSeq2SeqDataset, SyntheticCommandDataset,
    SyntheticMusicDataset, SyntheticSpeakerDataset,
};

// =============================================================================
// Prelude
// =============================================================================

/// Common imports for audio processing.
pub mod prelude {
    pub use crate::{
        AddNoise,
        // Datasets
        AudioClassificationDataset,
        AudioSeq2SeqDataset,
        MFCC,
        MelSpectrogram,
        NormalizeAudio,
        PitchShift,
        // Transforms
        Resample,
        SyntheticCommandDataset,
        SyntheticMusicDataset,
        SyntheticSpeakerDataset,
        TimeStretch,
        TrimSilence,
    };

    pub use axonml_data::{DataLoader, Dataset, Transform};
    pub use axonml_tensor::Tensor;
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axonml_data::{Dataset, Transform};

    #[test]
    fn test_transform_on_dataset() {
        let dataset = SyntheticCommandDataset::small();
        let mel = MelSpectrogram::with_params(16000, 512, 256, 40);

        let (waveform, _label) = dataset.get(0).unwrap();
        let spectrogram = mel.apply(&waveform);

        assert_eq!(spectrogram.shape()[0], 40);
        assert!(spectrogram.shape()[1] > 0);
    }

    #[test]
    fn test_mfcc_on_dataset() {
        let dataset = SyntheticCommandDataset::small();
        let mfcc = MFCC::new(16000, 13);

        let (waveform, _) = dataset.get(0).unwrap();
        let coeffs = mfcc.apply(&waveform);

        assert_eq!(coeffs.shape()[0], 13);
    }

    #[test]
    fn test_resample_on_dataset() {
        let dataset = SyntheticCommandDataset::new(10, 22050, 0.5, 5);
        let resample = Resample::new(22050, 16000);

        let (waveform, _) = dataset.get(0).unwrap();
        let resampled = resample.apply(&waveform);

        // 0.5s at 22050Hz = 11025 samples
        // 0.5s at 16000Hz = 8000 samples
        assert_eq!(waveform.shape()[0], 11025);
        assert_eq!(resampled.shape()[0], 8000);
    }

    #[test]
    fn test_noise_augmentation() {
        let dataset = SyntheticMusicDataset::small();
        let add_noise = AddNoise::new(30.0); // High SNR for minimal noise

        let (waveform, _) = dataset.get(0).unwrap();
        let noisy = add_noise.apply(&waveform);

        assert_eq!(noisy.shape(), waveform.shape());
    }

    #[test]
    fn test_normalize_audio() {
        let dataset = SyntheticSpeakerDataset::small();
        let normalize = NormalizeAudio::new();

        let (waveform, _) = dataset.get(0).unwrap();
        let normalized = normalize.apply(&waveform);

        let max_val = normalized
            .to_vec()
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);
        assert!((max_val - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_pipeline() {
        let dataset = SyntheticCommandDataset::small();

        // Create a processing pipeline
        let resample = Resample::new(16000, 8000);
        let normalize = NormalizeAudio::new();
        let mel = MelSpectrogram::with_params(8000, 256, 128, 40);

        let (waveform, _) = dataset.get(0).unwrap();

        // Apply pipeline
        let resampled = resample.apply(&waveform);
        let normalized = normalize.apply(&resampled);
        let spectrogram = mel.apply(&normalized);

        assert_eq!(spectrogram.shape()[0], 40);
    }

    #[test]
    fn test_time_stretch_preserves_audio_characteristics() {
        let dataset = SyntheticMusicDataset::small();
        let stretch = TimeStretch::new(1.0); // No change

        let (waveform, _) = dataset.get(0).unwrap();
        let stretched = stretch.apply(&waveform);

        // Should be approximately the same length
        assert_eq!(stretched.shape()[0], waveform.shape()[0]);
    }

    #[test]
    fn test_pitch_shift() {
        let dataset = SyntheticCommandDataset::small();
        let shift = PitchShift::new(0.0); // No shift

        let (waveform, _) = dataset.get(0).unwrap();
        let shifted = shift.apply(&waveform);

        assert_eq!(shifted.shape()[0], waveform.shape()[0]);
    }

    #[test]
    fn test_dataset_with_dataloader() {
        use axonml_data::DataLoader;

        let dataset = SyntheticCommandDataset::small();
        let loader = DataLoader::new(dataset, 16);

        let mut batch_count = 0;
        for batch in loader.iter().take(3) {
            assert_eq!(batch.data.shape()[0], 16);
            batch_count += 1;
        }
        assert_eq!(batch_count, 3);
    }
}
