//! Axonml Serialize - Model Serialization for Axonml ML Framework
//!
//! This crate provides functionality for saving and loading trained models,
//! including state dictionaries, model checkpoints, and format conversion.
//!
//! # Supported Formats
//!
//! - **Axonml Native** (.axonml) - Efficient binary format
//! - **JSON** (.json) - Human-readable format for debugging
//! - **`SafeTensors`** (.safetensors) - Safe, fast format (optional feature)
//!
//! # Example
//!
//! ```ignore
//! use axonml_serialize::{save_model, load_model, StateDict};
//!
//! // Save model
//! save_model(&model, "model.axonml")?;
//!
//! // Load model
//! let state_dict = load_state_dict("model.axonml")?;
//! model.load_state_dict(&state_dict)?;
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

// =============================================================================
// Modules
// =============================================================================

mod checkpoint;
mod convert;
mod format;
mod state_dict;

// =============================================================================
// Re-exports
// =============================================================================

pub use checkpoint::{Checkpoint, CheckpointBuilder, TrainingState};
pub use convert::{
    convert_from_pytorch, from_onnx_shape, from_pytorch_key, pytorch_layer_mapping, to_onnx_shape,
    to_pytorch_key, transpose_linear_weights, OnnxOpType,
};
pub use format::{detect_format, detect_format_from_bytes, Format};
pub use state_dict::{StateDict, StateDictEntry, TensorData};

// =============================================================================
// Imports
// =============================================================================

use axonml_core::{Error, Result};
use axonml_nn::Module;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

// =============================================================================
// High-Level API
// =============================================================================

/// Save a model's state dictionary to a file.
///
/// The format is automatically determined from the file extension.
pub fn save_model<M: Module, P: AsRef<Path>>(model: &M, path: P) -> Result<()> {
    let path = path.as_ref();
    let format = detect_format(path);
    let state_dict = StateDict::from_module(model);

    save_state_dict(&state_dict, path, format)
}

/// Save a state dictionary to a file with specified format.
pub fn save_state_dict<P: AsRef<Path>>(
    state_dict: &StateDict,
    path: P,
    format: Format,
) -> Result<()> {
    let path = path.as_ref();
    let file = File::create(path).map_err(|e| Error::InvalidOperation {
        message: e.to_string(),
    })?;
    let mut writer = BufWriter::new(file);

    match format {
        Format::Axonml => {
            let encoded = bincode::serialize(state_dict).map_err(|e| Error::InvalidOperation {
                message: e.to_string(),
            })?;
            writer
                .write_all(&encoded)
                .map_err(|e| Error::InvalidOperation {
                    message: e.to_string(),
                })?;
        }
        Format::Json => {
            serde_json::to_writer_pretty(&mut writer, state_dict).map_err(|e| {
                Error::InvalidOperation {
                    message: e.to_string(),
                }
            })?;
        }
        #[cfg(feature = "safetensors")]
        Format::SafeTensors => {
            save_safetensors(state_dict, path)?;
        }
        #[cfg(not(feature = "safetensors"))]
        Format::SafeTensors => {
            return Err(Error::InvalidOperation {
                message: "SafeTensors format requires 'safetensors' feature".to_string(),
            });
        }
    }

    Ok(())
}

/// Load a state dictionary from a file.
pub fn load_state_dict<P: AsRef<Path>>(path: P) -> Result<StateDict> {
    let path = path.as_ref();
    let format = detect_format(path);

    let file = File::open(path).map_err(|e| Error::InvalidOperation {
        message: e.to_string(),
    })?;
    let mut reader = BufReader::new(file);

    match format {
        Format::Axonml => {
            let mut bytes = Vec::new();
            reader
                .read_to_end(&mut bytes)
                .map_err(|e| Error::InvalidOperation {
                    message: e.to_string(),
                })?;
            bincode::deserialize(&bytes).map_err(|e| Error::InvalidOperation {
                message: e.to_string(),
            })
        }
        Format::Json => serde_json::from_reader(reader).map_err(|e| Error::InvalidOperation {
            message: e.to_string(),
        }),
        #[cfg(feature = "safetensors")]
        Format::SafeTensors => load_safetensors(path),
        #[cfg(not(feature = "safetensors"))]
        Format::SafeTensors => Err(Error::InvalidOperation {
            message: "SafeTensors format requires 'safetensors' feature".to_string(),
        }),
    }
}

/// Save a complete training checkpoint.
pub fn save_checkpoint<P: AsRef<Path>>(checkpoint: &Checkpoint, path: P) -> Result<()> {
    let path = path.as_ref();
    let file = File::create(path).map_err(|e| Error::InvalidOperation {
        message: e.to_string(),
    })?;
    let writer = BufWriter::new(file);

    bincode::serialize_into(writer, checkpoint).map_err(|e| Error::InvalidOperation {
        message: e.to_string(),
    })
}

/// Load a training checkpoint.
pub fn load_checkpoint<P: AsRef<Path>>(path: P) -> Result<Checkpoint> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|e| Error::InvalidOperation {
        message: e.to_string(),
    })?;
    let reader = BufReader::new(file);

    bincode::deserialize_from(reader).map_err(|e| Error::InvalidOperation {
        message: e.to_string(),
    })
}

// =============================================================================
// SafeTensors Support
// =============================================================================

#[cfg(feature = "safetensors")]
fn save_safetensors<P: AsRef<Path>>(state_dict: &StateDict, path: P) -> Result<()> {
    use safetensors::tensor::SafeTensors;
    use std::collections::HashMap;

    let mut tensors: HashMap<String, Vec<u8>> = HashMap::new();
    let mut metadata: HashMap<String, String> = HashMap::new();

    for (name, entry) in state_dict.entries() {
        let data_bytes: Vec<u8> = entry
            .data
            .values
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        tensors.insert(name.clone(), data_bytes);
        metadata.insert(format!("{}.shape", name), format!("{:?}", entry.data.shape));
    }

    // Write using safetensors
    let bytes =
        safetensors::serialize(&tensors, &Some(metadata)).map_err(|e| Error::InvalidOperation {
            message: e.to_string(),
        })?;

    std::fs::write(path, bytes).map_err(|e| Error::InvalidOperation {
        message: e.to_string(),
    })
}

#[cfg(feature = "safetensors")]
fn load_safetensors<P: AsRef<Path>>(path: P) -> Result<StateDict> {
    let bytes = std::fs::read(path).map_err(|e| Error::InvalidOperation {
        message: e.to_string(),
    })?;

    let tensors =
        safetensors::SafeTensors::deserialize(&bytes).map_err(|e| Error::InvalidOperation {
            message: e.to_string(),
        })?;

    let mut state_dict = StateDict::new();

    for (name, tensor) in tensors.tensors() {
        let data = tensor.data();
        let shape: Vec<usize> = tensor.shape().to_vec();

        // Convert bytes to f32
        let values: Vec<f32> = data
            .chunks(4)
            .map(|chunk| {
                let bytes: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                f32::from_le_bytes(bytes)
            })
            .collect();

        state_dict.insert(name.to_string(), TensorData { shape, values });
    }

    Ok(state_dict)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_detection() {
        assert_eq!(detect_format("model.axonml"), Format::Axonml);
        assert_eq!(detect_format("model.json"), Format::Json);
        assert_eq!(detect_format("model.safetensors"), Format::SafeTensors);
        assert_eq!(detect_format("model.bin"), Format::Axonml); // default
    }

    #[test]
    fn test_state_dict_creation() {
        let state_dict = StateDict::new();
        assert!(state_dict.is_empty());
        assert_eq!(state_dict.len(), 0);
    }

    #[test]
    fn test_state_dict_insert_get() {
        let mut state_dict = StateDict::new();
        let data = TensorData {
            shape: vec![2, 3],
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };

        state_dict.insert("layer.weight".to_string(), data);

        assert_eq!(state_dict.len(), 1);
        assert!(state_dict.contains("layer.weight"));

        let retrieved = state_dict.get("layer.weight").unwrap();
        assert_eq!(retrieved.data.shape, vec![2, 3]);
    }

    #[test]
    fn test_tensor_data_to_tensor() {
        let data = TensorData {
            shape: vec![2, 2],
            values: vec![1.0, 2.0, 3.0, 4.0],
        };

        let tensor = data.to_tensor().unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }
}
