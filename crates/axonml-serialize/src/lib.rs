//! Axonml Serialize - Model Serialization for Axonml ML Framework
//!
//! # File
//! `crates/axonml-serialize/src/lib.rs`
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Updated
//! March 8, 2026
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
    OnnxOpType, convert_from_pytorch, from_onnx_shape, from_pytorch_key, pytorch_layer_mapping,
    to_onnx_shape, to_pytorch_key, transpose_linear_weights,
};
pub use format::{Format, detect_format, detect_format_from_bytes};
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

/// Load a model's parameters from a saved state dictionary file.
///
/// Matches parameters by name. Returns the number of parameters loaded.
/// Parameters not found in the state dict are left unchanged.
///
/// # Example
/// ```rust,ignore
/// let mut model = MyModel::new();
/// let loaded = load_model(&model, "model.axonml")?;
/// println!("Loaded {loaded} parameters");
/// ```
pub fn load_model<M: Module, P: AsRef<Path>>(model: &M, path: P) -> Result<usize> {
    let state_dict = load_state_dict(path)?;
    let named_params = model.named_parameters();
    let mut loaded = 0;

    for (name, param) in &named_params {
        if let Some(entry) = state_dict.get(name) {
            if let Ok(tensor) = entry.data.to_tensor() {
                if tensor.shape() == param.data().shape() {
                    param.update_data(tensor);
                    loaded += 1;
                }
            }
        }
    }

    // If named_parameters returned empty (model doesn't implement it),
    // fall back to positional parameter matching
    if named_params.is_empty() {
        let params = model.parameters();
        let entries: Vec<_> = state_dict.entries().collect();
        for ((_name, entry), param) in entries.iter().zip(params.iter()) {
            if let Ok(tensor) = entry.data.to_tensor() {
                if tensor.shape() == param.data().shape() {
                    param.update_data(tensor);
                    loaded += 1;
                }
            }
        }
    }

    Ok(loaded)
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
    use safetensors::tensor::{Dtype, TensorView};
    use std::collections::HashMap;

    // Build TensorView data for each parameter
    let mut tensor_data: HashMap<String, Vec<u8>> = HashMap::new();
    let mut tensor_shapes: HashMap<String, Vec<usize>> = HashMap::new();

    for (name, entry) in state_dict.entries() {
        let data_bytes: Vec<u8> = entry
            .data
            .values
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        tensor_data.insert(name.clone(), data_bytes);
        tensor_shapes.insert(name.clone(), entry.data.shape.clone());
    }

    // Create TensorViews referencing the data
    let views: Vec<(String, TensorView<'_>)> = tensor_data
        .iter()
        .map(|(name, data)| {
            let shape = tensor_shapes.get(name).expect("shape missing");
            (
                name.clone(),
                TensorView::new(Dtype::F32, shape.clone(), data).expect("invalid tensor view"),
            )
        })
        .collect();

    let view_refs: Vec<(&str, TensorView<'_>)> = views
        .iter()
        .map(|(name, view)| (name.as_str(), view.clone()))
        .collect();

    let bytes = safetensors::tensor::serialize(&view_refs, &None)
        .map_err(|e| Error::InvalidOperation {
            message: format!("SafeTensors serialize failed: {e}"),
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

        // Convert bytes to f32 based on dtype
        let dtype = tensor.dtype();
        let values: Vec<f32> = match dtype {
            safetensors::Dtype::F32 => {
                data.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
            safetensors::Dtype::F16 => {
                data.chunks_exact(2)
                    .map(|c| {
                        let h = half::f16::from_le_bytes([c[0], c[1]]);
                        h.to_f32()
                    })
                    .collect()
            }
            safetensors::Dtype::BF16 => {
                data.chunks_exact(2)
                    .map(|c| {
                        let h = half::bf16::from_le_bytes([c[0], c[1]]);
                        h.to_f32()
                    })
                    .collect()
            }
            safetensors::Dtype::F64 => {
                data.chunks_exact(8)
                    .map(|c| {
                        let v = f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
                        v as f32
                    })
                    .collect()
            }
            other => {
                return Err(Error::InvalidOperation {
                    message: format!("Unsupported safetensors dtype: {:?} for tensor '{}'", other, name),
                });
            }
        };

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

    #[test]
    fn test_state_dict_file_roundtrip_axonml() {
        let mut sd = StateDict::new();
        sd.insert(
            "layer.weight".to_string(),
            TensorData {
                shape: vec![3, 2],
                values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            },
        );
        sd.insert(
            "layer.bias".to_string(),
            TensorData {
                shape: vec![3],
                values: vec![0.1, 0.2, 0.3],
            },
        );

        let path = std::env::temp_dir().join("axonml_test_roundtrip.axonml");
        save_state_dict(&sd, &path, Format::Axonml).expect("save failed");

        let loaded = load_state_dict(&path).expect("load failed");
        assert_eq!(loaded.len(), 2);
        assert!(loaded.contains("layer.weight"));
        assert!(loaded.contains("layer.bias"));

        let w = loaded.get("layer.weight").unwrap();
        assert_eq!(w.data.shape, vec![3, 2]);
        assert_eq!(w.data.values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_state_dict_file_roundtrip_json() {
        let mut sd = StateDict::new();
        sd.insert(
            "fc.weight".to_string(),
            TensorData {
                shape: vec![2, 2],
                values: vec![1.5, -2.5, 3.5, -4.5],
            },
        );

        let path = std::env::temp_dir().join("axonml_test_roundtrip.json");
        save_state_dict(&sd, &path, Format::Json).expect("save failed");

        let loaded = load_state_dict(&path).expect("load failed");
        assert_eq!(loaded.len(), 1);
        let w = loaded.get("fc.weight").unwrap();
        assert_eq!(w.data.values, vec![1.5, -2.5, 3.5, -4.5]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_checkpoint_file_roundtrip() {
        let mut state = TrainingState::new();
        state.epoch = 10;
        state.global_step = 5000;
        state.record_loss(0.5);
        state.record_loss(0.3);
        state.update_best("loss", 0.3, false);

        let checkpoint = Checkpoint::builder()
            .training_state(state)
            .epoch(10)
            .global_step(5000)
            .config("lr", "0.001")
            .build();

        let path = std::env::temp_dir().join("axonml_test_checkpoint.axonml");
        save_checkpoint(&checkpoint, &path).expect("save failed");

        let loaded = load_checkpoint(&path).expect("load failed");
        assert_eq!(loaded.epoch(), 10);
        assert_eq!(loaded.global_step(), 5000);
        assert_eq!(loaded.best_metric(), Some(0.3));
        assert!(loaded.config.contains_key("lr"));
        assert!(loaded.timestamp.contains('T')); // ISO 8601 format

        std::fs::remove_file(&path).ok();
    }
}
