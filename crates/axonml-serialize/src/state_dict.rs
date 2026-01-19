//! State Dictionary - Model parameter storage
//!
//! Provides `StateDict` for storing and retrieving model parameters by name.

use axonml_core::Result;
use axonml_nn::Module;
use axonml_tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// TensorData
// =============================================================================

/// Serializable tensor data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorData {
    /// Shape of the tensor.
    pub shape: Vec<usize>,
    /// Flattened f32 values.
    pub values: Vec<f32>,
}

impl TensorData {
    /// Create `TensorData` from a Tensor.
    #[must_use] pub fn from_tensor(tensor: &Tensor<f32>) -> Self {
        Self {
            shape: tensor.shape().to_vec(),
            values: tensor.to_vec(),
        }
    }

    /// Convert `TensorData` back to a Tensor.
    pub fn to_tensor(&self) -> Result<Tensor<f32>> {
        Tensor::from_vec(self.values.clone(), &self.shape)
    }

    /// Get the number of elements.
    #[must_use] pub fn numel(&self) -> usize {
        self.values.len()
    }

    /// Get the shape.
    #[must_use] pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

// =============================================================================
// StateDictEntry
// =============================================================================

/// An entry in the state dictionary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateDictEntry {
    /// The tensor data.
    pub data: TensorData,
    /// Whether this parameter requires gradients.
    pub requires_grad: bool,
    /// Optional metadata.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl StateDictEntry {
    /// Create a new entry from tensor data.
    #[must_use] pub fn new(data: TensorData, requires_grad: bool) -> Self {
        Self {
            data,
            requires_grad,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the entry.
    #[must_use] pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

// =============================================================================
// StateDict
// =============================================================================

/// State dictionary for storing model parameters.
///
/// This is similar to `PyTorch`'s `state_dict`, mapping parameter names to tensors.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StateDict {
    entries: HashMap<String, StateDictEntry>,
    #[serde(default)]
    metadata: HashMap<String, String>,
}

impl StateDict {
    /// Create an empty state dictionary.
    #[must_use] pub fn new() -> Self {
        Self::default()
    }

    /// Create a state dictionary from a module.
    pub fn from_module<M: Module>(module: &M) -> Self {
        let mut state_dict = Self::new();

        for param in module.parameters() {
            let name = param.name().to_string();
            let tensor_data = TensorData::from_tensor(&param.data());
            let entry = StateDictEntry::new(tensor_data, param.requires_grad());
            state_dict.entries.insert(name, entry);
        }

        state_dict
    }

    /// Insert a tensor into the state dictionary.
    pub fn insert(&mut self, name: String, data: TensorData) {
        let entry = StateDictEntry::new(data, true);
        self.entries.insert(name, entry);
    }

    /// Insert an entry into the state dictionary.
    pub fn insert_entry(&mut self, name: String, entry: StateDictEntry) {
        self.entries.insert(name, entry);
    }

    /// Get an entry by name.
    #[must_use] pub fn get(&self, name: &str) -> Option<&StateDictEntry> {
        self.entries.get(name)
    }

    /// Get a mutable entry by name.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut StateDictEntry> {
        self.entries.get_mut(name)
    }

    /// Check if the state dictionary contains a key.
    #[must_use] pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Get the number of entries.
    #[must_use] pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the state dictionary is empty.
    #[must_use] pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get all keys.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.entries.keys()
    }

    /// Get all entries.
    pub fn entries(&self) -> impl Iterator<Item = (&String, &StateDictEntry)> {
        self.entries.iter()
    }

    /// Remove an entry.
    pub fn remove(&mut self, name: &str) -> Option<StateDictEntry> {
        self.entries.remove(name)
    }

    /// Merge another state dictionary into this one.
    pub fn merge(&mut self, other: StateDict) {
        for (name, entry) in other.entries {
            self.entries.insert(name, entry);
        }
    }

    /// Get a subset of entries matching a prefix.
    #[must_use] pub fn filter_prefix(&self, prefix: &str) -> StateDict {
        let mut filtered = StateDict::new();
        for (name, entry) in &self.entries {
            if name.starts_with(prefix) {
                filtered.entries.insert(name.clone(), entry.clone());
            }
        }
        filtered
    }

    /// Strip a prefix from all keys.
    #[must_use] pub fn strip_prefix(&self, prefix: &str) -> StateDict {
        let mut stripped = StateDict::new();
        for (name, entry) in &self.entries {
            let new_name = name.strip_prefix(prefix).unwrap_or(name).to_string();
            stripped.entries.insert(new_name, entry.clone());
        }
        stripped
    }

    /// Add a prefix to all keys.
    #[must_use] pub fn add_prefix(&self, prefix: &str) -> StateDict {
        let mut prefixed = StateDict::new();
        for (name, entry) in &self.entries {
            let new_name = format!("{prefix}{name}");
            prefixed.entries.insert(new_name, entry.clone());
        }
        prefixed
    }

    /// Set metadata on the state dictionary.
    pub fn set_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    /// Get metadata from the state dictionary.
    #[must_use] pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }

    /// Get total number of parameters (elements across all tensors).
    #[must_use] pub fn total_params(&self) -> usize {
        self.entries.values().map(|e| e.data.numel()).sum()
    }

    /// Get total size in bytes.
    #[must_use] pub fn size_bytes(&self) -> usize {
        self.total_params() * std::mem::size_of::<f32>()
    }

    /// Print a summary of the state dictionary.
    #[must_use] pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("StateDict with {} entries:", self.len()));
        lines.push(format!("  Total parameters: {}", self.total_params()));
        lines.push(format!("  Size: {} bytes", self.size_bytes()));
        lines.push("  Entries:".to_string());

        for (name, entry) in &self.entries {
            lines.push(format!(
                "    {} - shape: {:?}, numel: {}",
                name,
                entry.data.shape,
                entry.data.numel()
            ));
        }

        lines.join("\n")
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_data_roundtrip() {
        let original = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let data = TensorData::from_tensor(&original);
        let restored = data.to_tensor().unwrap();

        assert_eq!(original.shape(), restored.shape());
        assert_eq!(original.to_vec(), restored.to_vec());
    }

    #[test]
    fn test_state_dict_operations() {
        let mut state_dict = StateDict::new();

        let data1 = TensorData {
            shape: vec![10, 5],
            values: vec![0.0; 50],
        };
        let data2 = TensorData {
            shape: vec![5],
            values: vec![0.0; 5],
        };

        state_dict.insert("linear.weight".to_string(), data1);
        state_dict.insert("linear.bias".to_string(), data2);

        assert_eq!(state_dict.len(), 2);
        assert_eq!(state_dict.total_params(), 55);
        assert!(state_dict.contains("linear.weight"));
        assert!(state_dict.contains("linear.bias"));
    }

    #[test]
    fn test_state_dict_filter_prefix() {
        let mut state_dict = StateDict::new();

        state_dict.insert(
            "encoder.layer1.weight".to_string(),
            TensorData {
                shape: vec![10],
                values: vec![0.0; 10],
            },
        );
        state_dict.insert(
            "encoder.layer1.bias".to_string(),
            TensorData {
                shape: vec![10],
                values: vec![0.0; 10],
            },
        );
        state_dict.insert(
            "decoder.layer1.weight".to_string(),
            TensorData {
                shape: vec![10],
                values: vec![0.0; 10],
            },
        );

        let encoder_dict = state_dict.filter_prefix("encoder.");
        assert_eq!(encoder_dict.len(), 2);
        assert!(encoder_dict.contains("encoder.layer1.weight"));
    }

    #[test]
    fn test_state_dict_strip_prefix() {
        let mut state_dict = StateDict::new();

        state_dict.insert(
            "model.linear.weight".to_string(),
            TensorData {
                shape: vec![10],
                values: vec![0.0; 10],
            },
        );

        let stripped = state_dict.strip_prefix("model.");
        assert!(stripped.contains("linear.weight"));
    }

    #[test]
    fn test_state_dict_merge() {
        let mut dict1 = StateDict::new();
        dict1.insert(
            "a".to_string(),
            TensorData {
                shape: vec![1],
                values: vec![1.0],
            },
        );

        let mut dict2 = StateDict::new();
        dict2.insert(
            "b".to_string(),
            TensorData {
                shape: vec![1],
                values: vec![2.0],
            },
        );

        dict1.merge(dict2);
        assert_eq!(dict1.len(), 2);
        assert!(dict1.contains("a"));
        assert!(dict1.contains("b"));
    }

    #[test]
    fn test_state_dict_summary() {
        let mut state_dict = StateDict::new();
        state_dict.insert(
            "weight".to_string(),
            TensorData {
                shape: vec![10, 5],
                values: vec![0.0; 50],
            },
        );

        let summary = state_dict.summary();
        assert!(summary.contains("1 entries"));
        assert!(summary.contains("50"));
    }
}
