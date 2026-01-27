//! Function Cache
//!
//! Caches compiled functions for reuse.

use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::codegen::CompiledFunction;
use crate::ir::Graph;

/// Cache for compiled functions.
pub struct FunctionCache {
    cache: RwLock<FxHashMap<u64, CompiledFunction>>,
    max_size: usize,
}

impl FunctionCache {
    /// Creates a new function cache.
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: RwLock::new(FxHashMap::default()),
            max_size,
        }
    }

    /// Creates a cache with default size (1000).
    pub fn default_size() -> Self {
        Self::new(1000)
    }

    /// Computes a hash key for a graph.
    pub fn hash_graph(graph: &Graph) -> u64 {
        let mut hasher = DefaultHasher::new();

        // Hash graph structure
        for node in graph.nodes() {
            // Hash op type
            std::mem::discriminant(&node.op).hash(&mut hasher);

            // Hash dtype
            node.dtype.hash(&mut hasher);

            // Hash shape
            node.shape.dims().hash(&mut hasher);

            // Hash op-specific data
            match &node.op {
                crate::ir::Op::Input { name } => name.hash(&mut hasher),
                crate::ir::Op::Output { name, input } => {
                    name.hash(&mut hasher);
                    input.index().hash(&mut hasher);
                }
                crate::ir::Op::Constant { value } => {
                    value.to_bits().hash(&mut hasher);
                }
                crate::ir::Op::AddScalar { input, scalar }
                | crate::ir::Op::MulScalar { input, scalar } => {
                    input.index().hash(&mut hasher);
                    scalar.to_bits().hash(&mut hasher);
                }
                crate::ir::Op::Reshape { input, shape } => {
                    input.index().hash(&mut hasher);
                    shape.hash(&mut hasher);
                }
                crate::ir::Op::Transpose { input, dim0, dim1 } => {
                    input.index().hash(&mut hasher);
                    dim0.hash(&mut hasher);
                    dim1.hash(&mut hasher);
                }
                crate::ir::Op::SumAxis {
                    input,
                    axis,
                    keepdim,
                }
                | crate::ir::Op::MeanAxis {
                    input,
                    axis,
                    keepdim,
                }
                | crate::ir::Op::MaxAxis {
                    input,
                    axis,
                    keepdim,
                } => {
                    input.index().hash(&mut hasher);
                    axis.hash(&mut hasher);
                    keepdim.hash(&mut hasher);
                }
                crate::ir::Op::Squeeze { input, dim } | crate::ir::Op::Unsqueeze { input, dim } => {
                    input.index().hash(&mut hasher);
                    dim.hash(&mut hasher);
                }
                crate::ir::Op::Broadcast { input, shape } => {
                    input.index().hash(&mut hasher);
                    shape.hash(&mut hasher);
                }
                crate::ir::Op::Cast { input, dtype } => {
                    input.index().hash(&mut hasher);
                    dtype.hash(&mut hasher);
                }
                // Binary ops
                crate::ir::Op::Add { lhs, rhs }
                | crate::ir::Op::Sub { lhs, rhs }
                | crate::ir::Op::Mul { lhs, rhs }
                | crate::ir::Op::Div { lhs, rhs }
                | crate::ir::Op::Pow {
                    base: lhs,
                    exp: rhs,
                }
                | crate::ir::Op::Max { lhs, rhs }
                | crate::ir::Op::Min { lhs, rhs }
                | crate::ir::Op::MatMul { lhs, rhs }
                | crate::ir::Op::Gt { lhs, rhs }
                | crate::ir::Op::Lt { lhs, rhs }
                | crate::ir::Op::Eq { lhs, rhs } => {
                    lhs.index().hash(&mut hasher);
                    rhs.index().hash(&mut hasher);
                }
                // Ternary ops
                crate::ir::Op::Where { condition, x, y } => {
                    condition.index().hash(&mut hasher);
                    x.index().hash(&mut hasher);
                    y.index().hash(&mut hasher);
                }
                // Unary ops just hash input
                _ => {
                    for input in node.op.inputs() {
                        input.index().hash(&mut hasher);
                    }
                }
            }
        }

        hasher.finish()
    }

    /// Gets a cached function or returns None.
    pub fn get(&self, key: u64) -> Option<CompiledFunction> {
        self.cache.read().get(&key).cloned()
    }

    /// Gets a cached function by graph.
    pub fn get_by_graph(&self, graph: &Graph) -> Option<CompiledFunction> {
        let key = Self::hash_graph(graph);
        self.get(key)
    }

    /// Inserts a compiled function.
    pub fn insert(&self, key: u64, func: CompiledFunction) {
        let mut cache = self.cache.write();

        // Evict if at capacity
        if cache.len() >= self.max_size {
            // Simple eviction: remove first entry
            if let Some(&first_key) = cache.keys().next() {
                cache.remove(&first_key);
            }
        }

        cache.insert(key, func);
    }

    /// Inserts a compiled function for a graph.
    pub fn insert_for_graph(&self, graph: &Graph, func: CompiledFunction) {
        let key = Self::hash_graph(graph);
        self.insert(key, func);
    }

    /// Returns the number of cached functions.
    pub fn len(&self) -> usize {
        self.cache.read().len()
    }

    /// Returns whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.read().is_empty()
    }

    /// Clears the cache.
    pub fn clear(&self) {
        self.cache.write().clear();
    }

    /// Returns cache statistics.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.len(),
            max_size: self.max_size,
        }
    }
}

impl Default for FunctionCache {
    fn default() -> Self {
        Self::default_size()
    }
}

/// Cache statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of cached entries.
    pub entries: usize,
    /// Maximum cache size.
    pub max_size: usize,
}

impl CacheStats {
    /// Returns the utilization as a percentage.
    pub fn utilization(&self) -> f64 {
        if self.max_size == 0 {
            0.0
        } else {
            (self.entries as f64 / self.max_size as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::trace;

    #[test]
    fn test_graph_hash() {
        let graph1 = trace(|tracer| {
            let a = tracer.input("a", &[2, 3]);
            let b = tracer.input("b", &[2, 3]);
            let c = a.add(&b);
            tracer.output("result", c)
        });

        let graph2 = trace(|tracer| {
            let a = tracer.input("a", &[2, 3]);
            let b = tracer.input("b", &[2, 3]);
            let c = a.add(&b);
            tracer.output("result", c)
        });

        let graph3 = trace(|tracer| {
            let a = tracer.input("a", &[2, 3]);
            let b = tracer.input("b", &[2, 3]);
            let c = a.mul(&b); // Different op
            tracer.output("result", c)
        });

        let hash1 = FunctionCache::hash_graph(&graph1);
        let hash2 = FunctionCache::hash_graph(&graph2);
        let hash3 = FunctionCache::hash_graph(&graph3);

        // Same structure should have same hash
        assert_eq!(hash1, hash2);
        // Different structure should have different hash
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_cache_insert_get() {
        let cache = FunctionCache::new(10);

        let graph = trace(|tracer| {
            let a = tracer.input("a", &[2, 3]);
            tracer.output("result", a.relu())
        });

        let key = FunctionCache::hash_graph(&graph);
        let func = CompiledFunction::placeholder();

        assert!(cache.get(key).is_none());
        cache.insert(key, func.clone());
        assert!(cache.get(key).is_some());
    }

    #[test]
    fn test_cache_eviction() {
        let cache = FunctionCache::new(2);

        for i in 0..3 {
            cache.insert(i as u64, CompiledFunction::placeholder());
        }

        // Cache should only have 2 entries
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_cache_stats() {
        let cache = FunctionCache::new(100);
        cache.insert(1, CompiledFunction::placeholder());
        cache.insert(2, CompiledFunction::placeholder());

        let stats = cache.stats();
        assert_eq!(stats.entries, 2);
        assert_eq!(stats.max_size, 100);
        assert!((stats.utilization() - 2.0).abs() < 0.01);
    }
}
