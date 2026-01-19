//! Samplers - Data Access Patterns
//!
//! Provides different strategies for sampling data indices.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use rand::seq::SliceRandom;
use rand::Rng;

// =============================================================================
// Sampler Trait
// =============================================================================

/// Trait for all samplers.
///
/// A sampler generates indices that define the order of data access.
pub trait Sampler: Send + Sync {
    /// Returns the number of samples.
    fn len(&self) -> usize;

    /// Returns true if empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Creates an iterator over indices.
    fn iter(&self) -> Box<dyn Iterator<Item = usize> + '_>;
}

// =============================================================================
// SequentialSampler
// =============================================================================

/// Samples elements sequentially.
pub struct SequentialSampler {
    len: usize,
}

impl SequentialSampler {
    /// Creates a new `SequentialSampler`.
    #[must_use] pub fn new(len: usize) -> Self {
        Self { len }
    }
}

impl Sampler for SequentialSampler {
    fn len(&self) -> usize {
        self.len
    }

    fn iter(&self) -> Box<dyn Iterator<Item = usize> + '_> {
        Box::new(0..self.len)
    }
}

// =============================================================================
// RandomSampler
// =============================================================================

/// Samples elements randomly.
pub struct RandomSampler {
    len: usize,
    replacement: bool,
    num_samples: Option<usize>,
}

impl RandomSampler {
    /// Creates a new `RandomSampler` without replacement.
    #[must_use] pub fn new(len: usize) -> Self {
        Self {
            len,
            replacement: false,
            num_samples: None,
        }
    }

    /// Creates a `RandomSampler` with replacement.
    #[must_use] pub fn with_replacement(len: usize, num_samples: usize) -> Self {
        Self {
            len,
            replacement: true,
            num_samples: Some(num_samples),
        }
    }
}

impl Sampler for RandomSampler {
    fn len(&self) -> usize {
        self.num_samples.unwrap_or(self.len)
    }

    fn iter(&self) -> Box<dyn Iterator<Item = usize> + '_> {
        if self.replacement {
            // With replacement: random sampling
            let len = self.len;
            let num = self.num_samples.unwrap_or(len);
            Box::new(RandomWithReplacementIter::new(len, num))
        } else {
            // Without replacement: shuffled indices
            let mut indices: Vec<usize> = (0..self.len).collect();
            indices.shuffle(&mut rand::thread_rng());
            Box::new(indices.into_iter())
        }
    }
}

/// Iterator for random sampling with replacement.
struct RandomWithReplacementIter {
    len: usize,
    remaining: usize,
}

impl RandomWithReplacementIter {
    fn new(len: usize, num_samples: usize) -> Self {
        Self {
            len,
            remaining: num_samples,
        }
    }
}

impl Iterator for RandomWithReplacementIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        Some(rand::thread_rng().gen_range(0..self.len))
    }
}

// =============================================================================
// SubsetRandomSampler
// =============================================================================

/// Samples randomly from a subset of indices.
pub struct SubsetRandomSampler {
    indices: Vec<usize>,
}

impl SubsetRandomSampler {
    /// Creates a new `SubsetRandomSampler`.
    #[must_use] pub fn new(indices: Vec<usize>) -> Self {
        Self { indices }
    }
}

impl Sampler for SubsetRandomSampler {
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn iter(&self) -> Box<dyn Iterator<Item = usize> + '_> {
        let mut shuffled = self.indices.clone();
        shuffled.shuffle(&mut rand::thread_rng());
        Box::new(shuffled.into_iter())
    }
}

// =============================================================================
// WeightedRandomSampler
// =============================================================================

/// Samples elements with specified weights.
pub struct WeightedRandomSampler {
    weights: Vec<f64>,
    num_samples: usize,
    replacement: bool,
}

impl WeightedRandomSampler {
    /// Creates a new `WeightedRandomSampler`.
    #[must_use] pub fn new(weights: Vec<f64>, num_samples: usize, replacement: bool) -> Self {
        Self {
            weights,
            num_samples,
            replacement,
        }
    }

    /// Samples an index based on weights.
    fn sample_index(&self) -> usize {
        let total: f64 = self.weights.iter().sum();
        let mut cumulative = 0.0;
        let threshold: f64 = rand::thread_rng().gen::<f64>() * total;

        for (i, &weight) in self.weights.iter().enumerate() {
            cumulative += weight;
            if cumulative > threshold {
                return i;
            }
        }
        self.weights.len() - 1
    }
}

impl Sampler for WeightedRandomSampler {
    fn len(&self) -> usize {
        self.num_samples
    }

    fn iter(&self) -> Box<dyn Iterator<Item = usize> + '_> {
        if self.replacement {
            Box::new(WeightedIter::new(self))
        } else {
            // Without replacement: sample all unique indices
            let mut indices = Vec::with_capacity(self.num_samples);
            let mut available: Vec<usize> = (0..self.weights.len()).collect();
            let mut weights = self.weights.clone();

            while indices.len() < self.num_samples && !available.is_empty() {
                let total: f64 = weights.iter().sum();
                if total <= 0.0 {
                    break;
                }

                let threshold: f64 = rand::thread_rng().gen::<f64>() * total;
                let mut cumulative = 0.0;
                let mut selected = 0;

                for (i, &weight) in weights.iter().enumerate() {
                    cumulative += weight;
                    if cumulative > threshold {
                        selected = i;
                        break;
                    }
                }

                indices.push(available[selected]);
                available.remove(selected);
                weights.remove(selected);
            }

            Box::new(indices.into_iter())
        }
    }
}

/// Iterator for weighted random sampling with replacement.
struct WeightedIter<'a> {
    sampler: &'a WeightedRandomSampler,
    remaining: usize,
}

impl<'a> WeightedIter<'a> {
    fn new(sampler: &'a WeightedRandomSampler) -> Self {
        Self {
            sampler,
            remaining: sampler.num_samples,
        }
    }
}

impl Iterator for WeightedIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        Some(self.sampler.sample_index())
    }
}

// =============================================================================
// BatchSampler
// =============================================================================

/// Wraps a sampler to yield batches of indices.
pub struct BatchSampler<S: Sampler> {
    sampler: S,
    batch_size: usize,
    drop_last: bool,
}

impl<S: Sampler> BatchSampler<S> {
    /// Creates a new `BatchSampler`.
    pub fn new(sampler: S, batch_size: usize, drop_last: bool) -> Self {
        Self {
            sampler,
            batch_size,
            drop_last,
        }
    }

    /// Creates an iterator over batches of indices.
    pub fn iter(&self) -> BatchIter {
        let indices: Vec<usize> = self.sampler.iter().collect();
        BatchIter {
            indices,
            batch_size: self.batch_size,
            drop_last: self.drop_last,
            position: 0,
        }
    }

    /// Returns the number of batches.
    pub fn len(&self) -> usize {
        let total = self.sampler.len();
        if self.drop_last {
            total / self.batch_size
        } else {
            total.div_ceil(self.batch_size)
        }
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Iterator over batches of indices.
pub struct BatchIter {
    indices: Vec<usize>,
    batch_size: usize,
    drop_last: bool,
    position: usize,
}

impl Iterator for BatchIter {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.indices.len() {
            return None;
        }

        let end = (self.position + self.batch_size).min(self.indices.len());
        let batch: Vec<usize> = self.indices[self.position..end].to_vec();

        if batch.len() < self.batch_size && self.drop_last {
            return None;
        }

        self.position = end;
        Some(batch)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_sampler() {
        let sampler = SequentialSampler::new(5);
        let indices: Vec<usize> = sampler.iter().collect();
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_random_sampler() {
        let sampler = RandomSampler::new(10);
        let indices: Vec<usize> = sampler.iter().collect();

        assert_eq!(indices.len(), 10);
        // All indices should be unique (no replacement)
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 10);
    }

    #[test]
    fn test_random_sampler_with_replacement() {
        let sampler = RandomSampler::with_replacement(5, 20);
        let indices: Vec<usize> = sampler.iter().collect();

        assert_eq!(indices.len(), 20);
        // All indices should be in valid range
        assert!(indices.iter().all(|&i| i < 5));
    }

    #[test]
    fn test_subset_random_sampler() {
        let sampler = SubsetRandomSampler::new(vec![0, 5, 10, 15]);
        let indices: Vec<usize> = sampler.iter().collect();

        assert_eq!(indices.len(), 4);
        // All returned indices should be from the subset
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 5, 10, 15]);
    }

    #[test]
    fn test_weighted_random_sampler() {
        // Heavy weight on index 0
        let sampler = WeightedRandomSampler::new(vec![100.0, 1.0, 1.0, 1.0], 100, true);
        let indices: Vec<usize> = sampler.iter().collect();

        assert_eq!(indices.len(), 100);
        // Most samples should be index 0
        let zeros = indices.iter().filter(|&&i| i == 0).count();
        assert!(zeros > 50, "Expected mostly zeros, got {zeros}");
    }

    #[test]
    fn test_batch_sampler() {
        let base = SequentialSampler::new(10);
        let sampler = BatchSampler::new(base, 3, false);

        let batches: Vec<Vec<usize>> = sampler.iter().collect();
        assert_eq!(batches.len(), 4); // 10 / 3 = 3 full + 1 partial

        assert_eq!(batches[0], vec![0, 1, 2]);
        assert_eq!(batches[1], vec![3, 4, 5]);
        assert_eq!(batches[2], vec![6, 7, 8]);
        assert_eq!(batches[3], vec![9]); // Partial batch
    }

    #[test]
    fn test_batch_sampler_drop_last() {
        let base = SequentialSampler::new(10);
        let sampler = BatchSampler::new(base, 3, true);

        let batches: Vec<Vec<usize>> = sampler.iter().collect();
        assert_eq!(batches.len(), 3); // 10 / 3 = 3, drop the partial

        assert_eq!(batches[0], vec![0, 1, 2]);
        assert_eq!(batches[1], vec![3, 4, 5]);
        assert_eq!(batches[2], vec![6, 7, 8]);
    }
}
