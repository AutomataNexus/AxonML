//! Calibration for Quantization
//!
//! Calibration methods for determining optimal quantization parameters.
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use axonml_tensor::Tensor;

use crate::error::{QuantError, QuantResult};
use crate::types::QuantType;

// =============================================================================
// Calibration Data
// =============================================================================

/// Calibration data collected from sample inputs.
#[derive(Debug, Clone)]
pub struct CalibrationData {
    /// Minimum value seen.
    pub min: f32,
    /// Maximum value seen.
    pub max: f32,
    /// Mean value.
    pub mean: f32,
    /// Standard deviation.
    pub std_dev: f32,
    /// Number of samples.
    pub num_samples: usize,
    /// Histogram buckets (for percentile calibration).
    histogram: Vec<usize>,
    /// Histogram bin edges.
    bin_edges: Vec<f32>,
}

impl CalibrationData {
    /// Creates new calibration data from initial tensor.
    pub fn new(tensor: &Tensor<f32>, num_bins: usize) -> Self {
        let data = tensor.to_vec();
        let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mean = data.iter().sum::<f32>() / data.len() as f32;

        let variance = data
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>()
            / data.len() as f32;
        let std_dev = variance.sqrt();

        // Initialize histogram
        let bin_width = (max - min) / num_bins as f32;
        let mut histogram = vec![0usize; num_bins];
        let bin_edges: Vec<f32> = (0..=num_bins)
            .map(|i| min + i as f32 * bin_width)
            .collect();

        for &val in &data {
            let bin = ((val - min) / bin_width) as usize;
            let bin = bin.min(num_bins - 1);
            histogram[bin] += 1;
        }

        Self {
            min,
            max,
            mean,
            std_dev,
            num_samples: data.len(),
            histogram,
            bin_edges,
        }
    }

    /// Updates calibration data with more samples.
    pub fn update(&mut self, tensor: &Tensor<f32>) {
        let data = tensor.to_vec();
        let new_min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let new_max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Update min/max
        self.min = self.min.min(new_min);
        self.max = self.max.max(new_max);

        // Update running mean
        let old_count = self.num_samples as f32;
        let new_count = data.len() as f32;
        let new_mean = data.iter().sum::<f32>() / new_count;
        self.mean = (self.mean * old_count + new_mean * new_count) / (old_count + new_count);

        // Update histogram (rebuild with new range)
        self.num_samples += data.len();
        // Note: For proper histogram update, we'd need to keep all data or use streaming algorithms
    }

    /// Returns the dynamic range.
    pub fn dynamic_range(&self) -> f32 {
        self.max - self.min
    }

    /// Computes the optimal scale for symmetric quantization.
    pub fn symmetric_scale(&self, quant_type: QuantType) -> f32 {
        let max_abs = self.min.abs().max(self.max.abs());
        let max_int = match quant_type {
            QuantType::Q8_0 => 127.0,
            QuantType::Q4_0 | QuantType::Q4_1 => 7.0,
            QuantType::Q5_0 | QuantType::Q5_1 => 15.0,
            QuantType::F16 | QuantType::F32 => 1.0,
        };
        max_abs / max_int
    }

    /// Computes the optimal scale for asymmetric quantization.
    pub fn asymmetric_scale(&self, quant_type: QuantType) -> (f32, f32) {
        let max_int = match quant_type {
            QuantType::Q8_0 => 255.0,
            QuantType::Q4_0 | QuantType::Q4_1 => 15.0,
            QuantType::Q5_0 | QuantType::Q5_1 => 31.0,
            QuantType::F16 | QuantType::F32 => 1.0,
        };

        let scale = (self.max - self.min) / max_int;
        let zero_point = -self.min / scale;

        (scale, zero_point)
    }

    /// Returns the percentile value from the histogram.
    pub fn percentile(&self, p: f32) -> f32 {
        if p <= 0.0 {
            return self.min;
        }
        if p >= 100.0 {
            return self.max;
        }

        let target = (p / 100.0 * self.num_samples as f32) as usize;
        let mut cumsum = 0usize;

        for (i, &count) in self.histogram.iter().enumerate() {
            cumsum += count;
            if cumsum >= target {
                return self.bin_edges[i];
            }
        }

        self.max
    }
}

// =============================================================================
// Calibration Methods
// =============================================================================

/// Calibration method enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationMethod {
    /// Use min/max values directly.
    MinMax,
    /// Use percentiles (e.g., 99.9th) to reduce outlier impact.
    Percentile(u32), // percentile * 10 (e.g., 999 = 99.9%)
    /// Use entropy-based calibration (KL divergence).
    Entropy,
    /// Use mean + k*std_dev for range.
    MeanStd(u32), // k * 10 (e.g., 30 = 3.0 sigma)
}

/// Calibrates a tensor for quantization.
///
/// # Arguments
/// * `tensor` - The tensor to calibrate
/// * `method` - The calibration method to use
///
/// # Returns
/// Calibration data for the tensor
pub fn calibrate(tensor: &Tensor<f32>, method: CalibrationMethod) -> QuantResult<CalibrationData> {
    let mut data = CalibrationData::new(tensor, 2048);

    match method {
        CalibrationMethod::MinMax => {
            // Already computed in new()
        }
        CalibrationMethod::Percentile(p) => {
            let percentile = p as f32 / 10.0;
            let lower = data.percentile(100.0 - percentile);
            let upper = data.percentile(percentile);
            data.min = lower;
            data.max = upper;
        }
        CalibrationMethod::MeanStd(k) => {
            let k_factor = k as f32 / 10.0;
            data.min = data.mean - k_factor * data.std_dev;
            data.max = data.mean + k_factor * data.std_dev;
        }
        CalibrationMethod::Entropy => {
            // Simplified entropy calibration - use 99.99th percentile
            data.min = data.percentile(0.01);
            data.max = data.percentile(99.99);
        }
    }

    Ok(data)
}

/// Calibrates multiple tensors and returns combined calibration data.
pub fn calibrate_batch(
    tensors: &[&Tensor<f32>],
    method: CalibrationMethod,
) -> QuantResult<CalibrationData> {
    if tensors.is_empty() {
        return Err(QuantError::CalibrationError("No tensors provided".to_string()));
    }

    let mut combined = CalibrationData::new(tensors[0], 2048);

    for tensor in tensors.iter().skip(1) {
        combined.update(tensor);
    }

    // Apply method-specific adjustments
    match method {
        CalibrationMethod::Percentile(p) => {
            let percentile = p as f32 / 10.0;
            combined.min = combined.percentile(100.0 - percentile);
            combined.max = combined.percentile(percentile);
        }
        CalibrationMethod::MeanStd(k) => {
            let k_factor = k as f32 / 10.0;
            combined.min = combined.mean - k_factor * combined.std_dev;
            combined.max = combined.mean + k_factor * combined.std_dev;
        }
        _ => {}
    }

    Ok(combined)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::from_vec(data, &[5]).unwrap();

        let calib = CalibrationData::new(&tensor, 10);

        assert_eq!(calib.min, 1.0);
        assert_eq!(calib.max, 5.0);
        assert_eq!(calib.mean, 3.0);
        assert_eq!(calib.num_samples, 5);
    }

    #[test]
    fn test_symmetric_scale() {
        let data = vec![-4.0, -2.0, 0.0, 2.0, 4.0];
        let tensor = Tensor::from_vec(data, &[5]).unwrap();

        let calib = CalibrationData::new(&tensor, 10);
        let scale = calib.symmetric_scale(QuantType::Q8_0);

        // max_abs = 4.0, max_int = 127, scale = 4/127
        assert!((scale - 4.0 / 127.0).abs() < 0.001);
    }

    #[test]
    fn test_calibration_methods() {
        let data: Vec<f32> = (0..1000).map(|x| x as f32 / 100.0).collect();
        let tensor = Tensor::from_vec(data, &[1000]).unwrap();

        // Min/Max calibration
        let minmax = calibrate(&tensor, CalibrationMethod::MinMax).unwrap();
        assert!((minmax.min - 0.0).abs() < 0.01);
        assert!((minmax.max - 9.99).abs() < 0.01);

        // Percentile calibration (99.9%)
        let percentile = calibrate(&tensor, CalibrationMethod::Percentile(999)).unwrap();
        assert!(percentile.min >= 0.0);
        assert!(percentile.max <= 9.99);
    }

    #[test]
    fn test_dynamic_range() {
        let data = vec![-5.0, 10.0];
        let tensor = Tensor::from_vec(data, &[2]).unwrap();

        let calib = CalibrationData::new(&tensor, 10);
        assert_eq!(calib.dynamic_range(), 15.0);
    }
}
