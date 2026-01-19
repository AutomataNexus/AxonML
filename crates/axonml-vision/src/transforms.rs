//! Image Transforms - Vision-Specific Data Augmentation
//!
//! Provides image-specific transformations for data augmentation and preprocessing.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use axonml_data::Transform;
use axonml_tensor::Tensor;
use rand::Rng;

// =============================================================================
// Resize
// =============================================================================

/// Resizes an image to the specified size using bilinear interpolation.
pub struct Resize {
    height: usize,
    width: usize,
}

impl Resize {
    /// Creates a new Resize transform.
    #[must_use] pub fn new(height: usize, width: usize) -> Self {
        Self { height, width }
    }

    /// Creates a square Resize transform.
    #[must_use] pub fn square(size: usize) -> Self {
        Self::new(size, size)
    }
}

impl Transform for Resize {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let shape = input.shape();

        // Handle different input formats
        match shape.len() {
            2 => resize_2d(input, self.height, self.width),
            3 => resize_3d(input, self.height, self.width),
            4 => resize_4d(input, self.height, self.width),
            _ => input.clone(),
        }
    }
}

/// Bilinear interpolation resize for 2D tensor (H x W).
fn resize_2d(input: &Tensor<f32>, new_h: usize, new_w: usize) -> Tensor<f32> {
    let shape = input.shape();
    let (old_h, old_w) = (shape[0], shape[1]);
    let data = input.to_vec();

    let mut result = vec![0.0; new_h * new_w];

    let scale_h = old_h as f32 / new_h as f32;
    let scale_w = old_w as f32 / new_w as f32;

    for y in 0..new_h {
        for x in 0..new_w {
            let src_y = y as f32 * scale_h;
            let src_x = x as f32 * scale_w;

            let y0 = (src_y.floor() as usize).min(old_h - 1);
            let y1 = (y0 + 1).min(old_h - 1);
            let x0 = (src_x.floor() as usize).min(old_w - 1);
            let x1 = (x0 + 1).min(old_w - 1);

            let dy = src_y - y0 as f32;
            let dx = src_x - x0 as f32;

            let v00 = data[y0 * old_w + x0];
            let v01 = data[y0 * old_w + x1];
            let v10 = data[y1 * old_w + x0];
            let v11 = data[y1 * old_w + x1];

            let value = v00 * (1.0 - dx) * (1.0 - dy)
                + v01 * dx * (1.0 - dy)
                + v10 * (1.0 - dx) * dy
                + v11 * dx * dy;

            result[y * new_w + x] = value;
        }
    }

    Tensor::from_vec(result, &[new_h, new_w]).unwrap()
}

/// Bilinear interpolation resize for 3D tensor (C x H x W).
fn resize_3d(input: &Tensor<f32>, new_h: usize, new_w: usize) -> Tensor<f32> {
    let shape = input.shape();
    let (channels, old_h, old_w) = (shape[0], shape[1], shape[2]);
    let data = input.to_vec();

    let mut result = vec![0.0; channels * new_h * new_w];

    let scale_h = old_h as f32 / new_h as f32;
    let scale_w = old_w as f32 / new_w as f32;

    for c in 0..channels {
        for y in 0..new_h {
            for x in 0..new_w {
                let src_y = y as f32 * scale_h;
                let src_x = x as f32 * scale_w;

                let y0 = (src_y.floor() as usize).min(old_h - 1);
                let y1 = (y0 + 1).min(old_h - 1);
                let x0 = (src_x.floor() as usize).min(old_w - 1);
                let x1 = (x0 + 1).min(old_w - 1);

                let dy = src_y - y0 as f32;
                let dx = src_x - x0 as f32;

                let base = c * old_h * old_w;
                let v00 = data[base + y0 * old_w + x0];
                let v01 = data[base + y0 * old_w + x1];
                let v10 = data[base + y1 * old_w + x0];
                let v11 = data[base + y1 * old_w + x1];

                let value = v00 * (1.0 - dx) * (1.0 - dy)
                    + v01 * dx * (1.0 - dy)
                    + v10 * (1.0 - dx) * dy
                    + v11 * dx * dy;

                result[c * new_h * new_w + y * new_w + x] = value;
            }
        }
    }

    Tensor::from_vec(result, &[channels, new_h, new_w]).unwrap()
}

/// Bilinear interpolation resize for 4D tensor (N x C x H x W).
fn resize_4d(input: &Tensor<f32>, new_h: usize, new_w: usize) -> Tensor<f32> {
    let shape = input.shape();
    let (batch, channels, old_h, old_w) = (shape[0], shape[1], shape[2], shape[3]);
    let data = input.to_vec();

    let mut result = vec![0.0; batch * channels * new_h * new_w];

    let scale_h = old_h as f32 / new_h as f32;
    let scale_w = old_w as f32 / new_w as f32;

    for n in 0..batch {
        for c in 0..channels {
            for y in 0..new_h {
                for x in 0..new_w {
                    let src_y = y as f32 * scale_h;
                    let src_x = x as f32 * scale_w;

                    let y0 = (src_y.floor() as usize).min(old_h - 1);
                    let y1 = (y0 + 1).min(old_h - 1);
                    let x0 = (src_x.floor() as usize).min(old_w - 1);
                    let x1 = (x0 + 1).min(old_w - 1);

                    let dy = src_y - y0 as f32;
                    let dx = src_x - x0 as f32;

                    let base = n * channels * old_h * old_w + c * old_h * old_w;
                    let v00 = data[base + y0 * old_w + x0];
                    let v01 = data[base + y0 * old_w + x1];
                    let v10 = data[base + y1 * old_w + x0];
                    let v11 = data[base + y1 * old_w + x1];

                    let value = v00 * (1.0 - dx) * (1.0 - dy)
                        + v01 * dx * (1.0 - dy)
                        + v10 * (1.0 - dx) * dy
                        + v11 * dx * dy;

                    let out_idx = n * channels * new_h * new_w + c * new_h * new_w + y * new_w + x;
                    result[out_idx] = value;
                }
            }
        }
    }

    Tensor::from_vec(result, &[batch, channels, new_h, new_w]).unwrap()
}

// =============================================================================
// CenterCrop
// =============================================================================

/// Crops the center of an image to the specified size.
pub struct CenterCrop {
    height: usize,
    width: usize,
}

impl CenterCrop {
    /// Creates a new `CenterCrop` transform.
    #[must_use] pub fn new(height: usize, width: usize) -> Self {
        Self { height, width }
    }

    /// Creates a square `CenterCrop` transform.
    #[must_use] pub fn square(size: usize) -> Self {
        Self::new(size, size)
    }
}

impl Transform for CenterCrop {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let shape = input.shape();
        let data = input.to_vec();

        match shape.len() {
            2 => {
                let (h, w) = (shape[0], shape[1]);
                let start_h = (h.saturating_sub(self.height)) / 2;
                let start_w = (w.saturating_sub(self.width)) / 2;
                let crop_h = self.height.min(h);
                let crop_w = self.width.min(w);

                let mut result = Vec::with_capacity(crop_h * crop_w);
                for y in start_h..start_h + crop_h {
                    for x in start_w..start_w + crop_w {
                        result.push(data[y * w + x]);
                    }
                }
                Tensor::from_vec(result, &[crop_h, crop_w]).unwrap()
            }
            3 => {
                let (c, h, w) = (shape[0], shape[1], shape[2]);
                let start_h = (h.saturating_sub(self.height)) / 2;
                let start_w = (w.saturating_sub(self.width)) / 2;
                let crop_h = self.height.min(h);
                let crop_w = self.width.min(w);

                let mut result = Vec::with_capacity(c * crop_h * crop_w);
                for ch in 0..c {
                    for y in start_h..start_h + crop_h {
                        for x in start_w..start_w + crop_w {
                            result.push(data[ch * h * w + y * w + x]);
                        }
                    }
                }
                Tensor::from_vec(result, &[c, crop_h, crop_w]).unwrap()
            }
            _ => input.clone(),
        }
    }
}

// =============================================================================
// RandomHorizontalFlip
// =============================================================================

/// Randomly flips an image horizontally with given probability.
pub struct RandomHorizontalFlip {
    probability: f32,
}

impl RandomHorizontalFlip {
    /// Creates a new `RandomHorizontalFlip` with probability 0.5.
    #[must_use] pub fn new() -> Self {
        Self { probability: 0.5 }
    }

    /// Creates a `RandomHorizontalFlip` with custom probability.
    #[must_use] pub fn with_probability(probability: f32) -> Self {
        Self {
            probability: probability.clamp(0.0, 1.0),
        }
    }
}

impl Default for RandomHorizontalFlip {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for RandomHorizontalFlip {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let mut rng = rand::thread_rng();
        if rng.gen::<f32>() > self.probability {
            return input.clone();
        }

        let shape = input.shape();
        let data = input.to_vec();

        match shape.len() {
            2 => {
                let (h, w) = (shape[0], shape[1]);
                let mut result = vec![0.0; h * w];
                for y in 0..h {
                    for x in 0..w {
                        result[y * w + x] = data[y * w + (w - 1 - x)];
                    }
                }
                Tensor::from_vec(result, shape).unwrap()
            }
            3 => {
                let (c, h, w) = (shape[0], shape[1], shape[2]);
                let mut result = vec![0.0; c * h * w];
                for ch in 0..c {
                    for y in 0..h {
                        for x in 0..w {
                            result[ch * h * w + y * w + x] = data[ch * h * w + y * w + (w - 1 - x)];
                        }
                    }
                }
                Tensor::from_vec(result, shape).unwrap()
            }
            _ => input.clone(),
        }
    }
}

// =============================================================================
// RandomVerticalFlip
// =============================================================================

/// Randomly flips an image vertically with given probability.
pub struct RandomVerticalFlip {
    probability: f32,
}

impl RandomVerticalFlip {
    /// Creates a new `RandomVerticalFlip` with probability 0.5.
    #[must_use] pub fn new() -> Self {
        Self { probability: 0.5 }
    }

    /// Creates a `RandomVerticalFlip` with custom probability.
    #[must_use] pub fn with_probability(probability: f32) -> Self {
        Self {
            probability: probability.clamp(0.0, 1.0),
        }
    }
}

impl Default for RandomVerticalFlip {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for RandomVerticalFlip {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let mut rng = rand::thread_rng();
        if rng.gen::<f32>() > self.probability {
            return input.clone();
        }

        let shape = input.shape();
        let data = input.to_vec();

        match shape.len() {
            2 => {
                let (h, w) = (shape[0], shape[1]);
                let mut result = vec![0.0; h * w];
                for y in 0..h {
                    for x in 0..w {
                        result[y * w + x] = data[(h - 1 - y) * w + x];
                    }
                }
                Tensor::from_vec(result, shape).unwrap()
            }
            3 => {
                let (c, h, w) = (shape[0], shape[1], shape[2]);
                let mut result = vec![0.0; c * h * w];
                for ch in 0..c {
                    for y in 0..h {
                        for x in 0..w {
                            result[ch * h * w + y * w + x] = data[ch * h * w + (h - 1 - y) * w + x];
                        }
                    }
                }
                Tensor::from_vec(result, shape).unwrap()
            }
            _ => input.clone(),
        }
    }
}

// =============================================================================
// RandomRotation
// =============================================================================

/// Randomly rotates an image by 90-degree increments.
pub struct RandomRotation {
    /// Allowed rotations: 0, 90, 180, 270 degrees
    angles: Vec<i32>,
}

impl RandomRotation {
    /// Creates a `RandomRotation` that can rotate by any 90-degree increment.
    #[must_use] pub fn new() -> Self {
        Self {
            angles: vec![0, 90, 180, 270],
        }
    }

    /// Creates a `RandomRotation` with specific allowed angles.
    #[must_use] pub fn with_angles(angles: Vec<i32>) -> Self {
        let valid: Vec<i32> = angles
            .into_iter()
            .filter(|&a| a == 0 || a == 90 || a == 180 || a == 270)
            .collect();
        Self {
            angles: if valid.is_empty() { vec![0] } else { valid },
        }
    }
}

impl Default for RandomRotation {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for RandomRotation {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let mut rng = rand::thread_rng();
        let angle = self.angles[rng.gen_range(0..self.angles.len())];

        if angle == 0 {
            return input.clone();
        }

        let shape = input.shape();
        let data = input.to_vec();

        // Only handle 2D (H x W) for simplicity
        if shape.len() != 2 {
            return input.clone();
        }

        let (h, w) = (shape[0], shape[1]);

        match angle {
            90 => {
                // Rotate 90 degrees clockwise: (x, y) -> (y, h-1-x)
                let mut result = vec![0.0; h * w];
                for y in 0..h {
                    for x in 0..w {
                        result[x * h + (h - 1 - y)] = data[y * w + x];
                    }
                }
                Tensor::from_vec(result, &[w, h]).unwrap()
            }
            180 => {
                // Rotate 180 degrees: (x, y) -> (w-1-x, h-1-y)
                let mut result = vec![0.0; h * w];
                for y in 0..h {
                    for x in 0..w {
                        result[(h - 1 - y) * w + (w - 1 - x)] = data[y * w + x];
                    }
                }
                Tensor::from_vec(result, &[h, w]).unwrap()
            }
            270 => {
                // Rotate 270 degrees clockwise: (x, y) -> (w-1-y, x)
                let mut result = vec![0.0; h * w];
                for y in 0..h {
                    for x in 0..w {
                        result[(w - 1 - x) * h + y] = data[y * w + x];
                    }
                }
                Tensor::from_vec(result, &[w, h]).unwrap()
            }
            _ => input.clone(),
        }
    }
}

// =============================================================================
// ColorJitter
// =============================================================================

/// Randomly adjusts brightness, contrast, saturation of an image.
pub struct ColorJitter {
    brightness: f32,
    contrast: f32,
    saturation: f32,
}

impl ColorJitter {
    /// Creates a new `ColorJitter` with specified ranges.
    #[must_use] pub fn new(brightness: f32, contrast: f32, saturation: f32) -> Self {
        Self {
            brightness: brightness.abs(),
            contrast: contrast.abs(),
            saturation: saturation.abs(),
        }
    }
}

impl Transform for ColorJitter {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let mut rng = rand::thread_rng();
        let mut data = input.to_vec();
        let shape = input.shape();

        // Apply brightness adjustment
        if self.brightness > 0.0 {
            let factor = 1.0 + rng.gen_range(-self.brightness..self.brightness);
            for val in &mut data {
                *val = (*val * factor).clamp(0.0, 1.0);
            }
        }

        // Apply contrast adjustment
        if self.contrast > 0.0 {
            let factor = 1.0 + rng.gen_range(-self.contrast..self.contrast);
            let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
            for val in &mut data {
                *val = ((*val - mean) * factor + mean).clamp(0.0, 1.0);
            }
        }

        // Apply saturation adjustment (simplified - works best with 3-channel images)
        if self.saturation > 0.0 && shape.len() == 3 && shape[0] == 3 {
            let factor = 1.0 + rng.gen_range(-self.saturation..self.saturation);
            let (h, w) = (shape[1], shape[2]);

            for y in 0..h {
                for x in 0..w {
                    let r = data[0 * h * w + y * w + x];
                    let g = data[h * w + y * w + x];
                    let b = data[2 * h * w + y * w + x];

                    let gray = 0.299 * r + 0.587 * g + 0.114 * b;

                    data[0 * h * w + y * w + x] = (gray + (r - gray) * factor).clamp(0.0, 1.0);
                    data[h * w + y * w + x] = (gray + (g - gray) * factor).clamp(0.0, 1.0);
                    data[2 * h * w + y * w + x] = (gray + (b - gray) * factor).clamp(0.0, 1.0);
                }
            }
        }

        Tensor::from_vec(data, shape).unwrap()
    }
}

// =============================================================================
// Grayscale
// =============================================================================

/// Converts an RGB image to grayscale.
pub struct Grayscale {
    num_output_channels: usize,
}

impl Grayscale {
    /// Creates a Grayscale transform with 1 output channel.
    #[must_use] pub fn new() -> Self {
        Self {
            num_output_channels: 1,
        }
    }

    /// Creates a Grayscale transform with specified output channels.
    #[must_use] pub fn with_channels(num_output_channels: usize) -> Self {
        Self {
            num_output_channels: num_output_channels.max(1),
        }
    }
}

impl Default for Grayscale {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for Grayscale {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let shape = input.shape();

        // Only works with 3-channel images (C x H x W)
        if shape.len() != 3 || shape[0] != 3 {
            return input.clone();
        }

        let (_, h, w) = (shape[0], shape[1], shape[2]);
        let data = input.to_vec();

        let mut gray = Vec::with_capacity(h * w);
        for y in 0..h {
            for x in 0..w {
                let r = data[0 * h * w + y * w + x];
                let g = data[h * w + y * w + x];
                let b = data[2 * h * w + y * w + x];
                gray.push(0.299 * r + 0.587 * g + 0.114 * b);
            }
        }

        if self.num_output_channels == 1 {
            Tensor::from_vec(gray, &[1, h, w]).unwrap()
        } else {
            // Replicate grayscale across channels
            let mut result = Vec::with_capacity(self.num_output_channels * h * w);
            for _ in 0..self.num_output_channels {
                result.extend(&gray);
            }
            Tensor::from_vec(result, &[self.num_output_channels, h, w]).unwrap()
        }
    }
}

// =============================================================================
// Normalize (Image-specific)
// =============================================================================

/// Normalizes an image with per-channel mean and std.
pub struct ImageNormalize {
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl ImageNormalize {
    /// Creates a new `ImageNormalize` with per-channel mean and std.
    #[must_use] pub fn new(mean: Vec<f32>, std: Vec<f32>) -> Self {
        Self { mean, std }
    }

    /// Creates normalization for `ImageNet` pretrained models.
    #[must_use] pub fn imagenet() -> Self {
        Self::new(vec![0.485, 0.456, 0.406], vec![0.229, 0.224, 0.225])
    }

    /// Creates normalization for MNIST (single channel).
    #[must_use] pub fn mnist() -> Self {
        Self::new(vec![0.1307], vec![0.3081])
    }

    /// Creates normalization for CIFAR-10.
    #[must_use] pub fn cifar10() -> Self {
        Self::new(vec![0.4914, 0.4822, 0.4465], vec![0.2470, 0.2435, 0.2616])
    }
}

impl Transform for ImageNormalize {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let shape = input.shape();
        let mut data = input.to_vec();

        match shape.len() {
            3 => {
                let (c, h, w) = (shape[0], shape[1], shape[2]);
                for ch in 0..c {
                    let mean = self.mean.get(ch).copied().unwrap_or(0.0);
                    let std = self.std.get(ch).copied().unwrap_or(1.0);
                    for y in 0..h {
                        for x in 0..w {
                            let idx = ch * h * w + y * w + x;
                            data[idx] = (data[idx] - mean) / std;
                        }
                    }
                }
            }
            4 => {
                let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
                for batch in 0..n {
                    for ch in 0..c {
                        let mean = self.mean.get(ch).copied().unwrap_or(0.0);
                        let std = self.std.get(ch).copied().unwrap_or(1.0);
                        for y in 0..h {
                            for x in 0..w {
                                let idx = batch * c * h * w + ch * h * w + y * w + x;
                                data[idx] = (data[idx] - mean) / std;
                            }
                        }
                    }
                }
            }
            _ => {}
        }

        Tensor::from_vec(data, shape).unwrap()
    }
}

// =============================================================================
// Pad
// =============================================================================

/// Pads an image with a constant value.
pub struct Pad {
    padding: (usize, usize, usize, usize), // (left, right, top, bottom)
    fill_value: f32,
}

impl Pad {
    /// Creates a new Pad with uniform padding.
    #[must_use] pub fn new(padding: usize) -> Self {
        Self {
            padding: (padding, padding, padding, padding),
            fill_value: 0.0,
        }
    }

    /// Creates a Pad with asymmetric padding.
    #[must_use] pub fn asymmetric(left: usize, right: usize, top: usize, bottom: usize) -> Self {
        Self {
            padding: (left, right, top, bottom),
            fill_value: 0.0,
        }
    }

    /// Sets the fill value.
    #[must_use] pub fn with_fill(mut self, value: f32) -> Self {
        self.fill_value = value;
        self
    }
}

impl Transform for Pad {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let shape = input.shape();
        let data = input.to_vec();
        let (left, right, top, bottom) = self.padding;

        match shape.len() {
            2 => {
                let (h, w) = (shape[0], shape[1]);
                let new_h = h + top + bottom;
                let new_w = w + left + right;

                let mut result = vec![self.fill_value; new_h * new_w];
                for y in 0..h {
                    for x in 0..w {
                        result[(y + top) * new_w + (x + left)] = data[y * w + x];
                    }
                }
                Tensor::from_vec(result, &[new_h, new_w]).unwrap()
            }
            3 => {
                let (c, h, w) = (shape[0], shape[1], shape[2]);
                let new_h = h + top + bottom;
                let new_w = w + left + right;

                let mut result = vec![self.fill_value; c * new_h * new_w];
                for ch in 0..c {
                    for y in 0..h {
                        for x in 0..w {
                            result[ch * new_h * new_w + (y + top) * new_w + (x + left)] =
                                data[ch * h * w + y * w + x];
                        }
                    }
                }
                Tensor::from_vec(result, &[c, new_h, new_w]).unwrap()
            }
            _ => input.clone(),
        }
    }
}

// =============================================================================
// ToTensorImage
// =============================================================================

/// Converts image data from [0, 255] to [0, 1] range.
pub struct ToTensorImage;

impl ToTensorImage {
    /// Creates a new `ToTensorImage` transform.
    #[must_use] pub fn new() -> Self {
        Self
    }
}

impl Default for ToTensorImage {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for ToTensorImage {
    fn apply(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let data: Vec<f32> = input.to_vec().iter().map(|&x| x / 255.0).collect();
        Tensor::from_vec(data, input.shape()).unwrap()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resize_2d() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let resize = Resize::new(4, 4);
        let output = resize.apply(&input);

        assert_eq!(output.shape(), &[4, 4]);
    }

    #[test]
    fn test_resize_3d() {
        let input = Tensor::from_vec(vec![1.0; 3 * 8 * 8], &[3, 8, 8]).unwrap();

        let resize = Resize::new(4, 4);
        let output = resize.apply(&input);

        assert_eq!(output.shape(), &[3, 4, 4]);
    }

    #[test]
    fn test_center_crop() {
        let input = Tensor::from_vec((1..=16).map(|x| x as f32).collect(), &[4, 4]).unwrap();

        let crop = CenterCrop::new(2, 2);
        let output = crop.apply(&input);

        assert_eq!(output.shape(), &[2, 2]);
        // Center 2x2 of 4x4: values 6, 7, 10, 11
        assert_eq!(output.to_vec(), vec![6.0, 7.0, 10.0, 11.0]);
    }

    #[test]
    fn test_random_horizontal_flip() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let flip = RandomHorizontalFlip::with_probability(1.0);
        let output = flip.apply(&input);

        // [[1, 2], [3, 4]] -> [[2, 1], [4, 3]]
        assert_eq!(output.to_vec(), vec![2.0, 1.0, 4.0, 3.0]);
    }

    #[test]
    fn test_random_vertical_flip() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let flip = RandomVerticalFlip::with_probability(1.0);
        let output = flip.apply(&input);

        // [[1, 2], [3, 4]] -> [[3, 4], [1, 2]]
        assert_eq!(output.to_vec(), vec![3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn test_random_rotation_180() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let rotation = RandomRotation::with_angles(vec![180]);
        let output = rotation.apply(&input);

        // [[1, 2], [3, 4]] rotated 180 -> [[4, 3], [2, 1]]
        assert_eq!(output.to_vec(), vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_grayscale() {
        let input = Tensor::from_vec(
            vec![
                1.0, 1.0, 1.0, 1.0, // R channel
                0.5, 0.5, 0.5, 0.5, // G channel
                0.0, 0.0, 0.0, 0.0, // B channel
            ],
            &[3, 2, 2],
        )
        .unwrap();

        let gray = Grayscale::new();
        let output = gray.apply(&input);

        assert_eq!(output.shape(), &[1, 2, 2]);
        // Gray = 0.299 * 1.0 + 0.587 * 0.5 + 0.114 * 0.0 = 0.5925
        let expected = 0.299 + 0.587 * 0.5;
        for val in output.to_vec() {
            assert!((val - expected).abs() < 0.001);
        }
    }

    #[test]
    fn test_image_normalize() {
        let input = Tensor::from_vec(vec![0.5; 3 * 2 * 2], &[3, 2, 2]).unwrap();

        let normalize = ImageNormalize::new(vec![0.5, 0.5, 0.5], vec![0.5, 0.5, 0.5]);
        let output = normalize.apply(&input);

        // (0.5 - 0.5) / 0.5 = 0.0
        for val in output.to_vec() {
            assert!((val - 0.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_pad() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let pad = Pad::new(1);
        let output = pad.apply(&input);

        assert_eq!(output.shape(), &[4, 4]);
        // Check corners are zero
        let data = output.to_vec();
        assert_eq!(data[0], 0.0);
        assert_eq!(data[3], 0.0);
        assert_eq!(data[12], 0.0);
        assert_eq!(data[15], 0.0);
        // Check center values
        assert_eq!(data[5], 1.0);
        assert_eq!(data[6], 2.0);
        assert_eq!(data[9], 3.0);
        assert_eq!(data[10], 4.0);
    }

    #[test]
    fn test_to_tensor_image() {
        let input = Tensor::from_vec(vec![0.0, 127.5, 255.0], &[3]).unwrap();

        let transform = ToTensorImage::new();
        let output = transform.apply(&input);

        let data = output.to_vec();
        assert!((data[0] - 0.0).abs() < 0.001);
        assert!((data[1] - 0.5).abs() < 0.001);
        assert!((data[2] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_color_jitter() {
        let input = Tensor::from_vec(vec![0.5; 3 * 4 * 4], &[3, 4, 4]).unwrap();

        let jitter = ColorJitter::new(0.1, 0.1, 0.1);
        let output = jitter.apply(&input);

        assert_eq!(output.shape(), &[3, 4, 4]);
        // Values should be in valid range
        for val in output.to_vec() {
            assert!((0.0..=1.0).contains(&val));
        }
    }
}
