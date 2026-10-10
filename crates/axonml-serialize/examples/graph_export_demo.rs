//! Emit a graph-bearing .axonml with a NON-registered architecture tag, to prove a
//! downstream compiler consumes the embedded graph rather than reconstructing by name.
//!
//! Copyright (c) 2026 Andrew Jewell Sr. — AutomataNexus LLC.

use axonml_nn::{BatchNorm2d, Conv2d, Flatten, Linear, ReLU, Sequential};
use axonml_serialize::save_model_with_graph;
use axonml_tensor::Tensor;

fn main() {
    let model = Sequential::new()
        .add(Conv2d::new(3, 8, 3))
        .add(BatchNorm2d::new(8))
        .add(ReLU::new())
        .add(Flatten::new())
        .add(Linear::new(8 * 6 * 6, 4));
    let example = Tensor::from_vec(vec![0.1f32; 3 * 8 * 8], &[1, 3, 8, 8]).expect("tensor");
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/demo.axonml".to_string());
    let out = save_model_with_graph(&model, &example, "acme_detector_v1", &path).expect("save");
    println!("wrote {}", out.display());
}
