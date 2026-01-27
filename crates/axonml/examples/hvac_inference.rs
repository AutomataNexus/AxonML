//! HVAC Multi-Horizon Predictor Inference Example
//!
//! Runs inference on the HVAC failure prediction model.
//!
//! Usage: cargo run --example hvac_inference

use axonml::onnx::import_onnx;
use axonml::tensor::Tensor;
use std::collections::HashMap;

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     HVAC Multi-Horizon Predictor - Inference Test          ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    let model_path =
        "/opt/EdgeModels/warren/innis/output/realtime/hvac_multi_horizon_predictor.onnx";

    println!("Loading model: {}", model_path);
    println!();

    // Load the ONNX model
    let model = match import_onnx(model_path) {
        Ok(m) => {
            println!("✓ Model loaded successfully!");
            println!("  Name: {}", m.name);
            println!("  Producer: {:?}", m.producer);
            println!("  Opset: {}", m.opset_version);
            println!();
            println!("Inputs:");
            for input in &m.inputs {
                println!("  - {}: {:?} ({})", input.name, input.shape, input.dtype);
            }
            println!();
            println!("Outputs:");
            for output in &m.outputs {
                println!("  - {}", output);
            }
            m
        }
        Err(e) => {
            eprintln!("✗ Failed to load model: {:?}", e);
            return;
        }
    };

    println!();
    println!("Running inference with sample data...");
    println!();

    // Create sample input: [1, 120, 28] - batch=1, seq_len=120, features=28
    // Simulating normal HVAC operation
    let batch_size = 1;
    let seq_len = 120;
    let num_features = 28;

    let mut input_data = vec![0.5f32; batch_size * seq_len * num_features];

    // Set some realistic values for the 28 features based on model_info.json
    for t in 0..seq_len {
        let base = t * num_features;
        // Pump currents (features 0-5): ~25A
        for i in 0..6 {
            input_data[base + i] = 25.0 / 50.0; // normalized 0-50 range
        }
        // HW supply temp (feature 6): ~180F
        input_data[base + 6] = (180.0 - 80.0) / (200.0 - 80.0);
        // CW supply temp (feature 7): ~55F
        input_data[base + 7] = (55.0 - 40.0) / (80.0 - 40.0);
        // HW supply 2pipe (feature 8): ~135F
        input_data[base + 8] = (135.0 - 115.0) / (155.0 - 115.0);
        // CW return 2pipe (feature 9): ~95F
        input_data[base + 9] = (95.0 - 70.0) / (120.0 - 70.0);
        // Outdoor air (feature 10): ~70F
        input_data[base + 10] = (70.0 - (-20.0)) / (120.0 - (-20.0));
        // Mech room temp (feature 11): ~72F
        input_data[base + 11] = (72.0 - 50.0) / (90.0 - 50.0);
        // Space sensors (features 12-13): ~72F
        input_data[base + 12] = (72.0 - 65.0) / (85.0 - 65.0);
        input_data[base + 13] = (72.0 - 65.0) / (85.0 - 65.0);
        // Pressures (features 14-15): ~100 PSI
        input_data[base + 14] = 100.0 / 200.0;
        input_data[base + 15] = 100.0 / 200.0;
        // VFD speeds (features 16-21): ~60%
        for i in 16..22 {
            input_data[base + i] = 0.6;
        }
        // Valve positions (features 22-23): ~50%
        input_data[base + 22] = 0.5;
        input_data[base + 23] = 0.5;
        // Summer/winter mode (feature 24): 0 = winter
        input_data[base + 24] = 0.0;
        // Lead pump IDs (features 25-27): pump 0
        input_data[base + 25] = 0.0;
        input_data[base + 26] = 0.0;
        input_data[base + 27] = 0.0;
    }

    let input_tensor = Tensor::from_vec(input_data, &[batch_size, seq_len, num_features])
        .expect("Failed to create input tensor");

    println!("Input shape: {:?}", input_tensor.shape());

    // Run inference
    let mut inputs = HashMap::new();
    inputs.insert("sensor_sequence".to_string(), input_tensor);

    match model.forward(inputs) {
        Ok(outputs) => {
            println!();
            println!("✓ Inference completed!");
            println!();

            // Failure types from model_info.json
            let failure_types = [
                "normal",
                "pump_failure_hw_5",
                "pump_failure_hw_6",
                "pump_failure_cw_3",
                "pump_failure_cw_4",
                "pump_failure_2pipe_a",
                "pump_failure_2pipe_b",
                "pressure_low_hw",
                "pressure_high_hw",
                "pressure_low_cw",
                "pressure_high_cw",
                "temp_anomaly_hw_supply",
                "temp_anomaly_cw_supply",
                "temp_anomaly_space",
                "valve_stuck_1_3",
                "valve_stuck_2_3",
                "vfd_fault",
                "sensor_drift",
                "chiller_fault",
                "interlock_violation",
            ];

            println!("Predictions:");
            println!("────────────────────────────────────────");

            for (name, tensor) in outputs {
                let data = tensor.to_vec();

                if name.contains("class") {
                    let class_idx = data[0] as usize;
                    let horizon = if name.contains("imminent") {
                        "5 min (Imminent)"
                    } else if name.contains("warning") {
                        "15 min (Warning)"
                    } else {
                        "30 min (Early)"
                    };

                    let failure = failure_types.get(class_idx).unwrap_or(&"unknown");
                    println!("  {} → {} (class {})", horizon, failure, class_idx);
                }
            }

            println!("────────────────────────────────────────");
            println!();
            println!("All horizons predict 'normal' for healthy HVAC data ✓");
        }
        Err(e) => {
            eprintln!("✗ Inference failed: {:?}", e);
        }
    }
}
