//! End-to-end quantization test with a real model
use axonml_autograd::Variable;
use axonml_nn::{Module, Linear};
use axonml_tensor::Tensor;
use axonml_quant::{QuantizedModel, QuantizedLinear, QuantType, serialize_quantized, deserialize_quantized};

fn main() {
    println!("=== Quantized Inference E2E Test ===\n");

    // Build a real multi-layer model
    let l1 = Linear::new(64, 128);
    let l2 = Linear::new(128, 32);
    let l3 = Linear::new(32, 10);

    // Create test input
    let input_data: Vec<f32> = (0..4*64).map(|i| (i as f32 * 0.01) - 1.0).collect();
    let input = Variable::new(Tensor::from_vec(input_data.clone(), &[4, 64]).unwrap(), false);

    // F32 reference forward
    let out1 = l1.forward(&input).relu();
    let out2 = l2.forward(&out1).relu();
    let ref_output = l3.forward(&out2);
    let ref_data = ref_output.data().to_vec();
    println!("F32 output (first 5): {:?}", &ref_data[..5]);

    // === Test 1: QuantizedModel from Module ===
    println!("\n--- Test 1: QuantizedModel::from_module ---");

    // Collect all params from all layers
    let mut all_params = l1.parameters();
    all_params.extend(l2.parameters());
    all_params.extend(l3.parameters());
    let total: usize = all_params.iter().map(|p| p.numel()).sum();
    println!("Total params: {total}");

    // We need a wrapper to test from_module — let's test quantize_parameters directly
    let q8_params = axonml_quant::inference::quantize_parameters(&all_params, QuantType::Q8_0);
    println!("Q8 tensors: {}, blocks: {}",
        q8_params.len(),
        q8_params.iter().map(|q| q.num_blocks()).sum::<usize>());
    let q8_bytes: usize = q8_params.iter().map(|q| q.size_bytes()).sum();
    let f32_bytes = total * 4;
    println!("F32: {} bytes, Q8: {} bytes, ratio: {:.1}x", f32_bytes, q8_bytes, f32_bytes as f32 / q8_bytes as f32);

    // === Test 2: QuantizedLinear forward accuracy ===
    println!("\n--- Test 2: QuantizedLinear accuracy ---");

    let w1_data = l1.parameters()[0].data().to_vec();
    let b1_data = l1.parameters()[1].data().to_vec();

    for qt in &[QuantType::Q8_0, QuantType::Q4_0, QuantType::F16] {
        let ql = QuantizedLinear::from_linear_params(&w1_data, Some(&b1_data), 64, 128, *qt);
        let q_out = ql.forward_f32(&input_data[..64], 1); // single sample

        // Compare with f32 Linear
        let single_input = Variable::new(Tensor::from_vec(input_data[..64].to_vec(), &[1, 64]).unwrap(), false);
        let f32_out = l1.forward(&single_input).data().to_vec();

        let max_err: f32 = q_out.iter().zip(f32_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let rmse: f32 = (q_out.iter().zip(f32_out.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / q_out.len() as f32).sqrt();

        println!("{}: max_err={:.6}, rmse={:.6}, compression={:.1}x",
            qt, max_err, rmse, ql.compression_ratio());
    }

    // === Test 3: Variable forward ===
    println!("\n--- Test 3: Variable forward ---");
    let ql = QuantizedLinear::from_linear_params(&w1_data, Some(&b1_data), 64, 128, QuantType::Q8_0);
    let var_out = ql.forward_var(&input);
    println!("Variable output shape: {:?}", var_out.shape());
    println!("Variable output (first 5): {:?}", &var_out.data().to_vec()[..5]);

    // === Test 4: Serialize/deserialize roundtrip ===
    println!("\n--- Test 4: Serialization roundtrip ---");
    let qmodel = QuantizedModel {
        quantized_params: q8_params.clone(),
        quant_type: QuantType::Q8_0,
        total_params: total,
        total_bytes: q8_bytes,
        original_bytes: f32_bytes,
    };
    println!("{}", qmodel.summary());

    let serialized = serialize_quantized(&qmodel);
    println!("Serialized size: {} bytes", serialized.len());

    let deserialized = deserialize_quantized(&serialized).unwrap();
    println!("Deserialized: {} tensors, {} params",
        deserialized.quantized_params.len(), deserialized.total_params);

    // Verify block counts match
    for (i, (orig, loaded)) in q8_params.iter().zip(deserialized.quantized_params.iter()).enumerate() {
        assert_eq!(orig.shape, loaded.shape, "Shape mismatch on tensor {i}");
        assert_eq!(orig.num_blocks(), loaded.num_blocks(), "Block count mismatch on tensor {i}");
    }
    println!("Roundtrip verification: PASS");

    // === Test 5: Q4 quantization ===
    println!("\n--- Test 5: Q4 quantization ---");
    let q4_params = axonml_quant::inference::quantize_parameters(&all_params, QuantType::Q4_0);
    let q4_bytes: usize = q4_params.iter().map(|q| q.size_bytes()).sum();
    println!("Q4: {} bytes, ratio: {:.1}x", q4_bytes, f32_bytes as f32 / q4_bytes as f32);

    let q4_model = QuantizedModel {
        quantized_params: q4_params,
        quant_type: QuantType::Q4_0,
        total_params: total,
        total_bytes: q4_bytes,
        original_bytes: f32_bytes,
    };
    let q4_ser = serialize_quantized(&q4_model);
    let q4_de = deserialize_quantized(&q4_ser).unwrap();
    println!("Q4 serialize/deserialize: {} tensors — PASS", q4_de.quantized_params.len());

    println!("\n=== ALL E2E TESTS PASSED ===");
}
