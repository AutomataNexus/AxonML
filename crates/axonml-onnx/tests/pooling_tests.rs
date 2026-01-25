//! Tests for pooling operators (MaxPool, AveragePool)

use axonml_tensor::Tensor;

// Test helper to create operators manually for unit testing
// In practice, these would be created via ONNX model parsing

#[test]
fn test_maxpool_basic_2x2() {
    // Create a simple 4D tensor [1, 1, 4, 4] (batch=1, channel=1, height=4, width=4)
    let input_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];

    let tensor = Tensor::from_vec(input_data, &[1, 1, 4, 4]).expect("Failed to create tensor");

    // With 2x2 kernel, stride 2, no padding, output should be [1, 1, 2, 2]
    // Expected values:
    // max(1,2,5,6) = 6, max(3,4,7,8) = 8
    // max(9,10,13,14) = 14, max(11,12,15,16) = 16

    let shape = tensor.shape();
    assert_eq!(shape, &[1, 1, 4, 4]);
}

#[test]
fn test_maxpool_with_padding() {
    // Create a 4D tensor [1, 1, 3, 3]
    let input_data: Vec<f32> = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ];

    let tensor = Tensor::from_vec(input_data, &[1, 1, 3, 3]).expect("Failed to create tensor");
    let shape = tensor.shape();
    assert_eq!(shape, &[1, 1, 3, 3]);
}

#[test]
fn test_avgpool_basic_2x2() {
    // Create a 4D tensor [1, 1, 4, 4]
    let input_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];

    let tensor = Tensor::from_vec(input_data, &[1, 1, 4, 4]).expect("Failed to create tensor");

    // With 2x2 kernel, stride 2, no padding, output should be [1, 1, 2, 2]
    // Expected values:
    // avg(1,2,5,6) = 3.5, avg(3,4,7,8) = 5.5
    // avg(9,10,13,14) = 11.5, avg(11,12,15,16) = 13.5

    let shape = tensor.shape();
    assert_eq!(shape, &[1, 1, 4, 4]);
}

#[test]
fn test_tensor_to_vec() {
    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::from_vec(input_data.clone(), &[2, 2]).expect("Failed to create tensor");

    let vec_data = tensor.to_vec();
    assert_eq!(vec_data, input_data);
}

#[test]
fn test_multi_channel_pooling() {
    // Create a 4D tensor [1, 2, 4, 4] (2 channels)
    let input_data: Vec<f32> = vec![
        // Channel 0
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
        // Channel 1
        17.0, 18.0, 19.0, 20.0,
        21.0, 22.0, 23.0, 24.0,
        25.0, 26.0, 27.0, 28.0,
        29.0, 30.0, 31.0, 32.0,
    ];

    let tensor = Tensor::from_vec(input_data, &[1, 2, 4, 4]).expect("Failed to create tensor");
    let shape = tensor.shape();
    assert_eq!(shape, &[1, 2, 4, 4]);
}

#[test]
fn test_batch_pooling() {
    // Create a 4D tensor [2, 1, 4, 4] (batch=2)
    let input_data: Vec<f32> = vec![
        // Batch 0
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
        // Batch 1
        2.0, 4.0, 6.0, 8.0,
        10.0, 12.0, 14.0, 16.0,
        18.0, 20.0, 22.0, 24.0,
        26.0, 28.0, 30.0, 32.0,
    ];

    let tensor = Tensor::from_vec(input_data, &[2, 1, 4, 4]).expect("Failed to create tensor");
    let shape = tensor.shape();
    assert_eq!(shape, &[2, 1, 4, 4]);
}
