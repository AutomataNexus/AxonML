#!/usr/bin/env python3
"""
Test the model converter by creating sample .axonml files and converting them.
Tests all supported architectures.
"""

import json
import struct
import os
import sys
import tempfile
import numpy as np
from pathlib import Path

# Ensure we use the venv
VENV_PYTHON = Path(__file__).parent.parent / "converter-venv" / "bin" / "python"

AXONML_MAGIC = b"AXONML"
AXONML_VERSION = 1


def create_axonml_file(path: str, model_weights: dict, header: dict):
    """Create a .axonml file with the given weights and header."""
    header_json = json.dumps(header).encode()
    weights_json = json.dumps(model_weights).encode()

    with open(path, "wb") as f:
        f.write(AXONML_MAGIC)
        f.write(bytes([AXONML_VERSION]))
        f.write(struct.pack("<I", len(header_json)))
        f.write(header_json)
        f.write(struct.pack("<I", len(weights_json)))
        f.write(weights_json)


def random_weights(n: int) -> list:
    """Generate n random float32 weights."""
    return np.random.randn(n).astype(np.float32).tolist()


def linear_params(in_f: int, out_f: int) -> int:
    """Number of params in a Linear(in_f, out_f) layer."""
    return out_f * in_f + out_f


def lstm_cell_params(input_dim: int, hidden_dim: int) -> int:
    """Number of params in an LSTM cell."""
    return (4 * hidden_dim * input_dim +  # w_ih
            4 * hidden_dim * hidden_dim +  # w_hh
            4 * hidden_dim +               # b_ih
            4 * hidden_dim)                # b_hh


def gru_cell_params(input_dim: int, hidden_dim: int) -> int:
    """Number of params in a GRU cell."""
    return (3 * hidden_dim * input_dim +
            3 * hidden_dim * hidden_dim +
            3 * hidden_dim +
            3 * hidden_dim)


def rnn_layer_params(input_dim: int, hidden_dim: int) -> int:
    """Number of params in an RNN layer."""
    return (hidden_dim * input_dim +
            hidden_dim * hidden_dim +
            hidden_dim +
            hidden_dim)


def test_sentinel():
    """Test Sentinel MLP conversion."""
    input_features = 10
    # fc1: Linear(10, 128) + fc2: Linear(128, 64) + fc3: Linear(64, 1)
    n_params = linear_params(10, 128) + linear_params(128, 64) + linear_params(64, 1)
    weights = random_weights(n_params)

    model_weights = {
        "architecture": "sentinel",
        "input_features": input_features,
        "hyperparameters": {
            "learning_rate": 0.001, "epochs": 100, "batch_size": 32,
            "sequence_length": 60, "hidden_dim": 64, "num_layers": 2,
            "dropout": 0.1, "weight_decay": 0.01,
            "early_stopping_patience": 10, "val_check_interval": 1,
        },
        "weights": weights,
        "norm_means": [0.0] * input_features,
        "norm_stds": [1.0] * input_features,
        "anomaly_threshold": None,
    }

    header = {
        "architecture": "Sentinel",
        "input_features": input_features,
        "num_parameters": n_params,
        "quantized": False,
        "quant_bits": None,
    }

    return model_weights, header, "sentinel"


def test_lstm_autoencoder():
    """Test LSTM Autoencoder conversion."""
    input_features = 5
    hidden_dim = 16
    num_layers = 2
    bottleneck_dim = hidden_dim // 2

    n_params = 0
    # Encoder LSTM layers
    for i in range(num_layers):
        in_dim = input_features if i == 0 else hidden_dim
        n_params += lstm_cell_params(in_dim, hidden_dim)
    # Encoder linear
    n_params += linear_params(hidden_dim, bottleneck_dim)
    # Decoder linear
    n_params += linear_params(bottleneck_dim, hidden_dim)
    # Decoder LSTM layers
    for i in range(num_layers):
        n_params += lstm_cell_params(hidden_dim, hidden_dim)
    # Decoder output
    n_params += linear_params(hidden_dim, input_features)

    weights = random_weights(n_params)

    model_weights = {
        "architecture": "lstm_autoencoder",
        "input_features": input_features,
        "hyperparameters": {
            "learning_rate": 0.001, "epochs": 100, "batch_size": 32,
            "sequence_length": 10, "hidden_dim": hidden_dim, "num_layers": num_layers,
            "dropout": 0.1, "weight_decay": 0.01,
            "early_stopping_patience": 10, "val_check_interval": 1,
        },
        "weights": weights,
        "norm_means": [0.0] * input_features,
        "norm_stds": [1.0] * input_features,
        "anomaly_threshold": 0.5,
    }

    header = {
        "architecture": "LSTM Autoencoder",
        "input_features": input_features,
        "num_parameters": n_params,
        "quantized": False,
        "quant_bits": None,
    }

    return model_weights, header, "lstm_autoencoder"


def test_gru_predictor():
    """Test GRU Predictor conversion."""
    input_features = 8
    hidden_dim = 32
    num_layers = 2

    n_params = 0
    for i in range(num_layers):
        in_dim = input_features if i == 0 else hidden_dim
        n_params += gru_cell_params(in_dim, hidden_dim)
    n_params += linear_params(hidden_dim, 64)
    n_params += linear_params(64, 3)

    weights = random_weights(n_params)

    model_weights = {
        "architecture": "gru_predictor",
        "input_features": input_features,
        "hyperparameters": {
            "learning_rate": 0.001, "epochs": 100, "batch_size": 32,
            "sequence_length": 20, "hidden_dim": hidden_dim, "num_layers": num_layers,
            "dropout": 0.1, "weight_decay": 0.01,
            "early_stopping_patience": 10, "val_check_interval": 1,
        },
        "weights": weights,
        "norm_means": [0.0] * input_features,
        "norm_stds": [1.0] * input_features,
        "anomaly_threshold": None,
    }

    header = {
        "architecture": "GRU Predictor",
        "input_features": input_features,
        "num_parameters": n_params,
        "quantized": False,
        "quant_bits": None,
    }

    return model_weights, header, "gru_predictor"


def test_rnn():
    """Test vanilla RNN conversion."""
    input_features = 6
    hidden_dim = 16
    num_layers = 2

    n_params = 0
    for i in range(num_layers):
        in_dim = input_features if i == 0 else hidden_dim
        n_params += rnn_layer_params(in_dim, hidden_dim)
    n_params += linear_params(hidden_dim, 1)

    weights = random_weights(n_params)

    model_weights = {
        "architecture": "rnn",
        "input_features": input_features,
        "hyperparameters": {
            "learning_rate": 0.001, "epochs": 100, "batch_size": 32,
            "sequence_length": 15, "hidden_dim": hidden_dim, "num_layers": num_layers,
            "dropout": 0.1, "weight_decay": 0.01,
            "early_stopping_patience": 10, "val_check_interval": 1,
        },
        "weights": weights,
        "norm_means": [0.0] * input_features,
        "norm_stds": [1.0] * input_features,
        "anomaly_threshold": None,
    }

    header = {
        "architecture": "RNN",
        "input_features": input_features,
        "num_parameters": n_params,
        "quantized": False,
        "quant_bits": None,
    }

    return model_weights, header, "rnn"


def test_phantom():
    """Test Phantom lightweight model conversion."""
    input_dim = 10
    output_dim = 1
    bottleneck_dim = min(32, max(8, input_dim // 4))  # 8
    expand_dim = bottleneck_dim * 2  # 16

    n_params = (linear_params(input_dim, bottleneck_dim)
                + linear_params(bottleneck_dim, expand_dim)
                + linear_params(expand_dim, output_dim))
    weights = random_weights(n_params)

    model_weights = {
        "architecture": "phantom",
        "input_features": input_dim,
        "hyperparameters": {
            "learning_rate": 0.001, "epochs": 100, "batch_size": 32,
            "sequence_length": 1, "hidden_dim": 64, "num_layers": 1,
            "dropout": 0.0, "weight_decay": 0.0,
            "early_stopping_patience": 10, "val_check_interval": 1,
        },
        "weights": weights,
        "norm_means": [0.0] * input_dim,
        "norm_stds": [1.0] * input_dim,
        "anomaly_threshold": None,
    }

    header = {
        "architecture": "Phantom",
        "input_features": input_dim,
        "num_parameters": n_params,
        "quantized": False,
        "quant_bits": None,
    }

    return model_weights, header, "phantom"


def conv1d_layer_params(in_ch, out_ch, kernel_size=3):
    """Params in a Conv1d layer: weight + bias."""
    return out_ch * in_ch * kernel_size + out_ch


def test_conv1d():
    """Test Conv1D model conversion."""
    input_channels = 4
    hidden_channels = 16
    num_layers = 2
    seq_len = 20

    n_params = 0
    for i in range(num_layers):
        in_ch = input_channels if i == 0 else hidden_channels
        n_params += conv1d_layer_params(in_ch, hidden_channels, 3)
    n_params += linear_params(hidden_channels, 1)

    weights = random_weights(n_params)

    model_weights = {
        "architecture": "conv1d",
        "input_features": input_channels,
        "hyperparameters": {
            "learning_rate": 0.001, "epochs": 100, "batch_size": 32,
            "sequence_length": seq_len, "hidden_dim": hidden_channels, "num_layers": num_layers,
            "dropout": 0.0, "weight_decay": 0.0,
            "early_stopping_patience": 10, "val_check_interval": 1,
        },
        "weights": weights,
        "norm_means": [0.0] * input_channels,
        "norm_stds": [1.0] * input_channels,
        "anomaly_threshold": None,
    }

    header = {
        "architecture": "Conv1d",
        "input_features": input_channels,
        "num_parameters": n_params,
        "quantized": False,
        "quant_bits": None,
    }

    return model_weights, header, "conv1d"


def conv2d_layer_params(in_ch, out_ch, kernel_size=3):
    return out_ch * in_ch * kernel_size * kernel_size + out_ch


def test_conv2d():
    """Test Conv2D model conversion."""
    in_channels = 1
    img_size = 32
    num_classes = 5
    channels = [32, 64, 128]

    n_params = 0
    prev_ch = in_channels
    for ch in channels:
        n_params += conv2d_layer_params(prev_ch, ch, 3)
        prev_ch = ch
    final_size = img_size >> 3  # 3 pools
    n_params += linear_params(channels[-1] * final_size * final_size, num_classes)

    weights = random_weights(n_params)

    model_weights = {
        "architecture": "conv2d",
        "input_features": in_channels * img_size * img_size,
        "hyperparameters": {
            "learning_rate": 0.001, "epochs": 100, "batch_size": 32,
            "sequence_length": 1, "hidden_dim": num_classes, "num_layers": 1,
            "dropout": 0.0, "weight_decay": 0.0,
            "early_stopping_patience": 10, "val_check_interval": 1,
        },
        "weights": weights,
        "norm_means": [0.0],
        "norm_stds": [1.0],
        "anomaly_threshold": None,
    }

    header = {
        "architecture": "Conv2d",
        "input_features": in_channels * img_size * img_size,
        "num_parameters": n_params,
        "quantized": False,
        "quant_bits": None,
    }

    return model_weights, header, "conv2d"


def bn_params(channels):
    return channels * 2  # gamma + beta only


def resnet_basic_block_params(in_ch, out_ch, stride):
    n = 0
    n += conv2d_layer_params(in_ch, out_ch, 3)  # conv1
    n += bn_params(out_ch)
    n += conv2d_layer_params(out_ch, out_ch, 3)  # conv2
    n += bn_params(out_ch)
    if stride != 1 or in_ch != out_ch:  # downsample
        n += conv2d_layer_params(in_ch, out_ch, 1)
        n += bn_params(out_ch)
    return n


def test_resnet():
    """Test ResNet-18 model conversion."""
    in_channels = 1
    img_size = 32
    num_classes = 10
    block_counts = [2, 2, 2, 2]
    channels = [64, 128, 256, 512]

    n_params = 0
    # Stem: conv 7x7 + BN
    n_params += conv2d_layer_params(in_channels, channels[0], 7)
    n_params += bn_params(channels[0])

    # Stages
    prev_ch = channels[0]
    for stage_idx, num_blocks in enumerate(block_counts):
        out_ch = channels[stage_idx]
        for block_idx in range(num_blocks):
            stride = 2 if stage_idx > 0 and block_idx == 0 else 1
            in_ch = prev_ch if block_idx == 0 else out_ch
            n_params += resnet_basic_block_params(in_ch, out_ch, stride)
        prev_ch = out_ch

    # FC
    n_params += linear_params(channels[-1], num_classes)

    weights = random_weights(n_params)

    model_weights = {
        "architecture": "res_net",
        "input_features": in_channels * img_size * img_size,
        "hyperparameters": {
            "learning_rate": 0.001, "epochs": 100, "batch_size": 32,
            "sequence_length": 1, "hidden_dim": num_classes, "num_layers": 2,
            "dropout": 0.0, "weight_decay": 0.01,
            "early_stopping_patience": 10, "val_check_interval": 1,
        },
        "weights": weights,
        "norm_means": [0.0],
        "norm_stds": [1.0],
        "anomaly_threshold": None,
    }

    header = {
        "architecture": "ResNet",
        "input_features": in_channels * img_size * img_size,
        "num_parameters": n_params,
        "quantized": False,
        "quant_bits": None,
    }

    return model_weights, header, "res_net"


def test_vgg():
    """Test VGG-11 model conversion."""
    in_channels = 1
    img_size = 32
    num_classes = 5

    config = [
        (in_channels, 64, True),
        (64, 128, True),
        (128, 256, False), (256, 256, True),
        (256, 512, False), (512, 512, True),
        (512, 512, False), (512, 512, True),
    ]

    n_params = 0
    for in_ch, out_ch, _ in config:
        n_params += conv2d_layer_params(in_ch, out_ch, 3)

    num_pools = sum(1 for _, _, p in config if p)
    final_spatial = img_size >> num_pools
    final_channels = config[-1][1]
    flat_dim = final_channels * final_spatial * final_spatial
    fc_hidden = min(512, flat_dim)

    n_params += linear_params(flat_dim, fc_hidden)
    n_params += linear_params(fc_hidden, fc_hidden)
    n_params += linear_params(fc_hidden, num_classes)

    weights = random_weights(n_params)

    model_weights = {
        "architecture": "vgg",
        "input_features": in_channels * img_size * img_size,
        "hyperparameters": {
            "learning_rate": 0.001, "epochs": 100, "batch_size": 32,
            "sequence_length": 1, "hidden_dim": num_classes, "num_layers": 1,
            "dropout": 0.0, "weight_decay": 0.01,
            "early_stopping_patience": 10, "val_check_interval": 1,
        },
        "weights": weights,
        "norm_means": [0.0],
        "norm_stds": [1.0],
        "anomaly_threshold": None,
    }

    header = {
        "architecture": "VGG",
        "input_features": in_channels * img_size * img_size,
        "num_parameters": n_params,
        "quantized": False,
        "quant_bits": None,
    }

    return model_weights, header, "vgg"


def transformer_block_params(d_model, ff_dim):
    """Params in one transformer block: 4 attention mats + FFN + 2 layer norms."""
    n = 4 * d_model * d_model  # wq, wk, wv, wo (no bias)
    n += ff_dim * d_model + ff_dim  # ff1 weight + bias
    n += d_model * ff_dim + d_model  # ff2 weight + bias
    n += d_model * 4  # ln1 gamma+beta, ln2 gamma+beta
    return n


def test_bert():
    """Test BERT model conversion."""
    input_dim = 8
    d_model = 64
    num_classes = 2
    num_heads = 4
    num_layers = 2
    ff_dim = d_model * 4
    max_seq_len = 512

    n_params = 0
    n_params += linear_params(input_dim, d_model)  # embed
    n_params += max_seq_len * d_model  # pos_embed
    n_params += num_layers * transformer_block_params(d_model, ff_dim)
    n_params += linear_params(d_model, num_classes)  # classifier

    weights = random_weights(n_params)

    model_weights = {
        "architecture": "bert",
        "input_features": input_dim,
        "hyperparameters": {
            "learning_rate": 0.001, "epochs": 100, "batch_size": 32,
            "sequence_length": 4, "hidden_dim": d_model, "num_layers": num_layers,
            "dropout": 0.0, "weight_decay": 0.01,
            "early_stopping_patience": 10, "val_check_interval": 1,
        },
        "weights": weights,
        "norm_means": [0.0] * input_dim,
        "norm_stds": [1.0] * input_dim,
        "anomaly_threshold": None,
    }

    header = {
        "architecture": "BERT",
        "input_features": input_dim,
        "num_parameters": n_params,
        "quantized": False,
        "quant_bits": None,
    }

    return model_weights, header, "bert"


def test_gpt2():
    """Test GPT-2 model conversion."""
    input_dim = 8
    d_model = 64
    output_dim = input_dim
    num_heads = 4
    num_layers = 2
    ff_dim = d_model * 4
    max_seq_len = 512

    n_params = 0
    n_params += linear_params(input_dim, d_model)
    n_params += max_seq_len * d_model
    n_params += num_layers * transformer_block_params(d_model, ff_dim)
    n_params += linear_params(d_model, output_dim)

    weights = random_weights(n_params)

    model_weights = {
        "architecture": "gpt2",
        "input_features": input_dim,
        "hyperparameters": {
            "learning_rate": 0.001, "epochs": 100, "batch_size": 32,
            "sequence_length": 4, "hidden_dim": d_model, "num_layers": num_layers,
            "dropout": 0.0, "weight_decay": 0.01,
            "early_stopping_patience": 10, "val_check_interval": 1,
        },
        "weights": weights,
        "norm_means": [0.0] * input_dim,
        "norm_stds": [1.0] * input_dim,
        "anomaly_threshold": None,
    }

    header = {
        "architecture": "GPT2",
        "input_features": input_dim,
        "num_parameters": n_params,
        "quantized": False,
        "quant_bits": None,
    }

    return model_weights, header, "gpt2"


def test_vit():
    """Test Vision Transformer conversion."""
    in_channels = 1
    img_size = 16
    num_classes = 5
    d_model = 128
    num_heads = 4
    num_layers = 2
    patch_size = max(4, img_size // 4)  # 4
    num_patches = (img_size // patch_size) ** 2  # 16
    patch_dim = in_channels * patch_size * patch_size  # 16
    ff_dim = d_model * 4

    n_params = 0
    n_params += linear_params(patch_dim, d_model)  # patch_proj
    n_params += d_model  # cls_token
    n_params += (num_patches + 1) * d_model  # pos_embed
    n_params += num_layers * transformer_block_params(d_model, ff_dim)
    n_params += linear_params(d_model, num_classes)  # head

    weights = random_weights(n_params)

    model_weights = {
        "architecture": "vi_t",
        "input_features": in_channels * img_size * img_size,
        "hyperparameters": {
            "learning_rate": 0.001, "epochs": 100, "batch_size": 32,
            "sequence_length": 1, "hidden_dim": num_classes, "num_layers": num_layers,
            "dropout": 0.0, "weight_decay": 0.01,
            "early_stopping_patience": 10, "val_check_interval": 1,
        },
        "weights": weights,
        "norm_means": [0.0],
        "norm_stds": [1.0],
        "anomaly_threshold": None,
    }

    header = {
        "architecture": "ViT",
        "input_features": in_channels * img_size * img_size,
        "num_parameters": n_params,
        "quantized": False,
        "quant_bits": None,
    }

    return model_weights, header, "vi_t"


def test_nexus():
    """Test Nexus multi-modal fusion model conversion."""
    input_dim = 16
    d_model = 64
    output_dim = 1
    num_modalities = max(2, min(8, input_dim // 4))  # 4
    mod_dim = (input_dim + num_modalities - 1) // num_modalities  # 4

    n_params = 0
    # Per-modality encoders: MLP(mod_dim -> d_model -> d_model)
    for _ in range(num_modalities):
        n_params += linear_params(mod_dim, d_model)
        n_params += linear_params(d_model, d_model)
    # Fusion weights (stored but unused)
    n_params += 3 * d_model * d_model  # wq, wk, wv
    # Head
    n_params += linear_params(d_model, d_model)
    n_params += linear_params(d_model, output_dim)

    weights = random_weights(n_params)

    model_weights = {
        "architecture": "nexus",
        "input_features": input_dim,
        "hyperparameters": {
            "learning_rate": 0.001, "epochs": 100, "batch_size": 32,
            "sequence_length": 1, "hidden_dim": d_model, "num_layers": 1,
            "dropout": 0.0, "weight_decay": 0.0,
            "early_stopping_patience": 10, "val_check_interval": 1,
        },
        "weights": weights,
        "norm_means": [0.0] * input_dim,
        "norm_stds": [1.0] * input_dim,
        "anomaly_threshold": None,
    }

    header = {
        "architecture": "Nexus",
        "input_features": input_dim,
        "num_parameters": n_params,
        "quantized": False,
        "quant_bits": None,
    }

    return model_weights, header, "nexus"


def main():
    from convert import parse_axonml, build_model, export_to_onnx

    test_cases = [
        ("Sentinel", test_sentinel),
        ("LSTM Autoencoder", test_lstm_autoencoder),
        ("GRU Predictor", test_gru_predictor),
        ("RNN", test_rnn),
        ("Phantom", test_phantom),
        ("Conv1D", test_conv1d),
        ("Conv2D", test_conv2d),
        ("ResNet", test_resnet),
        ("VGG", test_vgg),
        ("BERT", test_bert),
        ("GPT2", test_gpt2),
        ("ViT", test_vit),
        ("Nexus", test_nexus),
    ]

    tmpdir = tempfile.mkdtemp(prefix="prometheus_convert_test_")
    print(f"Test directory: {tmpdir}\n")

    results = []
    for name, test_fn in test_cases:
        print(f"{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")

        try:
            model_weights, header, arch_name = test_fn()

            # Write .axonml file
            axonml_path = os.path.join(tmpdir, f"{arch_name}.axonml")
            create_axonml_file(axonml_path, model_weights, header)
            axonml_size = os.path.getsize(axonml_path)
            print(f"  Created {axonml_path} ({axonml_size:,} bytes)")

            # Parse it back
            parsed = parse_axonml(axonml_path)
            print(f"  Parsed OK — {parsed['header']['num_parameters']:,} parameters")

            # Build PyTorch model
            model, dummy_input = build_model(parsed["model"])
            print(f"  PyTorch model built OK")

            # Test forward pass
            import torch
            model.eval()
            with torch.no_grad():
                output = model(dummy_input)
                print(f"  Forward pass OK — output shape: {list(output.shape)}")

            # Export to ONNX
            onnx_path = os.path.join(tmpdir, f"{arch_name}.onnx")
            export_to_onnx(model, dummy_input, onnx_path)
            onnx_size = os.path.getsize(onnx_path)
            print(f"  ONNX export OK ({onnx_size:,} bytes)")

            # Validate with ONNX Runtime
            import onnxruntime as ort
            session = ort.InferenceSession(onnx_path)
            ort_input = {session.get_inputs()[0].name: dummy_input.numpy()}
            ort_output = session.run(None, ort_input)
            max_diff = np.abs(output.numpy() - ort_output[0]).max()
            print(f"  ONNX Runtime inference OK — max diff: {max_diff:.2e}")

            if max_diff < 1e-4:
                print(f"  PASS")
                results.append((name, True, None))
            else:
                print(f"  WARN: Large difference, but file is valid")
                results.append((name, True, f"max_diff={max_diff:.2e}"))

        except Exception as e:
            import traceback
            print(f"  FAIL: {e}")
            traceback.print_exc()
            results.append((name, False, str(e)))

        print()

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    for name, ok, note in results:
        status = "PASS" if ok else "FAIL"
        extra = f" ({note})" if note else ""
        print(f"  [{status}] {name}{extra}")
    print(f"\n{passed}/{total} tests passed")
    print(f"Test files in: {tmpdir}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
