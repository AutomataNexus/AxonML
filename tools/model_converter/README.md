# AxonML Model Converter

Converts AxonML `.axonml` bundles to **ONNX**.

## Why a Python tool?

AxonML is pure Rust, but ONNX export is hard to get right in Rust alone — every layer type needs its own node-graph emitter, dynamic-axis handling, opset versioning, and validation against the reference spec.

Rather than reinvent that, the converter:

1. Reads the `.axonml` bundle (architecture tag + hyperparameters + flat weight vector — produced by [`axonml_serialize::save_bundle`](../../crates/axonml-serialize/src/bundle.rs)).
2. Reconstructs the model in PyTorch layer-for-layer.
3. Loads the flat weight vector into the PyTorch `state_dict`.
4. Runs `torch.onnx.export()` with `opset_version=17` and `do_constant_folding=True`.
5. Validates the resulting ONNX via `onnx.checker` and an ONNX Runtime parity check (PyTorch output vs ORT output, max-diff reported).

The result is a production-grade ONNX file that TensorRT, ONNX Runtime, and `onnxruntime-web` all accept without complaint.

It lets any trained model round-trip through ONNX.

## Supported architectures

The `.axonml` bundle's `architecture` tag must be one of:

| Tag                 | PyTorch reconstruction                                      |
|---------------------|-------------------------------------------------------------|
| `sentinel`          | 3-layer MLP (784/128/64/1, Sigmoid)                        |
| `lstm_autoencoder`  | Encoder LSTM → bottleneck → Decoder LSTM → reconstruction  |
| `gru_predictor`     | GRU → FC → FC → Sigmoid (multi-horizon prediction)         |
| `rnn`               | Vanilla stacked RNN → FC → Sigmoid                         |
| `bottleneck-mlp`    | 3-layer bottleneck MLP with ReLU6                          |
| `conv1d`            | Stacked Conv1d → GAP → FC → Sigmoid                        |
| `conv2d`            | 3× (Conv2d → ReLU → MaxPool) → FC → Softmax                |
| `res_net`           | ResNet-18 (stem + 4 BasicBlock stages + GAP + FC)          |
| `vgg`               | VGG-11 (8 conv + 3 FC + Softmax)                           |
| `bert`              | Post-norm transformer encoder + CLS classifier              |
| `gpt2`              | Causal transformer + LM head                               |
| `vi_t`              | Patch-embed + CLS + pre-norm transformer + classifier       |
| `nexus`             | Per-modality encoders + cross-modal attention + FC head     |

Unsupported architectures fall through to a clear error message listing the supported set.

## Installation

```bash
# One-time setup
python3 -m venv /opt/AxonML/tools/converter-venv
/opt/AxonML/tools/converter-venv/bin/pip install -r /opt/AxonML/tools/model_converter/requirements.txt

# Tell the AxonML CLI where to find Python
export AXONML_CONVERTER_PYTHON=/opt/AxonML/tools/converter-venv/bin/python
```

Add the export to your shell rc if you want it to persist.

## Usage

Via the AxonML CLI (preferred):

```bash
axonml export my_model.axonml --format onnx --output my_model.onnx
```

Or directly:

```bash
/opt/AxonML/tools/converter-venv/bin/python \
    /opt/AxonML/tools/model_converter/convert.py \
    my_model.axonml --format onnx --output my_model.onnx
```

Validate-only (parses the bundle, doesn't convert):

```bash
python convert.py my_model.axonml --validate-only
```

## Environment variables

| Variable                   | Purpose                                                          |
|----------------------------|------------------------------------------------------------------|
| `AXONML_CONVERTER_PYTHON`  | Path to the Python interpreter with torch+onnx installed.        |
| `AXONML_CONVERTER_SCRIPT`  | Override path to `convert.py` (default: this file).              |
| `CONVERTER_VENV`           | Alternative: a venv root — `<root>/bin/python` is used.          |

## Testing

```bash
/opt/AxonML/tools/converter-venv/bin/python -m pytest test_convert.py
```
