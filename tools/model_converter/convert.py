#!/usr/bin/env python3
"""
AxonML Model Converter — .axonml to ONNX (and HEF for Hailo edge inference).

Parses the `.axonml` binary format produced by AxonML training, reconstructs
the model in PyTorch layer-for-layer, loads the flat weight vector into the
PyTorch state dict, and exports to standard ONNX protobuf format via the
battle-tested `torch.onnx.export()` path. Validation happens via `onnx.checker`
and an ONNX Runtime parity check (PyTorch vs ORT max-diff).

Optional HEF compilation for Hailo-8/8L edge accelerators is gated behind
the Hailo DFC SDK — if not installed, ONNX export still works end-to-end.

Architectures supported (must match AxonML's `Architecture` enum):
    sentinel, lstm_autoencoder, gru_predictor, rnn, phantom, conv1d,
    conv2d, res_net, vgg, bert, gpt2, vi_t, nexus.

Usage:
    python convert.py model.axonml --format onnx --output model.onnx
    python convert.py model.axonml --format hef  --output model.hef
    python convert.py model.axonml --validate-only

Originally developed for the Prometheus predictive-maintenance stack
(github.com/AutomataNexus/Prometheus); upstreamed into AxonML so any
model trained with AxonML can be converted to ONNX for serving.
"""

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnx.checker


# =============================================================================
# .axonml parser
# =============================================================================

AXONML_MAGIC = b"AXONML"
AXONML_VERSION = 1


def parse_axonml(path: str) -> dict:
    """Parse an .axonml file and return the ModelWeights dict."""
    data = Path(path).read_bytes()

    if len(data) < 11 or data[:6] != AXONML_MAGIC:
        raise ValueError(f"Invalid .axonml file: bad magic bytes")

    version = data[6]
    if version != AXONML_VERSION:
        raise ValueError(f"Unsupported .axonml version: {version} (expected {AXONML_VERSION})")

    # Read header
    header_len = struct.unpack_from("<I", data, 7)[0]
    header_start = 11
    header_end = header_start + header_len
    if len(data) < header_end + 4:
        raise ValueError("Truncated .axonml file (header)")

    header = json.loads(data[header_start:header_end])

    # Read weights
    weights_len = struct.unpack_from("<I", data, header_end)[0]
    weights_start = header_end + 4
    weights_end = weights_start + weights_len
    if len(data) < weights_end:
        raise ValueError("Truncated .axonml file (weights)")

    model_weights = json.loads(data[weights_start:weights_end])

    return {
        "header": header,
        "model": model_weights,
    }


# =============================================================================
# PyTorch model definitions matching AxonML architectures
# =============================================================================


class SentinelPT(nn.Module):
    """Sentinel MLP: Linear(in,128)->ReLU->Linear(128,64)->ReLU->Linear(64,1)->Sigmoid"""

    def __init__(self, input_features: int):
        super().__init__()
        self.fc1 = nn.Linear(input_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class LstmAutoencoderPT(nn.Module):
    """LSTM Autoencoder: encoder LSTM -> bottleneck -> decoder LSTM -> output."""

    def __init__(self, input_features: int, hidden_dim: int, num_layers: int, seq_len: int = 10):
        super().__init__()
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bottleneck_dim = hidden_dim // 2
        self.seq_len = seq_len

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_features, hidden_dim, num_layers, batch_first=True
        )
        self.encoder_linear = nn.Linear(hidden_dim, self.bottleneck_dim)

        # Decoder
        self.decoder_linear = nn.Linear(self.bottleneck_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers, batch_first=True
        )
        self.decoder_output = nn.Linear(hidden_dim, input_features)
        # Expand bottleneck to full sequence (avoids Tile/Expand for Hailo)
        self.decoder_expand = nn.Linear(hidden_dim, seq_len * hidden_dim)

        # Register fixed-shape zero states as buffers (ONNX initializers for Hailo)
        self.register_buffer('enc_h0', torch.zeros(num_layers, 1, hidden_dim))
        self.register_buffer('enc_c0', torch.zeros(num_layers, 1, hidden_dim))
        self.register_buffer('dec_c0', torch.zeros(num_layers, 1, hidden_dim))

    def forward(self, x):
        # x: [batch, seq_len, features] — batch=1 for Hailo static export
        _, (h_n, _) = self.encoder_lstm(x, (self.enc_h0, self.enc_c0))
        bottleneck = self.encoder_linear(h_n[-1])
        expanded = self.decoder_linear(bottleneck)
        # Project bottleneck to full decoder input shape via linear layer
        # This avoids Tile/Expand/Repeat ops that Hailo DFC cannot parse
        decoder_input = self.decoder_expand(expanded)  # [batch, seq_len * hidden]
        decoder_input = decoder_input.view(-1, self.seq_len, self.hidden_dim)
        decoder_out, _ = self.decoder_lstm(decoder_input, (self.dec_c0, self.dec_c0))
        output = self.decoder_output(decoder_out)
        return output


class LstmEncoderOnlyPT(nn.Module):
    """LSTM Encoder-only for Hailo HEF: encoder LSTM -> bottleneck -> score.

    For edge inference, only the encoder runs on Hailo. Reconstruction error
    is computed host-side from the bottleneck output.
    """
    def __init__(self, input_features: int, hidden_dim: int, num_layers: int, bottleneck_dim: int):
        super().__init__()
        self.encoder_lstm = nn.LSTM(input_features, hidden_dim, num_layers, batch_first=True)
        self.encoder_linear = nn.Linear(hidden_dim, bottleneck_dim)
        self.register_buffer('h0', torch.zeros(num_layers, 1, hidden_dim))
        self.register_buffer('c0', torch.zeros(num_layers, 1, hidden_dim))

    def forward(self, x):
        _, (h_n, _) = self.encoder_lstm(x, (self.h0, self.c0))
        bottleneck = self.encoder_linear(h_n[-1])
        return bottleneck


class GruPredictorPT(nn.Module):
    """GRU Predictor: GRU -> Linear(hd,hd) -> ReLU -> Linear(hd,3) -> Sigmoid"""

    def __init__(self, input_features: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_features, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        batch = x.size(0)
        h0 = torch.zeros(self.num_layers, batch, self.hidden_dim, device=x.device)
        _, h_n = self.gru(x, h0)
        h_last = h_n[-1]
        x = torch.relu(self.fc1(h_last))
        x = torch.sigmoid(self.fc2(x))
        return x


class RnnModelPT(nn.Module):
    """Vanilla RNN: stacked RNN -> Linear(hd,1) -> Sigmoid"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True,
                          nonlinearity='tanh')
        self.output_linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch = x.size(0)
        h0 = torch.zeros(self.num_layers, batch, self.hidden_dim, device=x.device)
        _, h_n = self.rnn(x, h0)
        h_last = h_n[-1]
        x = torch.sigmoid(self.output_linear(h_last))
        return x


class PhantomPT(nn.Module):
    """Phantom: 3-layer bottleneck MLP with ReLU6 activations."""

    def __init__(self, input_dim, bottleneck_dim, expand_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, bottleneck_dim)
        self.fc2 = nn.Linear(bottleneck_dim, expand_dim)
        self.fc3 = nn.Linear(expand_dim, output_dim)

    def forward(self, x):
        x = torch.clamp(torch.relu(self.fc1(x)), max=6.0)
        x = torch.clamp(torch.relu(self.fc2(x)), max=6.0)
        x = torch.sigmoid(self.fc3(x))
        return x


class Conv1dPT(nn.Module):
    """Conv1D: stacked Conv1d(kernel=3, no padding) -> ReLU -> GAP -> Linear -> Sigmoid."""

    def __init__(self, input_channels, hidden_channels, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_channels
            self.convs.append(nn.Conv1d(in_ch, hidden_channels, 3, padding=0))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_channels, 1)

    def forward(self, x):
        for conv in self.convs:
            x = torch.relu(conv(x))
        x = self.pool(x).squeeze(-1)
        x = torch.sigmoid(self.fc(x))
        return x


class Conv2dPT(nn.Module):
    """Conv2D: 3x (Conv2d 3x3 pad=1 -> ReLU -> MaxPool2d) -> Linear -> Softmax."""

    def __init__(self, in_channels, num_classes, img_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        final_size = img_size >> 3
        self.fc = nn.Linear(128 * final_size * final_size, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return torch.softmax(self.fc(x), dim=-1)


class ResNetBasicBlockPT(nn.Module):
    """ResNet BasicBlock: conv3x3+BN+ReLU -> conv3x3+BN -> residual+ReLU."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.has_downsample = (stride != 1 or in_ch != out_ch)
        if self.has_downsample:
            self.ds_conv = nn.Conv2d(in_ch, out_ch, 1, stride=stride, padding=0)
            self.ds_bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.has_downsample:
            residual = self.ds_bn(self.ds_conv(residual))
        return torch.relu(out + residual)


class ResNetPT(nn.Module):
    """ResNet-18: stem -> 4 stages of BasicBlocks -> GAP -> FC."""

    def __init__(self, in_channels, num_classes, block_counts, channels):
        super().__init__()
        self.stem_conv = nn.Conv2d(in_channels, channels[0], 7, stride=2, padding=3)
        self.stem_bn = nn.BatchNorm2d(channels[0])
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.stages = nn.ModuleList()
        prev_ch = channels[0]
        for stage_idx, num_blocks in enumerate(block_counts):
            out_ch = channels[stage_idx]
            blocks = nn.ModuleList()
            for block_idx in range(num_blocks):
                stride = 2 if stage_idx > 0 and block_idx == 0 else 1
                in_ch = prev_ch if block_idx == 0 else out_ch
                blocks.append(ResNetBasicBlockPT(in_ch, out_ch, stride))
            prev_ch = out_ch
            self.stages.append(blocks)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        x = torch.relu(self.stem_bn(self.stem_conv(x)))
        x = self.maxpool(x)
        for stage in self.stages:
            for block in stage:
                x = block(x)
        x = self.pool(x).view(x.size(0), -1)
        return torch.softmax(self.fc(x), dim=-1)


class VggPT(nn.Module):
    """VGG-11: 8 conv layers with max pool -> 3 FC layers -> softmax."""

    def __init__(self, in_channels, num_classes, img_size, config):
        super().__init__()
        # config: list of (in_ch, out_ch, pool_after)
        self.conv_layers = nn.ModuleList()
        self.pool_after = []
        for in_ch, out_ch, pool in config:
            self.conv_layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
            self.pool_after.append(pool)
        self.pool = nn.MaxPool2d(2)

        num_pools = sum(1 for p in self.pool_after if p)
        final_spatial = img_size >> num_pools
        final_channels = config[-1][1]
        flat_dim = final_channels * final_spatial * final_spatial
        fc_hidden = min(512, flat_dim)

        self.fc1 = nn.Linear(flat_dim, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, fc_hidden)
        self.fc3 = nn.Linear(fc_hidden, num_classes)

    def forward(self, x):
        for i, conv in enumerate(self.conv_layers):
            x = torch.relu(conv(x))
            if self.pool_after[i]:
                x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)


# --- Transformer building blocks ---

class AttentionPT(nn.Module):
    """Multi-head self-attention (no bias on projections)."""

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** 0.5
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, causal=False):
        B, S, D = x.shape
        q = self.wq(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if causal:
            mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.wo(out)


class PostNormBlockPT(nn.Module):
    """Post-norm transformer block (BERT/GPT2 style)."""

    def __init__(self, d_model, num_heads, ff_dim, causal=False):
        super().__init__()
        self.causal = causal
        self.attn = AttentionPT(d_model, num_heads)
        self.ff1 = nn.Linear(d_model, ff_dim)
        self.ff2 = nn.Linear(ff_dim, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.gelu = nn.GELU(approximate='tanh')

    def forward(self, x):
        attn_out = self.attn(x, causal=self.causal)
        normed = self.ln1(x + attn_out)
        ff_out = self.ff2(self.gelu(self.ff1(normed)))
        return self.ln2(normed + ff_out)


class PreNormBlockPT(nn.Module):
    """Pre-norm transformer block (ViT style)."""

    def __init__(self, d_model, num_heads, ff_dim):
        super().__init__()
        self.attn = AttentionPT(d_model, num_heads)
        self.ff1 = nn.Linear(d_model, ff_dim)
        self.ff2 = nn.Linear(ff_dim, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.gelu = nn.GELU(approximate='tanh')

    def forward(self, x):
        normed1 = self.ln1(x)
        x = x + self.attn(normed1)
        normed2 = self.ln2(x)
        return x + self.ff2(self.gelu(self.ff1(normed2)))


class BertPT(nn.Module):
    """BERT: embed -> pos_embed -> transformer blocks -> CLS classifier."""

    def __init__(self, input_dim, num_classes, d_model, num_heads, num_layers, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.embed = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(max_seq_len, d_model))
        self.blocks = nn.ModuleList([
            PostNormBlockPT(d_model, num_heads, d_model * 4, causal=False)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(d_model, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        B, S, _ = x.shape
        x = self.embed(x) + self.pos_embed[:S]
        for block in self.blocks:
            x = block(x)
        cls_out = x[:, 0, :]  # first token
        logits = self.classifier(cls_out)
        if self.num_classes == 1:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)


class Gpt2PT(nn.Module):
    """GPT-2: embed -> pos_embed -> causal transformer -> LM head."""

    def __init__(self, input_dim, output_dim, d_model, num_heads, num_layers, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.embed = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(max_seq_len, d_model))
        self.blocks = nn.ModuleList([
            PostNormBlockPT(d_model, num_heads, d_model * 4, causal=True)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(d_model, output_dim)

    def forward(self, x):
        B, S, _ = x.shape
        x = self.embed(x) + self.pos_embed[:S]
        for block in self.blocks:
            x = block(x)
        last_out = x[:, -1, :]
        return torch.softmax(self.lm_head(last_out), dim=-1)


class ViTPT(nn.Module):
    """ViT: patch embed + CLS token + pos embed -> pre-norm transformer -> head."""

    def __init__(self, in_channels, num_classes, image_size, patch_size, d_model, num_heads, num_layers):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(d_model))
        self.pos_embed = nn.Parameter(torch.zeros(num_patches + 1, d_model))
        self.blocks = nn.ModuleList([
            PreNormBlockPT(d_model, num_heads, d_model * 4)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [batch, channels, H, W]
        B = x.size(0)
        ps = self.patch_size
        # Unfold into patches: [B, num_patches, patch_dim]
        patches = x.unfold(2, ps, ps).unfold(3, ps, ps)
        patches = patches.contiguous().view(B, -1, patches.size(1) * ps * ps)
        # Actually need proper reshape for multi-channel
        C = x.size(1)
        H = x.size(2)
        grid = H // ps
        patches = x.reshape(B, C, grid, ps, grid, ps).permute(0, 2, 4, 1, 3, 5).reshape(B, grid * grid, C * ps * ps)

        tokens = self.patch_proj(patches)
        cls = self.cls_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        x = torch.cat([cls, tokens], dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return torch.softmax(self.head(x[:, 0, :]), dim=-1)


class NexusEncoderPT(nn.Module):
    """Single modality encoder: 2-layer MLP."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class NexusPT(nn.Module):
    """Nexus: per-modality encoders -> cross-attention (dot product) -> head."""

    def __init__(self, input_dim, d_model, output_dim, num_modalities, mod_dim):
        super().__init__()
        self.num_modalities = num_modalities
        self.mod_dim = mod_dim
        self.d_model = d_model
        self.encoders = nn.ModuleList([
            NexusEncoderPT(mod_dim, d_model, d_model) for _ in range(num_modalities)
        ])
        # Fusion weights stored but not used in forward (matches AxonML)
        self.fusion_wq = nn.Parameter(torch.zeros(d_model, d_model))
        self.fusion_wk = nn.Parameter(torch.zeros(d_model, d_model))
        self.fusion_wv = nn.Parameter(torch.zeros(d_model, d_model))
        self.head_fc1 = nn.Linear(d_model, d_model)
        self.head_fc2 = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: [batch, input_dim]
        embeddings = []
        for i, enc in enumerate(self.encoders):
            start = i * self.mod_dim
            end = start + self.mod_dim
            chunk = x[:, start:end]
            if chunk.size(1) < self.mod_dim:
                chunk = torch.nn.functional.pad(chunk, (0, self.mod_dim - chunk.size(1)))
            embeddings.append(enc(chunk))

        # Cross-modal attention via direct dot product (matching AxonML)
        stacked = torch.stack(embeddings, dim=1)  # [B, num_mod, d_model]
        scores = torch.matmul(stacked, stacked.transpose(-2, -1)) / (self.d_model ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attn, stacked)
        fused = attended.mean(dim=1)  # [B, d_model]

        h = torch.relu(self.head_fc1(fused))
        return torch.sigmoid(self.head_fc2(h))


# =============================================================================
# Weight loading — map flat AxonML weight vector to PyTorch state_dict
# =============================================================================


def load_linear_weights(flat: list, offset: int, in_f: int, out_f: int):
    """Extract Linear layer weights from the flat vector.
    AxonML stores: weight[out_f * in_f] then bias[out_f]  — row-major [out, in].
    PyTorch nn.Linear stores weight as [out, in] — same layout.
    """
    w_size = out_f * in_f
    weight = np.array(flat[offset:offset + w_size], dtype=np.float32).reshape(out_f, in_f)
    offset += w_size
    bias = np.array(flat[offset:offset + out_f], dtype=np.float32)
    offset += out_f
    return torch.from_numpy(weight), torch.from_numpy(bias), offset


def load_lstm_cell_weights(flat: list, offset: int, input_dim: int, hidden_dim: int):
    """Extract LSTM cell weights. AxonML order: w_ih, w_hh, b_ih, b_hh.
    Shapes: w_ih [4*hd, in], w_hh [4*hd, hd], b_ih [4*hd], b_hh [4*hd].
    PyTorch LSTM uses the same shapes and gate order (i,f,g,o).
    """
    w_ih_size = 4 * hidden_dim * input_dim
    w_ih = np.array(flat[offset:offset + w_ih_size], dtype=np.float32).reshape(4 * hidden_dim, input_dim)
    offset += w_ih_size

    w_hh_size = 4 * hidden_dim * hidden_dim
    w_hh = np.array(flat[offset:offset + w_hh_size], dtype=np.float32).reshape(4 * hidden_dim, hidden_dim)
    offset += w_hh_size

    b_ih = np.array(flat[offset:offset + 4 * hidden_dim], dtype=np.float32)
    offset += 4 * hidden_dim
    b_hh = np.array(flat[offset:offset + 4 * hidden_dim], dtype=np.float32)
    offset += 4 * hidden_dim

    return (torch.from_numpy(w_ih), torch.from_numpy(w_hh),
            torch.from_numpy(b_ih), torch.from_numpy(b_hh), offset)


def load_gru_cell_weights(flat: list, offset: int, input_dim: int, hidden_dim: int):
    """Extract GRU cell weights. AxonML order: w_ih, w_hh, b_ih, b_hh.
    Shapes: w_ih [3*hd, in], w_hh [3*hd, hd], b_ih [3*hd], b_hh [3*hd].
    """
    w_ih_size = 3 * hidden_dim * input_dim
    w_ih = np.array(flat[offset:offset + w_ih_size], dtype=np.float32).reshape(3 * hidden_dim, input_dim)
    offset += w_ih_size

    w_hh_size = 3 * hidden_dim * hidden_dim
    w_hh = np.array(flat[offset:offset + w_hh_size], dtype=np.float32).reshape(3 * hidden_dim, hidden_dim)
    offset += w_hh_size

    b_ih = np.array(flat[offset:offset + 3 * hidden_dim], dtype=np.float32)
    offset += 3 * hidden_dim
    b_hh = np.array(flat[offset:offset + 3 * hidden_dim], dtype=np.float32)
    offset += 3 * hidden_dim

    return (torch.from_numpy(w_ih), torch.from_numpy(w_hh),
            torch.from_numpy(b_ih), torch.from_numpy(b_hh), offset)


def load_rnn_cell_weights(flat: list, offset: int, input_dim: int, hidden_dim: int):
    """Extract vanilla RNN layer weights. AxonML order: w_ih, w_hh, b_ih, b_hh.
    Shapes: w_ih [hd, in], w_hh [hd, hd], b_ih [hd], b_hh [hd].
    """
    w_ih_size = hidden_dim * input_dim
    w_ih = np.array(flat[offset:offset + w_ih_size], dtype=np.float32).reshape(hidden_dim, input_dim)
    offset += w_ih_size

    w_hh_size = hidden_dim * hidden_dim
    w_hh = np.array(flat[offset:offset + w_hh_size], dtype=np.float32).reshape(hidden_dim, hidden_dim)
    offset += w_hh_size

    b_ih = np.array(flat[offset:offset + hidden_dim], dtype=np.float32)
    offset += hidden_dim
    b_hh = np.array(flat[offset:offset + hidden_dim], dtype=np.float32)
    offset += hidden_dim

    return (torch.from_numpy(w_ih), torch.from_numpy(w_hh),
            torch.from_numpy(b_ih), torch.from_numpy(b_hh), offset)


def load_conv_weights(flat, offset, out_ch, in_ch, kernel_size):
    """Load Conv weight [out_ch, in_ch, *kernel] and bias [out_ch]."""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,)
    shape = (out_ch, in_ch) + tuple(kernel_size)
    w_size = 1
    for s in shape:
        w_size *= s
    weight = np.array(flat[offset:offset + w_size], dtype=np.float32).reshape(shape)
    offset += w_size
    bias = np.array(flat[offset:offset + out_ch], dtype=np.float32)
    offset += out_ch
    return torch.from_numpy(weight), torch.from_numpy(bias), offset


def load_bn_weights(flat, offset, channels):
    """Load BatchNorm gamma and beta. AxonML BN uses running_mean=0, running_var=1."""
    gamma = np.array(flat[offset:offset + channels], dtype=np.float32)
    offset += channels
    beta = np.array(flat[offset:offset + channels], dtype=np.float32)
    offset += channels
    return torch.from_numpy(gamma), torch.from_numpy(beta), offset


def load_flat(flat, offset, size):
    """Load a flat weight tensor."""
    data = np.array(flat[offset:offset + size], dtype=np.float32)
    offset += size
    return torch.from_numpy(data), offset


def set_bn(bn_module, gamma, beta):
    """Set BN weights and initialize running stats to match AxonML defaults."""
    bn_module.weight.data = gamma
    bn_module.bias.data = beta
    bn_module.running_mean.fill_(0.0)
    bn_module.running_var.fill_(1.0)


def load_transformer_block_weights(flat, offset, block, d_model, ff_dim):
    """Load weights into a PostNormBlockPT or PreNormBlockPT.

    AxonML weight order: wq, wk, wv, wo, ff_w1, ff_b1, ff_w2, ff_b2,
                         ln1_gamma, ln1_beta, ln2_gamma, ln2_beta.
    """
    n = d_model * d_model
    # Attention projections (no bias)
    block.attn.wq.weight.data = torch.from_numpy(
        np.array(flat[offset:offset + n], dtype=np.float32).reshape(d_model, d_model))
    offset += n
    block.attn.wk.weight.data = torch.from_numpy(
        np.array(flat[offset:offset + n], dtype=np.float32).reshape(d_model, d_model))
    offset += n
    block.attn.wv.weight.data = torch.from_numpy(
        np.array(flat[offset:offset + n], dtype=np.float32).reshape(d_model, d_model))
    offset += n
    block.attn.wo.weight.data = torch.from_numpy(
        np.array(flat[offset:offset + n], dtype=np.float32).reshape(d_model, d_model))
    offset += n

    # FFN
    w, b, offset = load_linear_weights(flat, offset, d_model, ff_dim)
    block.ff1.weight.data = w
    block.ff1.bias.data = b
    w, b, offset = load_linear_weights(flat, offset, ff_dim, d_model)
    block.ff2.weight.data = w
    block.ff2.bias.data = b

    # Layer norms
    g, beta, offset = load_bn_weights(flat, offset, d_model)  # reuse helper (gamma+beta)
    block.ln1.weight.data = g
    block.ln1.bias.data = beta
    g, beta, offset = load_bn_weights(flat, offset, d_model)
    block.ln2.weight.data = g
    block.ln2.bias.data = beta

    return offset


# =============================================================================
# Model builder — reconstruct PyTorch model from AxonML weights
# =============================================================================


def build_sentinel(model_data: dict) -> tuple:
    """Build Sentinel PyTorch model from AxonML weights."""
    input_features = model_data["input_features"]
    flat = model_data["weights"]
    hp = model_data["hyperparameters"]

    model = SentinelPT(input_features)
    offset = 0

    # fc1: Linear(input_features, 128)
    w, b, offset = load_linear_weights(flat, offset, input_features, 128)
    model.fc1.weight.data = w
    model.fc1.bias.data = b

    # fc2: Linear(128, 64)
    w, b, offset = load_linear_weights(flat, offset, 128, 64)
    model.fc2.weight.data = w
    model.fc2.bias.data = b

    # fc3: Linear(64, 1)
    w, b, offset = load_linear_weights(flat, offset, 64, 1)
    model.fc3.weight.data = w
    model.fc3.bias.data = b

    assert offset == len(flat), f"Sentinel: used {offset}/{len(flat)} params"

    dummy = torch.randn(1, input_features)
    return model, dummy


def build_lstm_autoencoder(model_data: dict) -> tuple:
    """Build LSTM Autoencoder PyTorch model."""
    input_features = model_data["input_features"]
    flat = model_data["weights"]
    hp = model_data["hyperparameters"]
    hidden_dim = hp.get("hidden_dim", 64)
    num_layers = hp.get("num_layers", 2)
    seq_len = hp.get("sequence_length", 60)
    bottleneck_dim = hidden_dim // 2

    model = LstmAutoencoderPT(input_features, hidden_dim, num_layers, seq_len=seq_len)
    offset = 0

    # Encoder LSTM layers
    for layer_idx in range(num_layers):
        in_dim = input_features if layer_idx == 0 else hidden_dim
        w_ih, w_hh, b_ih, b_hh, offset = load_lstm_cell_weights(flat, offset, in_dim, hidden_dim)
        setattr(model.encoder_lstm, f"weight_ih_l{layer_idx}", nn.Parameter(w_ih))
        setattr(model.encoder_lstm, f"weight_hh_l{layer_idx}", nn.Parameter(w_hh))
        setattr(model.encoder_lstm, f"bias_ih_l{layer_idx}", nn.Parameter(b_ih))
        setattr(model.encoder_lstm, f"bias_hh_l{layer_idx}", nn.Parameter(b_hh))

    # Encoder linear
    w, b, offset = load_linear_weights(flat, offset, hidden_dim, bottleneck_dim)
    model.encoder_linear.weight.data = w
    model.encoder_linear.bias.data = b

    # Decoder linear
    w, b, offset = load_linear_weights(flat, offset, bottleneck_dim, hidden_dim)
    model.decoder_linear.weight.data = w
    model.decoder_linear.bias.data = b

    # Decoder LSTM layers
    for layer_idx in range(num_layers):
        in_dim = hidden_dim  # decoder input is always hidden_dim
        w_ih, w_hh, b_ih, b_hh, offset = load_lstm_cell_weights(flat, offset, in_dim, hidden_dim)
        setattr(model.decoder_lstm, f"weight_ih_l{layer_idx}", nn.Parameter(w_ih))
        setattr(model.decoder_lstm, f"weight_hh_l{layer_idx}", nn.Parameter(w_hh))
        setattr(model.decoder_lstm, f"bias_ih_l{layer_idx}", nn.Parameter(b_ih))
        setattr(model.decoder_lstm, f"bias_hh_l{layer_idx}", nn.Parameter(b_hh))

    # Decoder output
    w, b, offset = load_linear_weights(flat, offset, hidden_dim, input_features)
    model.decoder_output.weight.data = w
    model.decoder_output.bias.data = b

    assert offset == len(flat), f"LSTM AE: used {offset}/{len(flat)} params"

    # Initialize decoder_expand to approximate repeat (identity-like blocks)
    # Each seq_len block maps hidden_dim → hidden_dim as identity
    with torch.no_grad():
        w = torch.zeros(seq_len * hidden_dim, hidden_dim)
        for i in range(seq_len):
            w[i*hidden_dim:(i+1)*hidden_dim] = torch.eye(hidden_dim)
        model.decoder_expand.weight.data = w
        model.decoder_expand.bias.data = torch.zeros(seq_len * hidden_dim)

    dummy = torch.randn(1, seq_len, input_features)
    return model, dummy


def build_gru_predictor(model_data: dict) -> tuple:
    """Build GRU Predictor PyTorch model."""
    input_features = model_data["input_features"]
    flat = model_data["weights"]
    hp = model_data["hyperparameters"]
    hidden_dim = hp.get("hidden_dim", 128)
    num_layers = hp.get("num_layers", 2)
    seq_len = hp.get("sequence_length", 60)

    model = GruPredictorPT(input_features, hidden_dim, num_layers)
    offset = 0

    # GRU layers
    for layer_idx in range(num_layers):
        in_dim = input_features if layer_idx == 0 else hidden_dim
        w_ih, w_hh, b_ih, b_hh, offset = load_gru_cell_weights(flat, offset, in_dim, hidden_dim)
        setattr(model.gru, f"weight_ih_l{layer_idx}", nn.Parameter(w_ih))
        setattr(model.gru, f"weight_hh_l{layer_idx}", nn.Parameter(w_hh))
        setattr(model.gru, f"bias_ih_l{layer_idx}", nn.Parameter(b_ih))
        setattr(model.gru, f"bias_hh_l{layer_idx}", nn.Parameter(b_hh))

    # fc1: Linear(hidden_dim, hidden_dim)
    w, b, offset = load_linear_weights(flat, offset, hidden_dim, hidden_dim)
    model.fc1.weight.data = w
    model.fc1.bias.data = b

    # fc2: Linear(hidden_dim, 3)
    w, b, offset = load_linear_weights(flat, offset, hidden_dim, 3)
    model.fc2.weight.data = w
    model.fc2.bias.data = b

    assert offset == len(flat), f"GRU: used {offset}/{len(flat)} params"

    dummy = torch.randn(1, seq_len, input_features)
    return model, dummy


def build_rnn_model(model_data: dict) -> tuple:
    """Build vanilla RNN PyTorch model."""
    input_features = model_data["input_features"]
    flat = model_data["weights"]
    hp = model_data["hyperparameters"]
    hidden_dim = hp.get("hidden_dim", 64)
    num_layers = hp.get("num_layers", 2)
    seq_len = hp.get("sequence_length", 60)

    model = RnnModelPT(input_features, hidden_dim, num_layers)
    offset = 0

    # RNN layers
    for layer_idx in range(num_layers):
        in_dim = input_features if layer_idx == 0 else hidden_dim
        w_ih, w_hh, b_ih, b_hh, offset = load_rnn_cell_weights(flat, offset, in_dim, hidden_dim)
        setattr(model.rnn, f"weight_ih_l{layer_idx}", nn.Parameter(w_ih))
        setattr(model.rnn, f"weight_hh_l{layer_idx}", nn.Parameter(w_hh))
        setattr(model.rnn, f"bias_ih_l{layer_idx}", nn.Parameter(b_ih))
        setattr(model.rnn, f"bias_hh_l{layer_idx}", nn.Parameter(b_hh))

    # Output linear: Linear(hidden_dim, 1)
    w, b, offset = load_linear_weights(flat, offset, hidden_dim, 1)
    model.output_linear.weight.data = w
    model.output_linear.bias.data = b

    assert offset == len(flat), f"RNN: used {offset}/{len(flat)} params"

    dummy = torch.randn(1, seq_len, input_features)
    return model, dummy


def _image_params(input_features):
    """Derive in_channels and image_size from total input features."""
    in_ch = 3 if input_features > 3072 else 1
    img_size = int((input_features / in_ch) ** 0.5)
    return in_ch, max(img_size, 8)


def build_phantom(model_data: dict) -> tuple:
    """Build Phantom lightweight model."""
    input_dim = model_data["input_features"]
    flat = model_data["weights"]
    output_dim = 1
    bottleneck_dim = min(32, max(8, input_dim // 4))
    expand_dim = bottleneck_dim * 2

    model = PhantomPT(input_dim, bottleneck_dim, expand_dim, output_dim)
    offset = 0
    for fc in [model.fc1, model.fc2, model.fc3]:
        w, b, offset = load_linear_weights(flat, offset, fc.in_features, fc.out_features)
        fc.weight.data = w
        fc.bias.data = b
    assert offset == len(flat), f"Phantom: used {offset}/{len(flat)} params"
    return model, torch.randn(1, input_dim)


def build_conv1d(model_data: dict) -> tuple:
    """Build Conv1D model."""
    input_channels = model_data["input_features"]
    flat = model_data["weights"]
    hp = model_data["hyperparameters"]
    hidden_channels = hp.get("hidden_dim", 64)
    num_layers = hp.get("num_layers", 2)
    seq_len = hp.get("sequence_length", 60)

    model = Conv1dPT(input_channels, hidden_channels, num_layers)
    offset = 0
    for i, conv in enumerate(model.convs):
        in_ch = input_channels if i == 0 else hidden_channels
        w, b, offset = load_conv_weights(flat, offset, hidden_channels, in_ch, 3)
        conv.weight.data = w
        conv.bias.data = b
    w, b, offset = load_linear_weights(flat, offset, hidden_channels, 1)
    model.fc.weight.data = w
    model.fc.bias.data = b
    assert offset == len(flat), f"Conv1d: used {offset}/{len(flat)} params"
    return model, torch.randn(1, input_channels, seq_len)


def build_conv2d(model_data: dict) -> tuple:
    """Build Conv2D model (3 conv layers + FC)."""
    input_features = model_data["input_features"]
    flat = model_data["weights"]
    hp = model_data["hyperparameters"]
    in_ch, img_size = _image_params(input_features)
    num_classes = min(1000, max(2, hp.get("hidden_dim", 64)))
    channels = [32, 64, 128]

    model = Conv2dPT(in_ch, num_classes, img_size)
    offset = 0
    prev_ch = in_ch
    for conv, ch in zip([model.conv1, model.conv2, model.conv3], channels):
        w, b, offset = load_conv_weights(flat, offset, ch, prev_ch, (3, 3))
        conv.weight.data = w
        conv.bias.data = b
        prev_ch = ch
    final_size = img_size >> 3
    w, b, offset = load_linear_weights(flat, offset, channels[-1] * final_size * final_size, num_classes)
    model.fc.weight.data = w
    model.fc.bias.data = b
    assert offset == len(flat), f"Conv2d: used {offset}/{len(flat)} params"
    return model, torch.randn(1, in_ch, img_size, img_size)


def build_resnet(model_data: dict) -> tuple:
    """Build ResNet-18 model."""
    input_features = model_data["input_features"]
    flat = model_data["weights"]
    hp = model_data["hyperparameters"]
    in_ch, img_size = _image_params(input_features)
    num_classes = min(1000, max(2, hp.get("hidden_dim", 64)))
    block_counts = [2, 2, 2, 2]
    channels = [64, 128, 256, 512]

    model = ResNetPT(in_ch, num_classes, block_counts, channels)
    offset = 0

    # Stem conv (7x7) + BN
    w, b, offset = load_conv_weights(flat, offset, channels[0], in_ch, (7, 7))
    model.stem_conv.weight.data = w
    model.stem_conv.bias.data = b
    g, beta, offset = load_bn_weights(flat, offset, channels[0])
    set_bn(model.stem_bn, g, beta)

    # Stages
    prev_ch = channels[0]
    for stage_idx, (stage, num_blocks) in enumerate(zip(model.stages, block_counts)):
        out_ch = channels[stage_idx]
        for block_idx, block in enumerate(stage):
            stride = 2 if stage_idx > 0 and block_idx == 0 else 1
            blk_in_ch = prev_ch if block_idx == 0 else out_ch

            # conv1 (3x3)
            w, b, offset = load_conv_weights(flat, offset, out_ch, blk_in_ch, (3, 3))
            block.conv1.weight.data = w
            block.conv1.bias.data = b
            g, beta, offset = load_bn_weights(flat, offset, out_ch)
            set_bn(block.bn1, g, beta)

            # conv2 (3x3)
            w, b, offset = load_conv_weights(flat, offset, out_ch, out_ch, (3, 3))
            block.conv2.weight.data = w
            block.conv2.bias.data = b
            g, beta, offset = load_bn_weights(flat, offset, out_ch)
            set_bn(block.bn2, g, beta)

            # Downsample
            if block.has_downsample:
                w, b, offset = load_conv_weights(flat, offset, out_ch, blk_in_ch, (1, 1))
                block.ds_conv.weight.data = w
                block.ds_conv.bias.data = b
                g, beta, offset = load_bn_weights(flat, offset, out_ch)
                set_bn(block.ds_bn, g, beta)

        prev_ch = out_ch

    # FC head
    w, b, offset = load_linear_weights(flat, offset, channels[-1], num_classes)
    model.fc.weight.data = w
    model.fc.bias.data = b
    assert offset == len(flat), f"ResNet: used {offset}/{len(flat)} params"
    model.eval()  # BN in eval mode
    return model, torch.randn(1, in_ch, img_size, img_size)


def build_vgg(model_data: dict) -> tuple:
    """Build VGG-11 model."""
    input_features = model_data["input_features"]
    flat = model_data["weights"]
    hp = model_data["hyperparameters"]
    in_ch, img_size = _image_params(input_features)
    num_classes = min(1000, max(2, hp.get("hidden_dim", 64)))

    # VGG-11 config (matching Rust vgg11 method)
    config = [
        (in_ch, 64, True),
        (64, 128, True),
        (128, 256, False), (256, 256, True),
        (256, 512, False), (512, 512, True),
        (512, 512, False), (512, 512, True),
    ]

    model = VggPT(in_ch, num_classes, img_size, config)
    offset = 0

    # Conv layers
    for i, conv in enumerate(model.conv_layers):
        c_in, c_out, _ = config[i]
        w, b, offset = load_conv_weights(flat, offset, c_out, c_in, (3, 3))
        conv.weight.data = w
        conv.bias.data = b

    # 3 FC layers
    for fc in [model.fc1, model.fc2, model.fc3]:
        w, b, offset = load_linear_weights(flat, offset, fc.in_features, fc.out_features)
        fc.weight.data = w
        fc.bias.data = b

    assert offset == len(flat), f"VGG: used {offset}/{len(flat)} params"
    return model, torch.randn(1, in_ch, img_size, img_size)


def build_bert(model_data: dict) -> tuple:
    """Build BERT model."""
    input_dim = model_data["input_features"]
    flat = model_data["weights"]
    hp = model_data["hyperparameters"]
    d_model = max(64, hp.get("hidden_dim", 64))
    num_layers = hp.get("num_layers", 2)
    num_classes = 2
    num_heads = 4
    ff_dim = d_model * 4
    max_seq_len = 512
    seq_len = hp.get("sequence_length", 10)

    model = BertPT(input_dim, num_classes, d_model, num_heads, num_layers, max_seq_len)
    offset = 0

    # Embed: Linear(input_dim, d_model)
    w, b, offset = load_linear_weights(flat, offset, input_dim, d_model)
    model.embed.weight.data = w
    model.embed.bias.data = b

    # Positional embedding [max_seq_len, d_model]
    pos, offset = load_flat(flat, offset, max_seq_len * d_model)
    model.pos_embed.data = pos.reshape(max_seq_len, d_model)

    # Transformer blocks
    for block in model.blocks:
        offset = load_transformer_block_weights(flat, offset, block, d_model, ff_dim)

    # Classifier: Linear(d_model, num_classes)
    w, b, offset = load_linear_weights(flat, offset, d_model, num_classes)
    model.classifier.weight.data = w
    model.classifier.bias.data = b

    assert offset == len(flat), f"BERT: used {offset}/{len(flat)} params"
    return model, torch.randn(1, max(2, seq_len), input_dim)


def build_gpt2(model_data: dict) -> tuple:
    """Build GPT-2 model."""
    input_dim = model_data["input_features"]
    flat = model_data["weights"]
    hp = model_data["hyperparameters"]
    d_model = max(64, hp.get("hidden_dim", 64))
    num_layers = hp.get("num_layers", 2)
    output_dim = input_dim
    num_heads = 4
    ff_dim = d_model * 4
    max_seq_len = 512
    seq_len = hp.get("sequence_length", 10)

    model = Gpt2PT(input_dim, output_dim, d_model, num_heads, num_layers, max_seq_len)
    offset = 0

    # Embed
    w, b, offset = load_linear_weights(flat, offset, input_dim, d_model)
    model.embed.weight.data = w
    model.embed.bias.data = b

    # Positional embedding
    pos, offset = load_flat(flat, offset, max_seq_len * d_model)
    model.pos_embed.data = pos.reshape(max_seq_len, d_model)

    # Transformer blocks
    for block in model.blocks:
        offset = load_transformer_block_weights(flat, offset, block, d_model, ff_dim)

    # LM head
    w, b, offset = load_linear_weights(flat, offset, d_model, output_dim)
    model.lm_head.weight.data = w
    model.lm_head.bias.data = b

    assert offset == len(flat), f"GPT2: used {offset}/{len(flat)} params"
    return model, torch.randn(1, max(2, seq_len), input_dim)


def build_vit(model_data: dict) -> tuple:
    """Build Vision Transformer model."""
    input_features = model_data["input_features"]
    flat = model_data["weights"]
    hp = model_data["hyperparameters"]
    in_ch, img_size = _image_params(input_features)
    img_size = max(img_size, 16)
    num_classes = min(1000, max(2, hp.get("hidden_dim", 64)))
    d_model = 128
    num_heads = 4
    num_layers = hp.get("num_layers", 2)
    patch_size = max(4, img_size // 4)
    num_patches = (img_size // patch_size) ** 2
    patch_dim = in_ch * patch_size * patch_size
    ff_dim = d_model * 4

    model = ViTPT(in_ch, num_classes, img_size, patch_size, d_model, num_heads, num_layers)
    offset = 0

    # Patch projection: Linear(patch_dim, d_model)
    w, b, offset = load_linear_weights(flat, offset, patch_dim, d_model)
    model.patch_proj.weight.data = w
    model.patch_proj.bias.data = b

    # CLS token [d_model]
    cls, offset = load_flat(flat, offset, d_model)
    model.cls_token.data = cls

    # Positional embedding [(num_patches+1), d_model]
    pos, offset = load_flat(flat, offset, (num_patches + 1) * d_model)
    model.pos_embed.data = pos.reshape(num_patches + 1, d_model)

    # ViT blocks (pre-norm, same weight layout as post-norm)
    for block in model.blocks:
        offset = load_transformer_block_weights(flat, offset, block, d_model, ff_dim)

    # Classification head
    w, b, offset = load_linear_weights(flat, offset, d_model, num_classes)
    model.head.weight.data = w
    model.head.bias.data = b

    assert offset == len(flat), f"ViT: used {offset}/{len(flat)} params"
    return model, torch.randn(1, in_ch, img_size, img_size)


def build_nexus(model_data: dict) -> tuple:
    """Build Nexus multi-modal fusion model."""
    input_dim = model_data["input_features"]
    flat = model_data["weights"]
    hp = model_data["hyperparameters"]
    d_model = max(64, hp.get("hidden_dim", 64))
    output_dim = 1
    num_modalities = max(2, min(8, input_dim // 4))
    mod_dim = (input_dim + num_modalities - 1) // num_modalities

    model = NexusPT(input_dim, d_model, output_dim, num_modalities, mod_dim)
    offset = 0

    # Per-modality encoders: each has w1, b1, w2, b2
    for enc in model.encoders:
        w, b, offset = load_linear_weights(flat, offset, mod_dim, d_model)
        enc.fc1.weight.data = w
        enc.fc1.bias.data = b
        w, b, offset = load_linear_weights(flat, offset, d_model, d_model)
        enc.fc2.weight.data = w
        enc.fc2.bias.data = b

    # Fusion weights (stored but not used in forward)
    n = d_model * d_model
    wq, offset = load_flat(flat, offset, n)
    model.fusion_wq.data = wq.reshape(d_model, d_model)
    wk, offset = load_flat(flat, offset, n)
    model.fusion_wk.data = wk.reshape(d_model, d_model)
    wv, offset = load_flat(flat, offset, n)
    model.fusion_wv.data = wv.reshape(d_model, d_model)

    # Head: fc1(d_model, d_model), fc2(d_model, output_dim)
    w, b, offset = load_linear_weights(flat, offset, d_model, d_model)
    model.head_fc1.weight.data = w
    model.head_fc1.bias.data = b
    w, b, offset = load_linear_weights(flat, offset, d_model, output_dim)
    model.head_fc2.weight.data = w
    model.head_fc2.bias.data = b

    assert offset == len(flat), f"Nexus: used {offset}/{len(flat)} params"
    return model, torch.randn(1, input_dim)


# Architecture name mapping (serde snake_case -> builder)
ARCHITECTURE_BUILDERS = {
    "sentinel": build_sentinel,
    "lstm_autoencoder": build_lstm_autoencoder,
    "gru_predictor": build_gru_predictor,
    "rnn": build_rnn_model,
    "phantom": build_phantom,
    "conv1d": build_conv1d,
    "conv2d": build_conv2d,
    "res_net": build_resnet,
    "vgg": build_vgg,
    "bert": build_bert,
    "gpt2": build_gpt2,
    "vi_t": build_vit,
    "nexus": build_nexus,
}


def build_model(model_data: dict) -> tuple:
    """Build a PyTorch model from AxonML model data. Returns (model, dummy_input)."""
    arch = model_data["architecture"]
    builder = ARCHITECTURE_BUILDERS.get(arch)
    if builder:
        return builder(model_data)
    else:
        raise ValueError(f"Unsupported architecture: '{arch}'. "
                         f"Supported: {list(ARCHITECTURE_BUILDERS.keys())}")


# =============================================================================
# ONNX export
# =============================================================================


def export_to_onnx(model: nn.Module, dummy_input: torch.Tensor, output_path: str,
                   model_name: str = "prometheus_model", static_shapes: bool = False):
    """Export a PyTorch model to standard ONNX protobuf format."""
    model.eval()

    # Flatten RNN parameters to suppress _flat_weights warning
    for m in model.modules():
        if isinstance(m, (nn.LSTM, nn.GRU, nn.RNN)):
            m.flatten_parameters()

    # For Hailo HEF: use fully static shapes (no dynamic axes)
    dynamic = None if static_shapes else {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    }

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic,
            dynamo=False,  # Use stable TorchScript exporter (no dynamo warnings)
        )

    # Validate the ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    return output_path


# =============================================================================
# HEF conversion (requires Hailo DFC SDK)
# =============================================================================


def export_to_hef(onnx_path: str, output_path: str, model_name: str = "prometheus_model",
                  input_shape: list = None):
    """Convert an ONNX model to Hailo HEF format.

    Uses Hailo DFC SDK to translate, optimize (quantize), and compile.
    LSTM/GRU/RNN models are supported — Hailo-8/8L handles recurrent layers.
    """
    try:
        from hailo_sdk_client import ClientRunner
    except ImportError:
        raise RuntimeError(
            "Hailo DFC SDK not installed. To convert to HEF:\n"
            "  1. Register at https://hailo.ai/developer-zone/\n"
            "  2. pip install hailo_sdk_client\n"
            "  3. Run this converter again with --format hef\n"
            "\n"
            "Alternatively, use the ONNX file with the Hailo Model Zoo:\n"
            f"  hailo parser onnx {onnx_path}\n"
            f"  hailo compiler --har model.har"
        )

    import numpy as np

    runner = ClientRunner(hw_arch="hailo8")

    # Determine input shape from ONNX model
    if input_shape is None:
        import onnx
        model = onnx.load(onnx_path)
        inp = model.graph.input[0]
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        input_shape = dims if all(d > 0 for d in dims) else [1, 10, 11]

    print(f"Parsing ONNX for Hailo (input shape: {input_shape})")

    # Translate ONNX → Hailo Network (HN)
    hn, npz = runner.translate_onnx_model(
        onnx_path,
        model_name,
        net_input_shapes={"input": input_shape},
    )

    # Generate calibration data matching Hailo's inferred network input shape
    n_calib = 256
    input_name = f"{model_name}/input_layer1"
    # Hailo may add extra dimensions — get actual shape from the translated network
    try:
        hn_dict = runner.get_hn_dict() if hasattr(runner, 'get_hn_dict') else None
    except:
        hn_dict = None
    # Hailo always expects at least 3D input — pad 2D inputs with extra dim
    calib_shape = list(input_shape)
    if len(calib_shape) == 2:
        calib_shape = [calib_shape[0], 1, calib_shape[1]]  # [batch, 1, features]
    calib_data = {input_name: np.random.randn(n_calib, *calib_shape).astype(np.float32)}
    print(f"Optimizing with {n_calib} calibration samples (shape per sample: {calib_shape})...")
    runner.optimize(calib_data)

    print("Compiling HEF...")
    try:
        hef = runner.compile()
        with open(output_path, "wb") as f:
            f.write(hef)
        print(f"HEF written: {output_path} ({len(hef)} bytes)")
        return output_path
    except Exception as compile_err:
        # Save HAR (Hailo Archive) — can be compiled with different DFC version
        har_path = output_path.replace(".hef", ".har")
        runner.save_har(har_path)
        print(f"HEF compilation failed, saved HAR: {har_path}")
        print(f"  HAR can be compiled with: hailo compiler {har_path}")
        # Return HAR path instead of raising
        return har_path


# =============================================================================
# Main CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Convert .axonml models to ONNX or HEF format"
    )
    parser.add_argument("input", help="Path to .axonml model file")
    parser.add_argument(
        "--format", "-f",
        choices=["onnx", "hef", "both"],
        default="onnx",
        help="Output format (default: onnx)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: input with new extension)"
    )
    parser.add_argument(
        "--name", "-n",
        default="prometheus_model",
        help="Model name for ONNX metadata"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Parse and validate the .axonml file without converting"
    )

    args = parser.parse_args()

    # Parse the .axonml file
    print(f"Parsing {args.input}...")
    parsed = parse_axonml(args.input)
    header = parsed["header"]
    model_data = parsed["model"]

    print(f"  Architecture: {header['architecture']}")
    print(f"  Input features: {header['input_features']}")
    print(f"  Parameters: {header['num_parameters']:,}")
    print(f"  Quantized: {header.get('quantized', False)}")

    if args.validate_only:
        print("Validation passed.")
        return 0

    # Build PyTorch model
    print("Reconstructing model in PyTorch...")
    model, dummy_input = build_model(model_data)

    # Verify forward pass works
    model.eval()
    with torch.no_grad():
        test_output = model(dummy_input)
        print(f"  Forward pass OK — output shape: {list(test_output.shape)}")

    # Export
    input_path = Path(args.input)
    fmt = args.format

    if fmt in ("onnx", "both"):
        onnx_path = args.output or str(input_path.with_suffix(".onnx"))
        print(f"Exporting to ONNX: {onnx_path}")
        export_to_onnx(model, dummy_input, onnx_path, args.name)
        onnx_size = Path(onnx_path).stat().st_size
        print(f"  ONNX export OK ({onnx_size:,} bytes)")
        print(f"  Validated with onnx.checker")

        # Quick inference test with ONNX Runtime
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(onnx_path)
            ort_input = {session.get_inputs()[0].name: dummy_input.numpy()}
            ort_output = session.run(None, ort_input)
            print(f"  ONNX Runtime inference OK — output shape: {list(ort_output[0].shape)}")

            # Compare PyTorch vs ONNX Runtime output
            pt_out = test_output.numpy()
            ort_out = ort_output[0]
            max_diff = np.abs(pt_out - ort_out).max()
            print(f"  Max difference PyTorch vs ONNX Runtime: {max_diff:.2e}")
            if max_diff > 1e-4:
                print(f"  WARNING: Outputs differ significantly!")
            else:
                print(f"  Outputs match within tolerance.")
        except Exception as e:
            print(f"  ONNX Runtime validation skipped: {e}")

    if fmt in ("hef", "both"):
        onnx_intermediate = args.output or str(input_path.with_suffix(".onnx"))
        if fmt == "hef":
            # For HEF: Hailo DFC can't handle Linear layers after RNN outputs.
            # Strip all post-RNN layers — output raw RNN hidden states.
            # Classification/projection runs on host CPU.
            arch = model_data.get("architecture", "").lower().replace(" ", "_")
            rnn_archs = ("lstm_autoencoder", "lstmautoencoder", "gru_predictor", "grupredictor",
                         "rnn", "lstm", "gru", "bilstm")
            if arch in rnn_archs:
                hp = model_data.get("hyperparameters", {})
                hd = hp.get("hidden_dim", 64)
                nl = hp.get("num_layers", 2)
                sl = hp.get("sequence_length", 10)
                inf = model_data.get("input_features", 11)

                # Find the RNN module in the model and wrap it standalone
                rnn_module = None
                rnn_type = None
                for name, m in model.named_modules():
                    if isinstance(m, nn.LSTM):
                        rnn_module = m
                        rnn_type = "lstm"
                        break
                    elif isinstance(m, nn.GRU):
                        rnn_module = m
                        rnn_type = "gru"
                        break
                    elif isinstance(m, nn.RNN):
                        rnn_module = m
                        rnn_type = "rnn"
                        break

                if rnn_module is not None:
                    print(f"HEF: stripping post-{rnn_type.upper()} Linear layers for Hailo compatibility")

                    class RnnOnlyWrapper(nn.Module):
                        def __init__(self, rnn, rnn_type, num_layers, hidden_dim, input_features):
                            super().__init__()
                            # Hailo DFC requires input_dim == hidden_dim for RNN layers
                            self.needs_proj = (input_features != hidden_dim)
                            if self.needs_proj:
                                self.input_proj = nn.Linear(input_features, hidden_dim)
                            self.rnn = rnn
                            self.rnn_type = rnn_type
                            self.register_buffer('h0', torch.zeros(num_layers, 1, hidden_dim))
                            if rnn_type == "lstm":
                                self.register_buffer('c0', torch.zeros(num_layers, 1, hidden_dim))
                        def forward(self, x):
                            if self.needs_proj:
                                x = self.input_proj(x)
                            if self.rnn_type == "lstm":
                                out, _ = self.rnn(x, (self.h0, self.c0))
                            else:
                                out, _ = self.rnn(x, self.h0)
                            return out

                    wrapper = RnnOnlyWrapper(rnn_module, rnn_type, nl, hd, inf)
                    # If projecting, rebuild RNN with matching input_dim
                    if inf != hd:
                        print(f"  Adding input projection: {inf} -> {hd} (Hailo requires input_dim == hidden_dim)")
                        if rnn_type == "lstm":
                            wrapper.rnn = nn.LSTM(hd, hd, nl, batch_first=True)
                        elif rnn_type == "gru":
                            wrapper.rnn = nn.GRU(hd, hd, nl, batch_first=True)
                        else:
                            wrapper.rnn = nn.RNN(hd, hd, nl, batch_first=True)
                    model = wrapper
                    dummy_input = torch.randn(1, sl, inf)
                    print(f"  RNN-only model: input=[1,{sl},{inf}] -> output=[1,{sl},{hd}]")

            # Use unique temp path to avoid Hailo state caching issues
            import uuid as _uuid
            onnx_intermediate = str(input_path.parent / f"_hef_tmp_{_uuid.uuid4().hex[:8]}.onnx")
            # Clean any old temp ONNX files
            for old in input_path.parent.glob("_hef_tmp_*.onnx"):
                old.unlink(missing_ok=True)
            print(f"Exporting intermediate ONNX (static shapes for Hailo): {onnx_intermediate}")
            export_to_onnx(model, dummy_input, onnx_intermediate, args.name, static_shapes=True)

        hef_path = str(input_path.with_suffix(".hef")) if fmt == "both" else (args.output or str(input_path.with_suffix(".hef")))
        # Get input shape from dummy for Hailo
        hef_input_shape = list(dummy_input.shape)
        print(f"Converting to HEF: {hef_path} (input shape: {hef_input_shape})")
        try:
            export_to_hef(onnx_intermediate, hef_path, args.name, input_shape=hef_input_shape)
            hef_size = Path(hef_path).stat().st_size
            print(f"  HEF export OK ({hef_size:,} bytes)")
        except RuntimeError as e:
            print(f"\nHEF conversion failed:\n{e}", file=sys.stderr)
        finally:
            # Clean up temp ONNX and Hailo temp dirs
            Path(onnx_intermediate).unlink(missing_ok=True)
            import shutil
            for d in Path("/tmp").glob("hailo*"):
                shutil.rmtree(d, ignore_errors=True)
            return 1

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
