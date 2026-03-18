// =============================================================================
// Fused LSTM Gate Kernel
// =============================================================================
//
// Computes all 4 LSTM gates in a single kernel launch:
//   gates = x @ W_ih^T + h @ W_hh^T + bias_ih + bias_hh
//   i = sigmoid(gates[0:H])
//   f = sigmoid(gates[H:2H])
//   g = tanh(gates[2H:3H])
//   o = sigmoid(gates[3H:4H])
//   c_new = f * c + i * g
//   h_new = o * tanh(c_new)
//
// This kernel handles the post-GEMM gate computation.
// The GEMM itself (x @ W^T) is done via cuBLAS before this kernel.
//
// Author: AutomataNexus
// =============================================================================

extern "C" __global__ void lstm_gates_f32(
    const float* __restrict__ gates,   // [batch, 4*hidden] = ih + hh (pre-computed by cuBLAS)
    const float* __restrict__ c_prev,  // [batch, hidden]
    float* __restrict__ h_new,         // [batch, hidden] output
    float* __restrict__ c_new,         // [batch, hidden] output
    unsigned int hidden_size,
    unsigned int total                 // batch * hidden
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    unsigned int b = idx / hidden_size;
    unsigned int h = idx % hidden_size;
    unsigned int base = b * 4 * hidden_size;

    float i_gate = gates[base + h];
    float f_gate = gates[base + hidden_size + h];
    float g_gate = gates[base + 2 * hidden_size + h];
    float o_gate = gates[base + 3 * hidden_size + h];

    // Activations
    i_gate = 1.0f / (1.0f + expf(-i_gate));  // sigmoid
    f_gate = 1.0f / (1.0f + expf(-f_gate));  // sigmoid
    g_gate = tanhf(g_gate);                    // tanh
    o_gate = 1.0f / (1.0f + expf(-o_gate));  // sigmoid

    // Cell state update
    float c = f_gate * c_prev[idx] + i_gate * g_gate;
    c_new[idx] = c;

    // Hidden state update
    h_new[idx] = o_gate * tanhf(c);
}

// =============================================================================
// Fused GRU Gate Kernel
// =============================================================================
//
// gates_ih = x @ W_ih^T + bias_ih    [batch, 3*hidden]
// gates_hh = h @ W_hh^T + bias_hh    [batch, 3*hidden]
// r = sigmoid(gates_ih[0:H] + gates_hh[0:H])
// z = sigmoid(gates_ih[H:2H] + gates_hh[H:2H])
// n = tanh(gates_ih[2H:3H] + r * gates_hh[2H:3H])
// h_new = (1 - z) * n + z * h_prev

extern "C" __global__ void gru_gates_f32(
    const float* __restrict__ gates_ih,  // [batch, 3*hidden]
    const float* __restrict__ gates_hh,  // [batch, 3*hidden]
    const float* __restrict__ h_prev,    // [batch, hidden]
    float* __restrict__ h_new,           // [batch, hidden] output
    unsigned int hidden_size,
    unsigned int total                   // batch * hidden
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    unsigned int b = idx / hidden_size;
    unsigned int h = idx % hidden_size;
    unsigned int base = b * 3 * hidden_size;

    float r_ih = gates_ih[base + h];
    float z_ih = gates_ih[base + hidden_size + h];
    float n_ih = gates_ih[base + 2 * hidden_size + h];

    float r_hh = gates_hh[base + h];
    float z_hh = gates_hh[base + hidden_size + h];
    float n_hh = gates_hh[base + 2 * hidden_size + h];

    // Reset and update gates
    float r = 1.0f / (1.0f + expf(-(r_ih + r_hh)));  // sigmoid
    float z = 1.0f / (1.0f + expf(-(z_ih + z_hh)));  // sigmoid

    // New gate (reset applied to hidden-to-hidden)
    float n = tanhf(n_ih + r * n_hh);

    // Output
    h_new[idx] = (1.0f - z) * n + z * h_prev[idx];
}

// =============================================================================
// Fused BatchNorm Forward Kernel
// =============================================================================
//
// For each channel c:
//   mean[c] = mean(x[:, c, :, :])
//   var[c] = var(x[:, c, :, :])
//   x_norm = (x - mean) / sqrt(var + eps)
//   y = gamma * x_norm + beta
//
// Two-pass approach:
//   Pass 1: Compute mean and var per channel (reduction)
//   Pass 2: Normalize + affine transform

// Pass 1: Compute sum and sum_sq per channel using atomic adds
extern "C" __global__ void batchnorm_stats_f32(
    const float* __restrict__ x,     // [N, C, spatial]
    float* __restrict__ sum_out,     // [C]
    float* __restrict__ sum_sq_out,  // [C]
    unsigned int N,
    unsigned int C,
    unsigned int spatial              // H * W
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * C * spatial;
    if (idx >= total) return;

    unsigned int c = (idx / spatial) % C;

    float val = x[idx];
    atomicAdd(&sum_out[c], val);
    atomicAdd(&sum_sq_out[c], val * val);
}

// Pass 2: Normalize using pre-computed mean/var, apply affine
extern "C" __global__ void batchnorm_norm_f32(
    const float* __restrict__ x,       // [N, C, spatial]
    const float* __restrict__ mean,    // [C]
    const float* __restrict__ var,     // [C]
    const float* __restrict__ gamma,   // [C]
    const float* __restrict__ beta,    // [C]
    float* __restrict__ y,             // [N, C, spatial]
    float eps,
    unsigned int C,
    unsigned int spatial,
    unsigned int total                 // N * C * spatial
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    unsigned int c = (idx / spatial) % C;

    float x_val = x[idx];
    float m = mean[c];
    float v = var[c];
    float inv_std = rsqrtf(v + eps);

    y[idx] = gamma[c] * (x_val - m) * inv_std + beta[c];
}
