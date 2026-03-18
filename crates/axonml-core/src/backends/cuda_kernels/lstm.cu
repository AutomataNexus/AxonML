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
// Fused LSTM Gate Backward Kernel
// =============================================================================
//
// Given grad_h (from output) and grad_c_next (from next timestep cell),
// plus the saved forward states, computes:
//   grad_gates [batch, 4*hidden] - gradient w.r.t. pre-activation gates
//   grad_c_prev [batch, hidden] - gradient w.r.t. previous cell state
//
// For each (b, h):
//   Recompute activations from saved pre-activation gates
//   dc = grad_c_next + grad_h * o * (1 - tanh(c_new)^2)
//   di = dc * g * i * (1-i)
//   df = dc * c_prev * f * (1-f)
//   dg = dc * i * (1 - g^2)
//   do = grad_h * tanh(c_new) * o * (1-o)
//   grad_c_prev = dc * f

extern "C" __global__ void lstm_gates_backward_f32(
    const float* __restrict__ gates,       // [batch, 4*hidden] - pre-activation gates
    const float* __restrict__ c_prev,      // [batch, hidden]
    const float* __restrict__ c_new,       // [batch, hidden] - from forward
    const float* __restrict__ grad_h,      // [batch, hidden] - gradient from output
    const float* __restrict__ grad_c_next, // [batch, hidden] - gradient from next timestep cell
    float* __restrict__ grad_gates,        // [batch, 4*hidden] - output: gate gradients
    float* __restrict__ grad_c_prev,       // [batch, hidden] - output: cell gradient to prev timestep
    unsigned int hidden_size,
    unsigned int total                     // batch * hidden
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    unsigned int b = idx / hidden_size;
    unsigned int h = idx % hidden_size;
    unsigned int base = b * 4 * hidden_size;

    // Load pre-activation gates
    float i_pre = gates[base + h];
    float f_pre = gates[base + hidden_size + h];
    float g_pre = gates[base + 2 * hidden_size + h];
    float o_pre = gates[base + 3 * hidden_size + h];

    // Recompute activations
    float i_act = 1.0f / (1.0f + expf(-i_pre));   // sigmoid
    float f_act = 1.0f / (1.0f + expf(-f_pre));   // sigmoid
    float g_act = tanhf(g_pre);                     // tanh
    float o_act = 1.0f / (1.0f + expf(-o_pre));   // sigmoid

    // Load saved forward state
    float c = c_new[idx];
    float tanh_c = tanhf(c);

    // Load incoming gradients
    float dh = grad_h[idx];
    float dc_next = grad_c_next[idx];

    // Cell state gradient: dc = dc_next + dh * o * (1 - tanh(c)^2)
    float dc = dc_next + dh * o_act * (1.0f - tanh_c * tanh_c);

    // Gate gradients (pre-activation, i.e. through the activation derivative)
    float di = dc * g_act * i_act * (1.0f - i_act);              // sigmoid derivative
    float df = dc * c_prev[idx] * f_act * (1.0f - f_act);        // sigmoid derivative
    float dg = dc * i_act * (1.0f - g_act * g_act);              // tanh derivative
    float do_gate = dh * tanh_c * o_act * (1.0f - o_act);        // sigmoid derivative

    // Write gate gradients
    grad_gates[base + h] = di;
    grad_gates[base + hidden_size + h] = df;
    grad_gates[base + 2 * hidden_size + h] = dg;
    grad_gates[base + 3 * hidden_size + h] = do_gate;

    // Cell gradient to previous timestep
    grad_c_prev[idx] = dc * f_act;
}

// =============================================================================
// Fused GRU Gate Backward Kernel
// =============================================================================
//
// Given grad_h_new (gradient of loss w.r.t. h_new), plus saved forward
// intermediates, computes:
//   grad_gates_ih [batch, 3*hidden] - gradient w.r.t. input-hidden pre-activations
//   grad_gates_hh [batch, 3*hidden] - gradient w.r.t. hidden-hidden pre-activations
//   grad_h_prev [batch, hidden] - gradient w.r.t. previous hidden state
//
// Forward:
//   r = sigmoid(r_ih + r_hh)
//   z = sigmoid(z_ih + z_hh)
//   n = tanh(n_ih + r * n_hh)
//   h_new = (1 - z) * n + z * h_prev
//
// Backward:
//   dh_prev = grad_h_new * z
//   dn = grad_h_new * (1 - z)
//   dz = grad_h_new * (h_prev - n)
//
//   d_n_pre = dn * (1 - n^2)                    (tanh derivative)
//   d_n_ih = d_n_pre
//   d_n_hh = d_n_pre * r
//   dr = d_n_pre * n_hh
//
//   d_z_pre = dz * z * (1 - z)                  (sigmoid derivative)
//   d_z_ih = d_z_pre
//   d_z_hh = d_z_pre
//
//   d_r_pre = dr * r * (1 - r)                  (sigmoid derivative)
//   d_r_ih = d_r_pre
//   d_r_hh = d_r_pre

extern "C" __global__ void gru_gates_backward_f32(
    const float* __restrict__ gates_ih,    // [batch, 3*hidden] - pre-activation ih gates
    const float* __restrict__ gates_hh,    // [batch, 3*hidden] - pre-activation hh gates
    const float* __restrict__ h_prev,      // [batch, hidden]
    const float* __restrict__ grad_h_new,  // [batch, hidden] - gradient from output
    float* __restrict__ grad_gates_ih,     // [batch, 3*hidden] - output: ih gate gradients
    float* __restrict__ grad_gates_hh,     // [batch, 3*hidden] - output: hh gate gradients
    float* __restrict__ grad_h_prev,       // [batch, hidden] - output: gradient to prev hidden
    unsigned int hidden_size,
    unsigned int total                     // batch * hidden
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    unsigned int b = idx / hidden_size;
    unsigned int h = idx % hidden_size;
    unsigned int base = b * 3 * hidden_size;

    // Load pre-activation gates
    float r_ih = gates_ih[base + h];
    float z_ih = gates_ih[base + hidden_size + h];
    float n_ih = gates_ih[base + 2 * hidden_size + h];

    float r_hh = gates_hh[base + h];
    float z_hh = gates_hh[base + hidden_size + h];
    float n_hh_val = gates_hh[base + 2 * hidden_size + h];

    // Recompute activations
    float r = 1.0f / (1.0f + expf(-(r_ih + r_hh)));   // sigmoid
    float z = 1.0f / (1.0f + expf(-(z_ih + z_hh)));   // sigmoid
    float n = tanhf(n_ih + r * n_hh_val);               // tanh

    float hp = h_prev[idx];
    float dh = grad_h_new[idx];

    // Output gate gradients
    // h_new = (1 - z) * n + z * h_prev
    float dz = dh * (hp - n);
    float dn = dh * (1.0f - z);

    // Gradient through h_prev path
    grad_h_prev[idx] = dh * z;

    // tanh derivative for n
    float d_n_pre = dn * (1.0f - n * n);

    // sigmoid derivative for z
    float d_z_pre = dz * z * (1.0f - z);

    // r gradient: d_n_pre * n_hh_val (chain rule through r * n_hh)
    float dr = d_n_pre * n_hh_val;
    float d_r_pre = dr * r * (1.0f - r);

    // Write ih gate gradients
    grad_gates_ih[base + h] = d_r_pre;
    grad_gates_ih[base + hidden_size + h] = d_z_pre;
    grad_gates_ih[base + 2 * hidden_size + h] = d_n_pre;

    // Write hh gate gradients
    grad_gates_hh[base + h] = d_r_pre;
    grad_gates_hh[base + hidden_size + h] = d_z_pre;
    grad_gates_hh[base + 2 * hidden_size + h] = d_n_pre * r;
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
