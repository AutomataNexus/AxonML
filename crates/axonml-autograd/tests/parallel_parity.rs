//! Parity tests for the CPU backward paths that switch to a rayon branch above
//! a size threshold. Each case is large enough to take the parallel branch and
//! is checked against a plain sequential reference written from the math.

use axonml_autograd::GradientFunction;
use axonml_autograd::functions::{
    FusedAttentionBackward, GruGatesBackward, LogSoftmaxBackward, LstmGatesBackward,
    NarrowBackward, SoftmaxBackward, SumDimBackward,
};
use axonml_tensor::Tensor;

// ── helpers ──

fn seq(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2654435761).wrapping_add(12345);
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x % 2000) as f32 / 1000.0 - 1.0
        })
        .collect()
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol * (1.0 + w.abs()),
            "{what}: element {i} got {g} want {w}"
        );
    }
}

fn softmax_rows(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = row.iter().map(|v| (v - m).exp()).collect();
        let s: f32 = e.iter().sum();
        for c in 0..cols {
            out[r * cols + c] = e[c] / s;
        }
    }
    out
}

// ── softmax / log-softmax backward ──

fn softmax_bwd_ref(s: &[f32], g: &[f32], shape: &[usize], dim: usize) -> Vec<f32> {
    let dim_size = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    let outer: usize = shape[..dim].iter().product();
    let mut out = vec![0.0f32; s.len()];
    for o in 0..outer {
        for i in 0..inner {
            let idx = |d: usize| (o * dim_size + d) * inner + i;
            let dot: f32 = (0..dim_size).map(|d| s[idx(d)] * g[idx(d)]).sum();
            for d in 0..dim_size {
                out[idx(d)] = s[idx(d)] * (g[idx(d)] - dot);
            }
        }
    }
    out
}

fn log_softmax_bwd_ref(logp: &[f32], g: &[f32], shape: &[usize], dim: usize) -> Vec<f32> {
    let dim_size = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    let outer: usize = shape[..dim].iter().product();
    let mut out = vec![0.0f32; logp.len()];
    for o in 0..outer {
        for i in 0..inner {
            let idx = |d: usize| (o * dim_size + d) * inner + i;
            let sum_g: f32 = (0..dim_size).map(|d| g[idx(d)]).sum();
            for d in 0..dim_size {
                out[idx(d)] = g[idx(d)] - logp[idx(d)].exp() * sum_g;
            }
        }
    }
    out
}

fn check_softmax(shape: &[usize], dim: usize) {
    let n: usize = shape.iter().product();
    let s = seq(n, 1).iter().map(|v| v.abs() + 0.01).collect::<Vec<_>>();
    let g = seq(n, 2);
    let out = Tensor::from_vec(s.clone(), shape).unwrap();
    let grad = Tensor::from_vec(g.clone(), shape).unwrap();
    let got = SoftmaxBackward::new(None, out, dim as i64).apply(&grad)[0]
        .as_ref()
        .unwrap()
        .to_vec();
    assert_close(&got, &softmax_bwd_ref(&s, &g, shape, dim), 1e-5, "softmax");

    let logp = Tensor::from_vec(s.iter().map(|v| v.ln()).collect(), shape).unwrap();
    let got = LogSoftmaxBackward::new(None, logp, dim as i64).apply(&grad)[0]
        .as_ref()
        .unwrap()
        .to_vec();
    let lp: Vec<f32> = s.iter().map(|v| v.ln()).collect();
    assert_close(
        &got,
        &log_softmax_bwd_ref(&lp, &g, shape, dim),
        1e-4,
        "log_softmax",
    );
}

#[test]
fn softmax_backward_2d_dim0_parallel() {
    check_softmax(&[96, 80], 0);
}

#[test]
fn softmax_backward_2d_dim1_parallel() {
    check_softmax(&[96, 80], 1);
}

#[test]
fn softmax_backward_nd_middle_dim_parallel() {
    check_softmax(&[8, 24, 40], 1);
}

#[test]
fn softmax_backward_nd_last_dim_parallel() {
    check_softmax(&[8, 24, 40], 2);
}

// ── narrow / sum-dim backward ──

fn check_narrow(shape: &[usize], dim: usize, start: usize, len: usize) {
    let mut out_shape = shape.to_vec();
    out_shape[dim] = len;
    let out_n: usize = out_shape.iter().product();
    assert!(out_n >= 4096, "case must take the parallel branch");
    let g = seq(out_n, 3);
    let grad = Tensor::from_vec(g.clone(), &out_shape).unwrap();
    let got = NarrowBackward::new(None, shape.to_vec(), dim, start).apply(&grad)[0]
        .as_ref()
        .unwrap()
        .to_vec();

    let n: usize = shape.iter().product();
    let mut want = vec![0.0f32; n];
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    for (out_idx, &gv) in g.iter().enumerate() {
        let mut rem = out_idx;
        let mut in_idx = 0;
        for d in (0..shape.len()).rev() {
            let mut c = rem % out_shape[d];
            rem /= out_shape[d];
            if d == dim {
                c += start;
            }
            in_idx += c * strides[d];
        }
        want[in_idx] = gv;
    }
    assert_close(&got, &want, 0.0, "narrow");
}

#[test]
fn narrow_backward_first_dim_parallel() {
    check_narrow(&[64, 32, 8], 0, 7, 20);
}

#[test]
fn narrow_backward_middle_dim_parallel() {
    check_narrow(&[16, 64, 8], 1, 5, 40);
}

#[test]
fn narrow_backward_last_dim_parallel() {
    check_narrow(&[16, 32, 64], 2, 3, 12);
}

#[test]
fn narrow_backward_1d_parallel() {
    check_narrow(&[10000], 0, 100, 5000);
}

fn check_sum_dim(shape: &[usize], dim: usize) {
    let mut out_shape = shape.to_vec();
    out_shape.remove(dim);
    let out_n: usize = out_shape.iter().product();
    let g = seq(out_n, 4);
    let grad = Tensor::from_vec(g.clone(), &out_shape).unwrap();
    let got = SumDimBackward::new(None, shape.to_vec(), dim).apply(&grad)[0]
        .as_ref()
        .unwrap()
        .to_vec();
    let dim_size = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    let outer: usize = shape[..dim].iter().product();
    let mut want = vec![0.0f32; shape.iter().product()];
    for o in 0..outer {
        for d in 0..dim_size {
            for i in 0..inner {
                want[(o * dim_size + d) * inner + i] = g[o * inner + i];
            }
        }
    }
    assert_close(&got, &want, 0.0, "sum_dim");
}

#[test]
fn sum_dim_backward_first_dim_parallel() {
    check_sum_dim(&[48, 64, 4], 0);
}

#[test]
fn sum_dim_backward_middle_dim_parallel() {
    check_sum_dim(&[32, 48, 8], 1);
}

#[test]
fn sum_dim_backward_last_dim_parallel() {
    check_sum_dim(&[64, 80, 3], 2);
}

// ── LSTM / GRU gate backward ──

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[test]
fn lstm_gates_backward_parallel() {
    let (b, hs) = (24, 256);
    let gates = seq(b * 4 * hs, 5);
    let c_prev = seq(b * hs, 6);
    let c_new = seq(b * hs, 7);
    let gh = seq(b * hs, 8);
    let f = LstmGatesBackward::new(
        None,
        None,
        Tensor::from_vec(gates.clone(), &[b, 4 * hs]).unwrap(),
        Tensor::from_vec(c_prev.clone(), &[b, hs]).unwrap(),
        Tensor::from_vec(c_new.clone(), &[b, hs]).unwrap(),
        hs,
    );
    let grads = f.apply(&Tensor::from_vec(gh.clone(), &[b, hs]).unwrap());
    let got_gates = grads[0].as_ref().unwrap().to_vec();
    let got_c = grads[1].as_ref().unwrap().to_vec();

    let mut want_gates = vec![0.0f32; b * 4 * hs];
    let mut want_c = vec![0.0f32; b * hs];
    for bi in 0..b {
        for h in 0..hs {
            let idx = bi * hs + h;
            let base = bi * 4 * hs;
            let i_act = sigmoid(gates[base + h]);
            let f_act = sigmoid(gates[base + hs + h]);
            let g_act = gates[base + 2 * hs + h].tanh();
            let o_act = sigmoid(gates[base + 3 * hs + h]);
            let tanh_c = c_new[idx].tanh();
            let dh = gh[idx];
            let dc = dh * o_act * (1.0 - tanh_c * tanh_c);
            want_gates[base + h] = dc * g_act * i_act * (1.0 - i_act);
            want_gates[base + hs + h] = dc * c_prev[idx] * f_act * (1.0 - f_act);
            want_gates[base + 2 * hs + h] = dc * i_act * (1.0 - g_act * g_act);
            want_gates[base + 3 * hs + h] = dh * tanh_c * o_act * (1.0 - o_act);
            want_c[idx] = dc * f_act;
        }
    }
    assert_close(&got_gates, &want_gates, 1e-6, "lstm gates");
    assert_close(&got_c, &want_c, 1e-6, "lstm c_prev");
}

#[test]
fn gru_gates_backward_parallel() {
    let (b, hs) = (24, 256);
    let ih = seq(b * 3 * hs, 10);
    let hh = seq(b * 3 * hs, 11);
    let h_prev = seq(b * hs, 12);
    let g = seq(b * hs, 13);
    let f = GruGatesBackward::new(
        None,
        None,
        None,
        Tensor::from_vec(ih.clone(), &[b, 3 * hs]).unwrap(),
        Tensor::from_vec(hh.clone(), &[b, 3 * hs]).unwrap(),
        Tensor::from_vec(h_prev.clone(), &[b, hs]).unwrap(),
        hs,
    );
    let grads = f.apply(&Tensor::from_vec(g.clone(), &[b, hs]).unwrap());
    let got_ih = grads[0].as_ref().unwrap().to_vec();
    let got_hh = grads[1].as_ref().unwrap().to_vec();
    let got_hp = grads[2].as_ref().unwrap().to_vec();

    let mut want_ih = vec![0.0f32; b * 3 * hs];
    let mut want_hh = vec![0.0f32; b * 3 * hs];
    let mut want_hp = vec![0.0f32; b * hs];
    for bi in 0..b {
        for h in 0..hs {
            let idx = bi * hs + h;
            let base = bi * 3 * hs;
            let r = sigmoid(ih[base + h] + hh[base + h]);
            let z = sigmoid(ih[base + hs + h] + hh[base + hs + h]);
            let n_hh = hh[base + 2 * hs + h];
            let n = (ih[base + 2 * hs + h] + r * n_hh).tanh();
            let dh = g[idx];
            let dz = dh * (h_prev[idx] - n);
            let dn = dh * (1.0 - z);
            want_hp[idx] = dh * z;
            let d_n_pre = dn * (1.0 - n * n);
            let d_z_pre = dz * z * (1.0 - z);
            let d_r_pre = d_n_pre * n_hh * r * (1.0 - r);
            want_ih[base + h] = d_r_pre;
            want_ih[base + hs + h] = d_z_pre;
            want_ih[base + 2 * hs + h] = d_n_pre;
            want_hh[base + h] = d_r_pre;
            want_hh[base + hs + h] = d_z_pre;
            want_hh[base + 2 * hs + h] = d_n_pre * r;
        }
    }
    assert_close(&got_ih, &want_ih, 1e-6, "gru ih");
    assert_close(&got_hh, &want_hh, 1e-6, "gru hh");
    assert_close(&got_hp, &want_hp, 1e-6, "gru h_prev");
}

// ── fused attention backward (multi-head → parallel branch) ──

#[test]
fn fused_attention_backward_parallel_matches_reference() {
    let (b, h, tq, tk, d) = (2, 3, 5, 7, 4);
    let q = seq(b * h * tq * d, 14);
    let k = seq(b * h * tk * d, 15);
    let v = seq(b * h * tk * d, 16);
    let go = seq(b * h * tq * d, 17);
    let scale = 0.5f32;

    for &causal in &[false, true] {
        let mut o = vec![0.0f32; b * h * tq * d];
        let mut want_q = vec![0.0f32; q.len()];
        let mut want_k = vec![0.0f32; k.len()];
        let mut want_v = vec![0.0f32; v.len()];
        for bh in 0..b * h {
            for i in 0..tq {
                let eff = if causal { (i + 1).min(tk) } else { tk };
                let qi = (bh * tq + i) * d;
                let scores: Vec<f32> = (0..eff)
                    .map(|j| {
                        let kj = (bh * tk + j) * d;
                        (0..d).map(|x| q[qi + x] * k[kj + x]).sum::<f32>() * scale
                    })
                    .collect();
                let p = softmax_rows(&scores, 1, eff);
                for j in 0..eff {
                    let kj = (bh * tk + j) * d;
                    for x in 0..d {
                        o[qi + x] += p[j] * v[kj + x];
                    }
                }
                let d_i: f32 = (0..d).map(|x| go[qi + x] * o[qi + x]).sum();
                for j in 0..eff {
                    let kj = (bh * tk + j) * d;
                    let ga: f32 = (0..d).map(|x| go[qi + x] * v[kj + x]).sum();
                    let gs = p[j] * (ga - d_i) * scale;
                    for x in 0..d {
                        want_v[kj + x] += p[j] * go[qi + x];
                        want_q[qi + x] += gs * k[kj + x];
                        want_k[kj + x] += gs * q[qi + x];
                    }
                }
            }
        }
        let f = FusedAttentionBackward::new(
            None,
            None,
            None,
            Tensor::from_vec(q.clone(), &[b, h, tq, d]).unwrap(),
            Tensor::from_vec(k.clone(), &[b, h, tk, d]).unwrap(),
            Tensor::from_vec(v.clone(), &[b, h, tk, d]).unwrap(),
            Tensor::from_vec(o, &[b, h, tq, d]).unwrap(),
            scale,
            causal,
        );
        let grads = f.apply(&Tensor::from_vec(go.clone(), &[b, h, tq, d]).unwrap());
        assert_close(
            &grads[0].as_ref().unwrap().to_vec(),
            &want_q,
            1e-5,
            "attn grad_q",
        );
        assert_close(
            &grads[1].as_ref().unwrap().to_vec(),
            &want_k,
            1e-5,
            "attn grad_k",
        );
        assert_close(
            &grads[2].as_ref().unwrap().to_vec(),
            &want_v,
            1e-5,
            "attn grad_v",
        );
    }
}
