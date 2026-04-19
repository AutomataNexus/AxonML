//! Isolate the per-call cost of the 2D matmul that Linear's backward uses.
//!
//! Qwen3-0.6B: bs*seq = 2048, hidden = 1024, inter = 3072.
//! Linear forward: [2048, 1024] @ [1024, 3072] → [2048, 3072]
//! MatMulBackward:
//!    grad_lhs = go[2048,3072] @ weight[3072,1024]    (go @ rhs_t)
//!    grad_rhs = lhs_t[1024,2048] @ go[2048,3072]     (lhs_t @ go)

use std::time::Instant;

use axonml_core::Device;
use axonml_tensor::Tensor;

fn main() {
    let m = 2048;
    let k = 1024;
    let n = 3072;
    let device = Device::Cuda(0);

    let a = mkrand(&[m, k], 0.1, device);
    let b = mkrand(&[k, n], 0.2, device);
    let go = mkrand(&[m, n], 0.01, device);

    // warm
    for _ in 0..5 {
        let _ = a.matmul(&b).unwrap();
    }

    let n_iter = 100;

    // Forward: [m,k] @ [k,n]
    let t0 = Instant::now();
    for _ in 0..n_iter {
        let c = a.matmul(&b).unwrap();
        std::hint::black_box(c);
    }
    let _ = a.matmul(&b).unwrap().to_vec();
    println!(
        "2D forward [m={m},k={k}] @ [k,n={n}]         {:>7.2} µs/call",
        per_us(t0, n_iter + 1)
    );

    // grad_lhs = go @ b.t()    — b is contiguous [k,n], b.t() is last2-transposed view
    let t1 = Instant::now();
    for _ in 0..n_iter {
        let bt = b.t().unwrap();
        let gl = go.matmul(&bt).unwrap();
        std::hint::black_box(gl);
    }
    let _ = go.matmul(&b.t().unwrap()).unwrap().to_vec();
    println!(
        "grad_lhs = go @ b.t()                          {:>7.2} µs/call  [last2-trans b]",
        per_us(t1, n_iter + 1)
    );

    // grad_rhs = a.t() @ go    — a is contiguous [m,k], a.t() is last2-transposed view
    let t2 = Instant::now();
    for _ in 0..n_iter {
        let at = a.t().unwrap();
        let gr = at.matmul(&go).unwrap();
        std::hint::black_box(gr);
    }
    let _ = a.t().unwrap().matmul(&go).unwrap().to_vec();
    println!(
        "grad_rhs = a.t() @ go                          {:>7.2} µs/call  [last2-trans a]",
        per_us(t2, n_iter + 1)
    );

    // Full MatMulBackward-shaped
    let t3 = Instant::now();
    for _ in 0..n_iter {
        let bt = b.t().unwrap();
        let at = a.t().unwrap();
        let gl = go.matmul(&bt).unwrap();
        let gr = at.matmul(&go).unwrap();
        std::hint::black_box((gl, gr));
    }
    let _ = go.matmul(&b.t().unwrap()).unwrap().to_vec();
    println!(
        "\nFULL 2D MatMulBackward-shaped              {:>7.2} µs/call",
        per_us(t3, n_iter + 1)
    );
}

fn mkrand(shape: &[usize], seed: f32, dev: Device) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| seed + ((i % 17) as f32) * 0.003).collect();
    Tensor::from_vec(data, shape)
        .unwrap()
        .to_device(dev)
        .unwrap()
}

fn per_us(start: Instant, n: usize) -> f64 {
    start.elapsed().as_micros() as f64 / n as f64
}
