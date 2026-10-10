//! Finite-difference check of the RMSNorm weight gradient.
use axonml_autograd::Variable;
use axonml_llm::llama::RMSNorm;
use axonml_tensor::Tensor;

fn loss_of(n: &RMSNorm, x: &Tensor<f32>) -> f32 {
    let v = Variable::new(x.clone(), false);
    let y = n.forward(&v);
    // scalar loss = sum(y * c) with fixed c, so dL/dy = c
    let c: Vec<f32> = (0..y.data().numel())
        .map(|i| 0.3 + 0.1 * (i % 5) as f32)
        .collect();
    y.data().to_vec().iter().zip(&c).map(|(a, b)| a * b).sum()
}

fn main() {
    let (m, d) = (4usize, 6usize);
    let xv: Vec<f32> = (0..m * d)
        .map(|i| ((i * 37 % 23) as f32 - 11.0) / 7.0)
        .collect();
    let x = Tensor::from_vec(xv, &[m, d]).unwrap();

    let mut n = RMSNorm::new(d, 1e-6);
    for (i, w) in (0..d).enumerate() {
        let _ = (i, w);
    }
    let p = n.make_trainable();

    // analytic gradient
    let v = Variable::new(x.clone(), true);
    let y = n.forward(&v);
    let c: Vec<f32> = (0..y.data().numel())
        .map(|i| 0.3 + 0.1 * (i % 5) as f32)
        .collect();
    let seed = Tensor::from_vec(c, y.data().shape()).unwrap();
    y.backward_with_grad(&seed);
    let analytic = p.variable().grad().expect("no weight grad").to_vec();

    // finite differences
    let eps = 1e-3f32;
    let base = p.data().to_vec();
    let mut fd = vec![0f32; d];
    for j in 0..d {
        let mut up = base.clone();
        up[j] += eps;
        let mut dn = base.clone();
        dn[j] -= eps;
        p.variable().set_data(Tensor::from_vec(up, &[d]).unwrap());
        let lu = loss_of(&n, &x);
        p.variable().set_data(Tensor::from_vec(dn, &[d]).unwrap());
        let ld = loss_of(&n, &x);
        fd[j] = (lu - ld) / (2.0 * eps);
    }
    p.variable().set_data(Tensor::from_vec(base, &[d]).unwrap());

    let mut worst = 0f32;
    for j in 0..d {
        let denom = analytic[j].abs().max(fd[j].abs()).max(1e-6);
        worst = worst.max((analytic[j] - fd[j]).abs() / denom);
    }
    println!(
        "analytic {:?}",
        analytic
            .iter()
            .map(|v| (v * 1e4).round() / 1e4)
            .collect::<Vec<_>>()
    );
    println!(
        "finitediff {:?}",
        fd.iter()
            .map(|v| (v * 1e4).round() / 1e4)
            .collect::<Vec<_>>()
    );
    println!(
        "worst relative error = {worst:.6}  -> {}",
        if worst < 1e-2 {
            "GRADIENT CORRECT"
        } else {
            "MISMATCH"
        }
    );
}
