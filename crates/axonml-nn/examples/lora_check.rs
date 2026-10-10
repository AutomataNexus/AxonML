//! Low-rank adapter checks: identity at attach, gradients flow to A and B only, finite-difference agreement.
use axonml_autograd::Variable;
use axonml_nn::Module;
use axonml_nn::layers::linear::Linear;
use axonml_tensor::Tensor;

fn seeded(n: usize, s: &mut u64) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*s >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 0.4
        })
        .collect()
}

fn main() {
    let (m, din, dout, r) = (4usize, 6usize, 5usize, 2usize);
    let mut s = 3u64;
    let x = Tensor::from_vec(seeded(m * din, &mut s), &[m, din]).unwrap();
    let mut lin = Linear::with_bias(din, dout, false);
    lin.weight
        .variable()
        .set_data(Tensor::from_vec(seeded(dout * din, &mut s), &[dout, din]).unwrap());

    let before = lin
        .forward(&Variable::new(x.clone(), false))
        .data()
        .to_vec();
    let ps = lin.attach_lora(r, 8.0);
    let after = lin
        .forward(&Variable::new(x.clone(), false))
        .data()
        .to_vec();
    let ident = before.iter().zip(&after).all(|(a, b)| (a - b).abs() < 1e-7);
    println!("identity at attach: {}", if ident { "YES" } else { "NO" });

    // ── give A and B real values, then check the analytic gradient of B against finite differences ──
    ps[0]
        .variable()
        .set_data(Tensor::from_vec(seeded(r * din, &mut s), &[r, din]).unwrap());
    ps[1]
        .variable()
        .set_data(Tensor::from_vec(seeded(dout * r, &mut s), &[dout, r]).unwrap());
    let c: Vec<f32> = (0..m * dout).map(|i| 0.2 + 0.1 * (i % 4) as f32).collect();
    let loss_of = |l: &Linear| -> f32 {
        let y = l.forward(&Variable::new(x.clone(), false));
        y.data().to_vec().iter().zip(&c).map(|(a, b)| a * b).sum()
    };
    let v = Variable::new(x.clone(), false);
    let y = lin.forward(&v);
    y.backward_with_grad(&Tensor::from_vec(c.clone(), &[m, dout]).unwrap());
    let ga = ps[0].grad().expect("no grad on A").to_vec();
    let gb = ps[1].grad().expect("no grad on B").to_vec();
    println!(
        "base weight grad present (unfrozen): {}",
        lin.weight.grad().is_some()
    );
    {
        let mut frozen = lin.clone();
        frozen.freeze_base();
        let fv = Variable::new(x.clone(), false);
        let fy = frozen.forward(&fv);
        fy.backward_with_grad(&Tensor::from_vec(c.clone(), &[m, dout]).unwrap());
        let lp = frozen.lora_parameters();
        println!(
            "after freeze_base — base grad: {}, adapter grads: {}",
            frozen.weight.grad().is_some(),
            lp.iter().all(|p| p.grad().is_some())
        );
        let fout = frozen
            .forward(&Variable::new(x.clone(), false))
            .data()
            .to_vec();
        let uout = lin
            .forward(&Variable::new(x.clone(), false))
            .data()
            .to_vec();
        println!(
            "frozen output matches unfrozen: {}",
            fout.iter().zip(&uout).all(|(a, b)| (a - b).abs() < 1e-6)
        );
    }

    let eps = 1e-3f32;
    let base_b = ps[1].data().to_vec();
    let mut worst = 0f32;
    for j in 0..base_b.len() {
        let mut up = base_b.clone();
        up[j] += eps;
        let mut dn = base_b.clone();
        dn[j] -= eps;
        ps[1]
            .variable()
            .set_data(Tensor::from_vec(up, &[dout, r]).unwrap());
        let lu = loss_of(&lin);
        ps[1]
            .variable()
            .set_data(Tensor::from_vec(dn, &[dout, r]).unwrap());
        let ld = loss_of(&lin);
        let fd = (lu - ld) / (2.0 * eps);
        let den = fd.abs().max(gb[j].abs()).max(1e-6);
        worst = worst.max((fd - gb[j]).abs() / den);
    }
    ps[1]
        .variable()
        .set_data(Tensor::from_vec(base_b, &[dout, r]).unwrap());
    println!(
        "A grad norm {:.4e}, B grad norm {:.4e}",
        ga.iter().map(|v| v * v).sum::<f32>().sqrt(),
        gb.iter().map(|v| v * v).sum::<f32>().sqrt()
    );
    println!(
        "worst rel err on B vs finite difference = {worst:.6} -> {}",
        if worst < 1e-2 {
            "GRADIENT CORRECT"
        } else {
            "MISMATCH"
        }
    );
}
