//! Train All Model Types — End-to-End Pipeline Verification
//!
//! # File
//! `crates/axonml-nn/examples/train_all_types.rs`
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Updated
//! March 9, 2026
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use axonml_autograd::Variable;
use axonml_nn::layers::*;
use axonml_nn::*;
use axonml_optim::Optimizer;
use axonml_tensor::Tensor;
use std::time::Instant;

const STEPS: usize = 20;
const LR: f32 = 1e-3;

fn rand_tensor(shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 0.7123 + 0.3).sin() * 0.5)
        .collect();
    Tensor::from_vec(data, shape).unwrap()
}

fn rand_target(shape: &[usize]) -> Variable {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 1.234 + 0.9).cos() * 0.3)
        .collect();
    Variable::new(Tensor::from_vec(data, shape).unwrap(), false)
}

fn train_loop(
    name: &str,
    params: Vec<Parameter>,
    forward_fn: &dyn Fn() -> Variable,
    target: &Variable,
) -> bool {
    let t0 = Instant::now();
    let mse = MSELoss::new();
    let mut optimizer = axonml_optim::Adam::new(params, LR);

    let mut first_loss = 0.0f32;
    let mut last_loss = 0.0f32;

    for step in 0..STEPS {
        optimizer.zero_grad();
        let output = forward_fn();
        let loss = mse.compute(&output, target);
        let loss_val = loss.data().to_vec()[0];

        if step == 0 {
            first_loss = loss_val;
        }
        last_loss = loss_val;

        if loss.requires_grad() {
            loss.backward();
            optimizer.step();
        } else {
            println!("  {:24} | NO GRAD — cannot train", name);
            return false;
        }
    }

    let elapsed = t0.elapsed().as_millis();
    let improved = last_loss < first_loss;
    let pct = if first_loss > 0.0 {
        (1.0 - last_loss / first_loss) * 100.0
    } else {
        0.0
    };

    println!(
        "  {:24} | {:.6} → {:.6} | {:+6.1}% | {:5}ms | {}",
        name,
        first_loss,
        last_loss,
        pct,
        elapsed,
        if improved { "PASS" } else { "WARN" }
    );

    improved
}

fn main() {
    println!("=== AxonML Training Pipeline — All Model Types ===");
    println!("  Steps: {STEPS} | LR: {LR} | Optimizer: Adam\n");
    println!(
        "  {:24} | {:26} | {:>7} | {:>6} | {}",
        "Model", "Loss (start → end)", "Δ%", "Time", "Status"
    );
    println!(
        "  {:-<24}-+-{:-<26}-+-{:-<7}-+-{:-<6}-+-{:-<6}",
        "", "", "", "", ""
    );

    let mut results: Vec<(&str, bool)> = Vec::new();

    // 1. Linear (MLP)
    {
        let l1 = Linear::new(16, 32);
        let l2 = Linear::new(32, 8);
        let input = Variable::new(rand_tensor(&[4, 16]), true);
        let target = rand_target(&[4, 8]);
        let params = [l1.parameters(), l2.parameters()].concat();
        let ok = train_loop(
            "Linear (MLP)",
            params,
            &|| l2.forward(&l1.forward(&input).relu()),
            &target,
        );
        results.push(("Linear (MLP)", ok));
    }

    // 2. Conv1d
    {
        let conv = Conv1d::new(3, 8, 3);
        let fc = Linear::new(8 * 30, 10);
        let input = Variable::new(rand_tensor(&[2, 3, 32]), true);
        let target = rand_target(&[2, 10]);
        let params = [conv.parameters(), fc.parameters()].concat();
        let ok = train_loop(
            "Conv1d",
            params,
            &|| {
                let x = conv.forward(&input).relu();
                let x = x.reshape(&[2, 8 * 30]);
                fc.forward(&x)
            },
            &target,
        );
        results.push(("Conv1d", ok));
    }

    // 3. Conv2d
    {
        let conv = Conv2d::new(3, 8, 3);
        let fc = Linear::new(8 * 14 * 14, 10);
        let input = Variable::new(rand_tensor(&[2, 3, 16, 16]), true);
        let target = rand_target(&[2, 10]);
        let params = [conv.parameters(), fc.parameters()].concat();
        let ok = train_loop(
            "Conv2d",
            params,
            &|| {
                let x = conv.forward(&input).relu();
                let x = x.reshape(&[2, 8 * 14 * 14]);
                fc.forward(&x)
            },
            &target,
        );
        results.push(("Conv2d", ok));
    }

    // 4. ConvTranspose2d
    {
        let conv = ConvTranspose2d::new(8, 3, 3);
        let input = Variable::new(rand_tensor(&[2, 8, 8, 8]), true);
        let target = rand_target(&[2, 3, 10, 10]);
        let params = conv.parameters();
        let ok = train_loop("ConvTranspose2d", params, &|| conv.forward(&input), &target);
        results.push(("ConvTranspose2d", ok));
    }

    // 5. RNN
    {
        let rnn = RNN::new(8, 16, 1);
        let fc = Linear::new(16, 4);
        let input = Variable::new(rand_tensor(&[2, 10, 8]), true);
        let target = rand_target(&[2, 4]);
        let params = [rnn.parameters(), fc.parameters()].concat();
        let ok = train_loop(
            "RNN",
            params,
            &|| {
                // forward returns [batch, seq, hidden], take last step
                let out = rnn.forward(&input);
                let last = out.narrow(1, 9, 1).reshape(&[2, 16]);
                fc.forward(&last)
            },
            &target,
        );
        results.push(("RNN", ok));
    }

    // 6. LSTM
    {
        let lstm = LSTM::new(8, 16, 1);
        let fc = Linear::new(16, 4);
        let input = Variable::new(rand_tensor(&[2, 10, 8]), true);
        let target = rand_target(&[2, 4]);
        let params = [lstm.parameters(), fc.parameters()].concat();
        let ok = train_loop(
            "LSTM",
            params,
            &|| {
                let out = lstm.forward(&input);
                let last = out.narrow(1, 9, 1).reshape(&[2, 16]);
                fc.forward(&last)
            },
            &target,
        );
        results.push(("LSTM", ok));
    }

    // 7. GRU
    {
        let gru = GRU::new(8, 16, 1);
        let fc = Linear::new(16, 4);
        let input = Variable::new(rand_tensor(&[2, 10, 8]), true);
        let target = rand_target(&[2, 4]);
        let params = [gru.parameters(), fc.parameters()].concat();
        let ok = train_loop(
            "GRU",
            params,
            &|| {
                let last = gru.forward_last(&input);
                fc.forward(&last)
            },
            &target,
        );
        results.push(("GRU", ok));
    }

    // 8. TransformerEncoder
    {
        let encoder = TransformerEncoder::new(16, 2, 32, 2);
        let fc = Linear::new(16, 4);
        let input = Variable::new(rand_tensor(&[2, 8, 16]), true);
        let target = rand_target(&[2, 4]);
        let params = [encoder.parameters(), fc.parameters()].concat();
        let ok = train_loop(
            "TransformerEncoder",
            params,
            &|| {
                let enc = encoder.forward(&input);
                // Mean pool over sequence
                let pooled = enc.mean_dim(1, false);
                fc.forward(&pooled)
            },
            &target,
        );
        results.push(("TransformerEncoder", ok));
    }

    // 9. TransformerDecoder
    {
        let decoder = TransformerDecoder::new(16, 2, 32, 2);
        let memory = Variable::new(rand_tensor(&[2, 8, 16]), false);
        let tgt_input = Variable::new(rand_tensor(&[2, 6, 16]), true);
        let target = rand_target(&[2, 6, 16]);
        let params = decoder.parameters();
        let ok = train_loop(
            "TransformerDecoder",
            params,
            &|| decoder.forward_with_memory(&tgt_input, &memory, None, None),
            &target,
        );
        results.push(("TransformerDecoder", ok));
    }

    // 10. Seq2SeqTransformer
    {
        let s2s = Seq2SeqTransformer::new(16, 2, 2, 2, 32);
        let src = Variable::new(rand_tensor(&[2, 8, 16]), true);
        let tgt = Variable::new(rand_tensor(&[2, 6, 16]), true);
        let target = rand_target(&[2, 6, 16]);
        let params = s2s.parameters();
        let ok = train_loop(
            "Seq2SeqTransformer",
            params,
            &|| s2s.forward_seq2seq(&src, &tgt, None, None, None),
            &target,
        );
        results.push(("Seq2SeqTransformer", ok));
    }

    // 11. MultiHeadAttention
    {
        let mha = MultiHeadAttention::new(16, 2);
        let input = Variable::new(rand_tensor(&[2, 8, 16]), true);
        let target = rand_target(&[2, 8, 16]);
        let params = mha.parameters();
        let ok = train_loop(
            "MultiHeadAttention",
            params,
            &|| mha.forward(&input),
            &target,
        );
        results.push(("MultiHeadAttention", ok));
    }

    // 12. CrossAttention
    {
        let ca = CrossAttention::new(16, 2);
        let query = Variable::new(rand_tensor(&[2, 6, 16]), true);
        let kv = Variable::new(rand_tensor(&[2, 8, 16]), false);
        let target = rand_target(&[2, 6, 16]);
        let params = ca.parameters();
        let ok = train_loop(
            "CrossAttention",
            params,
            &|| ca.cross_attention(&query, &kv, None),
            &target,
        );
        results.push(("CrossAttention", ok));
    }

    // 13. Embedding
    {
        let emb = Embedding::new(100, 16);
        let fc = Linear::new(16, 4);
        let indices: Vec<f32> = vec![5.0, 12.0, 3.0, 45.0, 8.0, 22.0, 71.0, 1.0];
        let input = Variable::new(Tensor::from_vec(indices, &[2, 4]).unwrap(), false);
        let target = rand_target(&[2, 4]);
        let params = [emb.parameters(), fc.parameters()].concat();
        let ok = train_loop(
            "Embedding",
            params,
            &|| {
                let e = emb.forward(&input); // [2, 4, 16]
                let pooled = e.mean_dim(1, false); // [2, 16] — mean over token dim
                fc.forward(&pooled)
            },
            &target,
        );
        results.push(("Embedding", ok));
    }

    // 14. BatchNorm1d
    {
        let linear = Linear::new(16, 16);
        let bn = BatchNorm1d::new(16);
        let input = Variable::new(rand_tensor(&[4, 16]), true);
        let target = rand_target(&[4, 16]);
        let params = [linear.parameters(), bn.parameters()].concat();
        let ok = train_loop(
            "BatchNorm1d",
            params,
            &|| bn.forward(&linear.forward(&input)),
            &target,
        );
        results.push(("BatchNorm1d", ok));
    }

    // 15. LayerNorm
    {
        let linear = Linear::new(16, 16);
        let ln = LayerNorm::new(vec![16]);
        let input = Variable::new(rand_tensor(&[4, 16]), true);
        let target = rand_target(&[4, 16]);
        let params = [linear.parameters(), ln.parameters()].concat();
        let ok = train_loop(
            "LayerNorm",
            params,
            &|| ln.forward(&linear.forward(&input)),
            &target,
        );
        results.push(("LayerNorm", ok));
    }

    // 16. GroupNorm + Conv2d
    {
        let conv = Conv2d::new(3, 8, 3);
        let gn = GroupNorm::new(4, 8);
        let fc = Linear::new(8 * 14 * 14, 4);
        let input = Variable::new(rand_tensor(&[2, 3, 16, 16]), true);
        let target = rand_target(&[2, 4]);
        let params = [conv.parameters(), gn.parameters(), fc.parameters()].concat();
        let ok = train_loop(
            "GroupNorm + Conv2d",
            params,
            &|| {
                let x = gn.forward(&conv.forward(&input)).relu();
                let x = x.reshape(&[2, 8 * 14 * 14]);
                fc.forward(&x)
            },
            &target,
        );
        results.push(("GroupNorm + Conv2d", ok));
    }

    // 17. MaxPool2d + Conv2d → FC
    {
        let conv = Conv2d::new(3, 8, 3);
        let pool = MaxPool2d::new(2);
        let fc = Linear::new(8 * 7 * 7, 10);
        let input = Variable::new(rand_tensor(&[2, 3, 16, 16]), true);
        let target = rand_target(&[2, 10]);
        let params = [conv.parameters(), fc.parameters()].concat();
        let ok = train_loop(
            "Conv→MaxPool→FC",
            params,
            &|| {
                let x = pool.forward(&conv.forward(&input).relu());
                let x = x.reshape(&[2, 8 * 7 * 7]);
                fc.forward(&x)
            },
            &target,
        );
        results.push(("Conv→MaxPool→FC", ok));
    }

    // =========================================================================
    // Summary
    // =========================================================================
    println!();
    let passed = results.iter().filter(|(_, ok)| *ok).count();
    let failed = results.iter().filter(|(_, ok)| !*ok).count();
    let total = results.len();

    println!("=== Results: {passed}/{total} passed, {failed} failed ===");
    if failed > 0 {
        println!("\nFailed:");
        for (name, ok) in &results {
            if !ok {
                println!("  FAIL: {name}");
            }
        }
    } else {
        println!("All {total} model types train successfully with gradient flow!");
    }
}
