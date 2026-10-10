//! Attach adapters across a small LLaMA: identity at attach, parameter count, and adapter-only training after freeze.
use axonml_llm::llama::{LLaMAConfig, LLaMAForCausalLM};
use axonml_nn::Module;
use axonml_tensor::Tensor;

fn main() {
    let cfg = LLaMAConfig {
        vocab_size: 97,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 3,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 500000.0,
        attention_dropout: 0.0,
        hidden_dropout: 0.0,
    };
    let mut model = LLaMAForCausalLM::new(&cfg);
    let ids = Tensor::from_vec((0..12u32).map(|i| (i * 7 + 3) % 97).collect(), &[1, 12]).unwrap();

    let before = model.model.forward_ids(&ids).data().to_vec();
    let base_params: usize = model.parameters().iter().map(|p| p.data().numel()).sum();
    let rank = 4;
    let ps = model.attach_lora(rank, 8.0);
    let after = model.model.forward_ids(&ids).data().to_vec();
    let ident = before.iter().zip(&after).all(|(a, b)| (a - b).abs() < 1e-6);
    let lora_params: usize = ps.iter().map(|p| p.data().numel()).sum();
    println!("identity at attach: {}", if ident { "YES" } else { "NO" });
    println!(
        "adapters: {} tensors, {lora_params} params vs {base_params} base ({:.2}%)",
        ps.len(),
        100.0 * lora_params as f32 / base_params as f32
    );
    println!(
        "named: {} entries, first = {}",
        model.lora_named_parameters().len(),
        model.lora_named_parameters()[0].0
    );

    model.freeze_lora_base();
    let h = model.model.forward_ids(&ids);
    let seed = Tensor::from_vec(vec![0.01f32; h.data().numel()], h.data().shape()).unwrap();
    h.backward_with_grad(&seed);
    let with_grad = ps.iter().filter(|p| p.grad().is_some()).count();
    let base_with_grad = model.model.layers[0]
        .self_attn
        .q_proj
        .weight
        .grad()
        .is_some();
    println!("adapter tensors with grad: {with_grad}/{}", ps.len());
    println!("frozen base q_proj has grad: {base_with_grad}");
    println!(
        "{}",
        if ident && with_grad == ps.len() && !base_with_grad {
            "MODULE WIRING CORRECT"
        } else {
            "PROBLEM"
        }
    );
}
