//! Runtime-JIT elementwise chain fusion.
//!
//! A sequence of unary / scalar elementwise ops applied to one tensor is normally one kernel launch
//! per op, each reading and writing the full tensor to global memory. For a long chain that is
//! memory-bound: N launches, 2N global passes. [`ChainOp`] describes such a chain; [`fuse_unary`]
//! generates a single CUDA C kernel that loads each element once, runs the whole chain in registers,
//! and writes once — compiled (NVRTC) and cached per unique chain signature, then dispatched for free.
//!
//! This is where the "2x for memory-bound elementwise" lives: it is the CHAIN that wins, not a
//! pairwise op (a 2-op fusion was measured slower — launch-bound, not memory-bound).

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use crate::alloc_prelude::*;

/// One step in a unary elementwise chain. Each maps `x` (the running value) to a new value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChainOp {
    /// `x + c`.
    AddScalar(f32),
    /// `x * c`.
    MulScalar(f32),
    /// `-x`.
    Neg,
    /// `1 / x`.
    Recip,
    /// `x * x`.
    Square,
    /// `sqrt(x)`.
    Sqrt,
    /// `exp(x)`.
    Exp,
    /// `ln(x)`.
    Ln,
    /// `max(x, 0)`.
    Relu,
    /// `1 / (1 + exp(-x))`.
    Sigmoid,
    /// `tanh(x)`.
    Tanh,
    /// `x^exponent` via C99 `powf` (sign-correct, unlike the older PTX pow).
    Pow(f32),
    /// `min(max(x, lo), hi)`.
    Clamp(f32, f32),
    /// Binary: `x + b[i]` (second input tensor). Only valid in a binary chain.
    AddTensor,
    /// Binary: `x * b[i]`.
    MulTensor,
    /// Binary: `x - b[i]`.
    SubTensor,
}

impl ChainOp {
    fn emit(self, v: &str) -> String {
        match self {
            ChainOp::AddScalar(s) => format!("{v} + {s:e}f"),
            ChainOp::MulScalar(s) => format!("{v} * {s:e}f"),
            ChainOp::Neg => format!("-{v}"),
            ChainOp::Recip => format!("1.0f / {v}"),
            ChainOp::Square => format!("{v} * {v}"),
            ChainOp::Sqrt => format!("sqrtf({v})"),
            ChainOp::Exp => format!("expf({v})"),
            ChainOp::Ln => format!("logf({v})"),
            ChainOp::Relu => format!("fmaxf({v}, 0.0f)"),
            ChainOp::Sigmoid => format!("1.0f / (1.0f + expf(-({v})))"),
            ChainOp::Tanh => format!("tanhf({v})"),
            ChainOp::Pow(e) => format!("powf({v}, {e:e}f)"),
            ChainOp::Clamp(lo, hi) => format!("fminf(fmaxf({v}, {lo:e}f), {hi:e}f)"),
            ChainOp::AddTensor => format!("{v} + b[i]"),
            ChainOp::MulTensor => format!("{v} * b[i]"),
            ChainOp::SubTensor => format!("{v} - b[i]"),
        }
    }

    fn sig(self) -> String {
        match self {
            ChainOp::AddScalar(s) => format!("as{}", s.to_bits()),
            ChainOp::MulScalar(s) => format!("ms{}", s.to_bits()),
            ChainOp::Neg => "neg".into(),
            ChainOp::Recip => "rcp".into(),
            ChainOp::Square => "sq".into(),
            ChainOp::Sqrt => "sqrt".into(),
            ChainOp::Exp => "exp".into(),
            ChainOp::Ln => "ln".into(),
            ChainOp::Relu => "relu".into(),
            ChainOp::Sigmoid => "sig".into(),
            ChainOp::Tanh => "tanh".into(),
            ChainOp::Pow(e) => format!("pw{}", e.to_bits()),
            ChainOp::Clamp(lo, hi) => format!("cl{}_{}", lo.to_bits(), hi.to_bits()),
            ChainOp::AddTensor => "addt".into(),
            ChainOp::MulTensor => "mult".into(),
            ChainOp::SubTensor => "subt".into(),
        }
    }

    /// Whether this op references the second input tensor.
    #[must_use]
    pub fn is_binary(self) -> bool {
        matches!(
            self,
            ChainOp::AddTensor | ChainOp::MulTensor | ChainOp::SubTensor
        )
    }
}

fn codegen(chain: &[ChainOp]) -> (String, String) {
    let mut key = String::from("fc_");
    let mut expr = String::from("x");
    for op in chain {
        key.push_str(&op.sig());
        key.push('_');
        expr = op.emit(&expr);
        // wrap so precedence never bites across steps
        expr = format!("({expr})");
    }
    (key, expr)
}

/// Codegen entry point used by `Tensor::fuse_unary_chain` in `cuda_ops`.
/// Returns the cache key and the C expression over `x` (the input element);
/// the CUDA backend wraps it in the fixed `fused_chain(in, out, n)` entry.
#[must_use]
pub fn chain_codegen(chain: &[ChainOp]) -> (String, String) {
    codegen(chain)
}

fn codegen_binary(chain: &[ChainOp]) -> (String, String) {
    let mut key = String::from("fcb_");
    let mut expr = String::from("x");
    for op in chain {
        key.push_str(&op.sig());
        key.push('_');
        expr = op.emit(&expr);
        expr = format!("({expr})");
    }
    (key, expr)
}

/// Codegen entry point for a two-input chain (uses `AddTensor`/`MulTensor`/`SubTensor` on `b`).
#[must_use]
pub fn chain_codegen_binary(chain: &[ChainOp]) -> (String, String) {
    codegen_binary(chain)
}

// ── tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codegen_is_deterministic_and_keyed() {
        let a = codegen(&[ChainOp::MulScalar(2.0), ChainOp::Sigmoid, ChainOp::Sqrt]);
        let b = codegen(&[ChainOp::MulScalar(2.0), ChainOp::Sigmoid, ChainOp::Sqrt]);
        assert_eq!(a.0, b.0, "same chain -> same cache key");
        assert_ne!(
            a.0,
            codegen(&[ChainOp::Sigmoid]).0,
            "different chain -> different key"
        );
        assert!(a.1.starts_with('(') && a.1.contains("sqrtf"));
    }

    #[test]
    fn binary_ops_flagged_and_keyed() {
        assert!(ChainOp::MulTensor.is_binary() && ChainOp::AddTensor.is_binary());
        assert!(!ChainOp::Sigmoid.is_binary() && !ChainOp::MulScalar(2.0).is_binary());
        let (k, expr) = codegen_binary(&[ChainOp::Sigmoid, ChainOp::MulTensor]);
        assert!(k.starts_with("fcb_") && expr.contains("b[i]") && expr.contains('x'));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn fused_binary_matches_op_by_op() {
        use crate::Tensor;
        use axonml_core::device::Device;
        if Tensor::from_vec(vec![0.0f32; 1], &[1])
            .unwrap()
            .to_device(Device::Cuda(0))
            .is_err()
        {
            return;
        }
        let n = 4096usize;
        let av: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 - 48.0) / 40.0).collect();
        let bv: Vec<f32> = (0..n).map(|i| ((i % 53) as f32 - 26.0) / 25.0).collect();
        let a = Tensor::from_vec(av, &[n])
            .unwrap()
            .to_device(Device::Cuda(0))
            .unwrap();
        let b = Tensor::from_vec(bv, &[n])
            .unwrap()
            .to_device(Device::Cuda(0))
            .unwrap();
        let chain = [ChainOp::Sigmoid, ChainOp::MulTensor];
        let refv = a
            .clone()
            .sigmoid()
            .mul(&b)
            .unwrap()
            .to_device(Device::Cpu)
            .unwrap()
            .to_vec();
        let fused = a
            .fuse_binary_chain(&b, &chain)
            .unwrap()
            .to_device(Device::Cpu)
            .unwrap()
            .to_vec();
        let md = refv
            .iter()
            .zip(&fused)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(md < 1e-5, "fused binary chain diverges: {md}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn fused_matches_op_by_op() {
        use crate::Tensor;
        use axonml_core::device::Device;
        if Tensor::from_vec(vec![0.0f32; 1], &[1])
            .unwrap()
            .to_device(Device::Cuda(0))
            .is_err()
        {
            return;
        }
        let n = 4096usize;
        let v: Vec<f32> = (0..n)
            .map(|i| ((i % 97) as f32 - 48.0) / 40.0 + 0.5)
            .collect();
        let x = Tensor::from_vec(v, &[n])
            .unwrap()
            .to_device(Device::Cuda(0))
            .unwrap();
        let chain = [
            ChainOp::MulScalar(2.0),
            ChainOp::AddScalar(1.0),
            ChainOp::Sigmoid,
            ChainOp::Sqrt,
        ];
        let refv = x
            .clone()
            .mul_scalar(2.0)
            .add_scalar(1.0)
            .sigmoid()
            .sqrt()
            .to_device(Device::Cpu)
            .unwrap()
            .to_vec();
        let fused = x
            .fuse_unary_chain(&chain)
            .unwrap()
            .to_device(Device::Cpu)
            .unwrap()
            .to_vec();
        let md = refv
            .iter()
            .zip(&fused)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(md < 1e-5, "fused chain diverges: {md}");
    }
}
