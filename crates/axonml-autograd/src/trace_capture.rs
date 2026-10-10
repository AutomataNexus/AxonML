//! Forward-op recorder: captures the model's REAL forward as it runs.
//!
//! There is no second declaration and no DSL. You wrap the actual `model.forward(x)` in
//! [`capture`]; every fusable elementwise op it executes appends a [`RecordedOp`] carrying its
//! kind, its scalar operand (kept here because the backward graph discards forward-only constants —
//! `AddScalarBackward` is identity and loses its `+c`), and its input node ids. Non-elementwise ops
//! (matmul, conv, …) record nothing, so they become natural chain boundaries: any input id with no
//! producing op is a graph leaf. The recording is later lowered to a JIT graph and fused.
//!
//! # File
//! `crates/axonml-autograd/src/trace_capture.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060

use std::cell::RefCell;

use crate::graph::NodeId;
use crate::variable::Variable;

// ── recorded op ──

/// One fusable elementwise step, with its forward operand preserved.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TraceKind {
    /// `x + scalar`.
    AddScalar(f32),
    /// `x * scalar`.
    MulScalar(f32),
    /// `x - scalar`.
    SubScalar(f32),
    /// `-x`.
    Neg,
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
    /// `x^exponent`.
    Pow(f32),
    /// `min(max(x, lo), hi)`.
    Clamp(f32, f32),
    /// Binary: `x + b[i]`.
    AddTensor,
    /// Binary: `x * b[i]`.
    MulTensor,
    /// Binary: `x - b[i]`.
    SubTensor,
}

impl TraceKind {
    /// Number of tensor inputs this op consumes (1 for unary/scalar, 2 for binary).
    #[must_use]
    pub fn arity(self) -> usize {
        match self {
            TraceKind::AddTensor | TraceKind::MulTensor | TraceKind::SubTensor => 2,
            _ => 1,
        }
    }
}

/// A single recorded forward op: which node it produced, what it did, and its input node ids.
#[derive(Debug, Clone)]
pub struct RecordedOp {
    /// Graph node id this op produced.
    pub out: NodeId,
    /// The elementwise op and its scalar operand(s).
    pub kind: TraceKind,
    /// Input node ids (1 for unary/scalar, 2 for binary), in operand order.
    pub inputs: Vec<NodeId>,
}

/// The ordered list of fusable ops seen during one [`capture`], in forward execution order.
#[derive(Debug, Default, Clone)]
pub struct Recording {
    /// Fusable ops in forward execution order.
    pub ops: Vec<RecordedOp>,
}

// ── thread-local recorder ──

thread_local! {
    static RECORDER: RefCell<Option<Recording>> = const { RefCell::new(None) };
}

/// Whether a capture is currently active on this thread.
#[must_use]
pub fn is_recording() -> bool {
    RECORDER.with(|r| r.borrow().is_some())
}

pub(crate) fn record(out: NodeId, kind: TraceKind, inputs: &[NodeId]) {
    RECORDER.with(|r| {
        if let Some(rec) = r.borrow_mut().as_mut() {
            rec.ops.push(RecordedOp {
                out,
                kind,
                inputs: inputs.to_vec(),
            });
        }
    });
}

/// Run `f` (the real forward) with recording on and return what executed plus the output node id.
///
/// Save/restore of the previous recorder makes nested captures safe. The returned `NodeId` is the
/// output `Variable`'s graph node — the lowering pass roots the fused graph there.
pub fn capture<F>(f: F) -> (Recording, Option<NodeId>)
where
    F: FnOnce() -> Variable,
{
    let prev = RECORDER.with(|r| r.borrow_mut().replace(Recording::default()));
    let out = f();
    let rec = RECORDER
        .with(|r| std::mem::replace(&mut *r.borrow_mut(), prev))
        .unwrap_or_default();
    (rec, out.node_id())
}

// ── tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captures_the_real_forward_chain() {
        use axonml_tensor::Tensor;
        let x = Variable::new(Tensor::from_vec(vec![0.5f32; 4], &[4]).unwrap(), true);
        let (rec, out) = capture(|| x.mul_scalar(2.0).add_scalar(1.0).sigmoid().tanh());
        assert!(out.is_some());
        let kinds: Vec<TraceKind> = rec.ops.iter().map(|o| o.kind).collect();
        assert_eq!(
            kinds,
            vec![
                TraceKind::MulScalar(2.0),
                TraceKind::AddScalar(1.0),
                TraceKind::Sigmoid,
                TraceKind::Tanh,
            ],
            "recorder must preserve op order and scalar operands"
        );
        for w in rec.ops.windows(2) {
            assert_eq!(w[1].inputs[0], w[0].out, "chain must be node-linked");
        }
        assert!(!is_recording(), "recorder cleared after capture");
    }

    #[test]
    fn binary_op_records_two_inputs() {
        use axonml_tensor::Tensor;
        let a = Variable::new(Tensor::from_vec(vec![1.0f32; 3], &[3]).unwrap(), true);
        let b = Variable::new(Tensor::from_vec(vec![2.0f32; 3], &[3]).unwrap(), true);
        let (rec, _) = capture(|| a.sigmoid().mul(&b));
        let last = rec.ops.last().unwrap();
        assert_eq!(last.kind, TraceKind::MulTensor);
        assert_eq!(last.inputs.len(), 2, "binary op records both tensor inputs");
    }
}
