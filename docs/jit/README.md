# axonml-jit Documentation

> JIT compilation, tracing, and graph optimization for AxonML tensor operations.

## Overview

`axonml-jit` is AxonML's just-in-time compiler. It traces tensor
computations into a typed IR, runs a six-pass optimizer over the graph, and
compiles the result through a Cranelift-backed code generator. A
higher-level `torch.compile`-style facade (`compile_fn`, `compile_graph`,
`CompiledModel`, `LazyCompiled`) ties it all together. Compiled functions
are cached by graph hash for reuse.

## Pipeline

```
+---------------------------------------------------------------+
|                          User Code                            |
|       trace(|t| { a.add(&b).relu() })                         |
+---------------------------------------------------------------+
|                      Tracer (`trace`)                         |
|       Records operations into a computation graph             |
+---------------------------------------------------------------+
|                Intermediate Representation (`ir`)             |
|       Graph, Node, NodeId, Op, Shape, DataType                |
+---------------------------------------------------------------+
|                   Optimizer (`optimize`)                      |
|       Six passes: constant folding, DCE, CSE,                 |
|       algebraic simplification, elementwise fusion, strength  |
+---------------------------------------------------------------+
|                 Function Cache (`cache`)                      |
|       LRU hash-indexed reuse of compiled functions            |
+---------------------------------------------------------------+
|                  Code Generator (`codegen`)                   |
|       JitCompiler + Cranelift -> CompiledFunction             |
+---------------------------------------------------------------+
|              Compile Facade (`compile`)                       |
|       compile_fn / compile_graph / CompiledModel              |
+---------------------------------------------------------------+
```

## Modules

### `ir`

Typed intermediate representation.

```rust
pub struct NodeId(pub usize);

pub enum DataType { F32, F64, I32, I64, Bool }
pub struct Shape(pub Vec<usize>);

pub enum Op {
    // IO
    Input { name: String },
    Output { name: String, input: NodeId },
    Constant { value: f64 },

    // Binary
    Add, Sub, Mul, Div, MatMul,

    // Unary
    Neg, Abs, Sqrt, Exp, Log, Sin, Cos, Tanh,

    // Activations
    Relu, Sigmoid, Gelu, Silu,

    // Reductions
    Sum, SumAxis, Mean, MeanAxis, MaxAxis,

    // Shape
    Reshape, Transpose, Squeeze, Unsqueeze, Broadcast,

    // Comparisons / select
    Gt, Lt, Eq, Where,
}
```

`Graph` holds a `Vec<Node>`, input/output name maps, and exposes
`inputs()`, `outputs()`, `topological_order()`, and `validate()`.

### `trace`

`Tracer` + `TracedValue` — fluent surface used inside `trace(|t| ...)` to
record ops into a `Graph`.

```rust
pub fn trace<F>(f: F) -> Graph
where F: FnOnce(&Tracer) -> TracedValue;
```

`TracedValue` methods cover binary (`add`, `sub`, `mul`, `div`, `pow`,
`matmul`), scalar (`add_scalar`, `mul_scalar`), unary (`neg`, `abs`,
`sqrt`, `exp`, `log`, `sin`, `cos`, `tanh`), activations (`relu`,
`sigmoid`, `gelu`, `silu`), reductions (`sum`, `sum_axis`, `mean`,
`mean_axis`), and shape ops (`reshape`, `transpose`, `squeeze`,
`unsqueeze`).

### `optimize`

```rust
pub enum OptimizationPass {
    ConstantFolding,
    DeadCodeElimination,
    ElementwiseFusion,
    CommonSubexpressionElimination,
    AlgebraicSimplification,
    StrengthReduction,
}

pub struct Optimizer { /* list of passes */ }

impl Optimizer {
    pub fn new() -> Self;
    pub fn default_passes() -> Self;
    pub fn add_pass(&mut self, pass: OptimizationPass);
    pub fn optimize(&self, graph: Graph) -> Graph;
}
```

### `codegen`

Cranelift-backed compilation.

```rust
pub struct JitCompiler { /* ... */ }
pub struct CompiledFunction { /* ... */ }

impl JitCompiler {
    pub fn new() -> Self;
    pub fn compile(&self, graph: &Graph) -> JitResult<CompiledFunction>;
    pub fn cache_stats(&self) -> CacheStats;
    pub fn clear_cache(&self);
}

impl CompiledFunction {
    pub fn run(&self, inputs: &[(&str, &[f32])]) -> JitResult<Vec<f32>>;
}
```

### `cache`

`FunctionCache` — LRU hash-indexed cache of compiled graphs, keyed by
`FunctionCache::hash_graph(&Graph) -> u64`.

### `compile`

`torch.compile`-style high-level API.

```rust
pub enum Mode { Default, ReduceOverhead, MaxAutotune }
pub enum Backend { Default /* Cranelift */, Eager, AOT, ONNX }

pub struct CompileConfig {
    pub mode: Mode,
    pub backend: Backend,
    pub fullgraph: bool,
    pub dynamic: bool,
    pub disable: bool,
    pub passes: Vec<OptimizationPass>,
}

pub fn compile_fn<F>(f: F) -> CompiledModel where F: FnOnce(&Tracer) -> TracedValue;
pub fn compile_fn_with_config<F>(f: F, cfg: CompileConfig) -> CompiledModel;
pub fn compile_graph(graph: Graph) -> CompiledModel;
pub fn compile_graph_with_config(graph: Graph, cfg: CompileConfig) -> CompiledModel;
```

`CompiledModel` is the runnable output; `LazyCompiled` defers compilation
until first call; `CompileStats` reports graph size / pass counts / cache
hits.

### `error`

`JitError` + `JitResult<T>`. Variants include `InvalidGraph`,
`CompilationFailed`, `RuntimeError`, `ShapeMismatch`, `InputNotFound`,
`OutputNotFound`, `UnsupportedOp`.

## Usage

### Basic trace + compile + run

```rust
use axonml_jit::{trace, JitCompiler};

let graph = trace(|t| {
    let a = t.input("a", &[2, 3]);
    let b = t.input("b", &[2, 3]);
    let c = a.add(&b).relu();
    t.output("result", c)
});

let compiler = JitCompiler::new();
let func = compiler.compile(&graph).unwrap();

let a_data = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
let b_data = [1.0,  1.0, 1.0,  1.0, 1.0,  1.0];
let result = func.run(&[("a", &a_data), ("b", &b_data)]).unwrap();
// result = [2.0, 0.0, 4.0, 0.0, 6.0, 0.0]
```

### Chained ops

```rust
let graph = trace(|t| {
    let x = t.input("x", &[4, 4]);
    let y = x.relu()
        .mul_scalar(2.0)
        .add_scalar(1.0)
        .sigmoid();
    t.output("y", y)
});
```

### Matrix multiplication

```rust
let graph = trace(|t| {
    let a = t.input("a", &[2, 3]);
    let b = t.input("b", &[3, 4]);
    let c = a.matmul(&b);
    t.output("c", c)
});
```

### Reductions

```rust
let graph = trace(|t| {
    let x = t.input("x", &[2, 3, 4]);
    let sum = x.sum_axis(1, true); // [2, 1, 4]
    t.output("result", sum)
});
```

### Custom optimizer

```rust
use axonml_jit::{Optimizer, OptimizationPass};

let mut opt = Optimizer::new();
opt.add_pass(OptimizationPass::ConstantFolding);
opt.add_pass(OptimizationPass::DeadCodeElimination);
opt.add_pass(OptimizationPass::AlgebraicSimplification);

let optimized = opt.optimize(graph);
```

### High-level `compile_fn`

```rust
use axonml_jit::{compile_fn, CompileConfig, Mode, Backend};

let model = compile_fn(|t| {
    let x = t.input("x", &[8]);
    let y = x.silu().mul_scalar(0.5);
    t.output("y", y)
});

let out = model.run(&[("x", &[0.0, 1.0, 2.0, -1.0, -2.0, 0.5, -0.5, 3.0])]).unwrap();
```

## Performance Tips

1. Reuse `CompiledFunction`s — the `FunctionCache` keys off graph hash.
2. Batch operations for better amortization of compilation overhead.
3. Enable the default passes (`Optimizer::default_passes()`) for a balanced
   mix of CSE, constant folding, and algebraic simplification.
4. Small graphs may not beat eager execution — fall back to `Backend::Eager`
   when tracing overhead dominates.

## Limitations

- Execution data type is currently f32.
- Matrix multiplication in the interpreter path is limited to 2D tensors.
- `Backend::AOT` and `Backend::ONNX` are stubs for future work; the Default
  backend is Cranelift JIT.

## Last updated

0.6.5 (2026-06-06)
