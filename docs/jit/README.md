# axonml-jit Documentation

> JIT compilation and tracing for Axonml tensor operations.

## Overview

`axonml-jit` provides Just-In-Time compilation capabilities for tensor operations. It allows you to trace tensor computations, build computation graphs, optimize them, and execute them efficiently.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      User Code                          │
│   trace(|t| { a.add(&b).relu() })                       │
├─────────────────────────────────────────────────────────┤
│                    Tracer (trace.rs)                    │
│   Records operations into computation graph             │
├─────────────────────────────────────────────────────────┤
│              Intermediate Representation (ir.rs)        │
│   Graph, Node, Op, Shape, DataType                      │
├─────────────────────────────────────────────────────────┤
│                  Optimizer (optimize.rs)                │
│   Constant folding, DCE, CSE, algebraic simplification  │
├─────────────────────────────────────────────────────────┤
│                Function Cache (cache.rs)                │
│   Hash-based caching of compiled functions              │
├─────────────────────────────────────────────────────────┤
│                Code Generator (codegen.rs)              │
│   JitCompiler, CompiledFunction, Interpreter            │
└─────────────────────────────────────────────────────────┘
```

## Modules

### ir.rs

Defines the Intermediate Representation for computation graphs.

**Types:**

```rust
pub struct NodeId(usize);

pub enum DataType {
    F32, F64, I32, I64, Bool
}

pub struct Shape(Vec<usize>);

pub enum Op {
    // I/O
    Input { name: String },
    Output { name: String, input: NodeId },
    Constant { value: f64 },

    // Binary ops
    Add { lhs: NodeId, rhs: NodeId },
    Sub { lhs: NodeId, rhs: NodeId },
    Mul { lhs: NodeId, rhs: NodeId },
    Div { lhs: NodeId, rhs: NodeId },
    MatMul { lhs: NodeId, rhs: NodeId },

    // Unary ops
    Neg, Abs, Sqrt, Exp, Log, Sin, Cos, Tanh,

    // Activations
    Relu, Sigmoid, Gelu, Silu,

    // Reductions
    Sum, SumAxis, Mean, MeanAxis, MaxAxis,

    // Shape ops
    Reshape, Transpose, Squeeze, Unsqueeze, Broadcast,

    // Comparisons
    Gt, Lt, Eq, Where,
}

pub struct Node {
    pub id: NodeId,
    pub op: Op,
    pub dtype: DataType,
    pub shape: Shape,
}

pub struct Graph {
    nodes: Vec<Node>,
    inputs: HashMap<String, NodeId>,
    outputs: HashMap<String, NodeId>,
}
```

**Key methods:**
- `Graph::new()` - Create empty graph
- `Graph::add_node(op, dtype, shape)` - Add node to graph
- `Graph::validate()` - Validate graph structure
- `Graph::topological_order()` - Get nodes in execution order

### trace.rs

Operation tracing to build computation graphs.

**Types:**

```rust
pub struct TracedValue {
    id: NodeId,
}

pub struct Tracer {
    // Handle for recording operations
}
```

**TracedValue operations:**
- Binary: `add`, `sub`, `mul`, `div`, `pow`, `matmul`
- Scalar: `add_scalar`, `mul_scalar`
- Unary: `neg`, `abs`, `sqrt`, `exp`, `log`, `sin`, `cos`, `tanh`
- Activations: `relu`, `sigmoid`, `gelu`, `silu`
- Reductions: `sum`, `sum_axis`, `mean`, `mean_axis`
- Shape: `reshape`, `transpose`, `squeeze`, `unsqueeze`

**Tracing function:**

```rust
pub fn trace<F>(f: F) -> Graph
where
    F: FnOnce(&Tracer) -> TracedValue;
```

### optimize.rs

Graph optimization passes.

**Optimization passes:**

```rust
pub enum OptimizationPass {
    ConstantFolding,           // Evaluate constant expressions
    DeadCodeElimination,       // Remove unused nodes
    ElementwiseFusion,         // Fuse consecutive elementwise ops
    CommonSubexpressionElimination,  // Reuse common subexpressions
    AlgebraicSimplification,   // x * 1 = x, x + 0 = x, etc.
    StrengthReduction,         // Replace expensive ops with cheaper ones
}
```

**Optimizer:**

```rust
pub struct Optimizer {
    passes: Vec<OptimizationPass>,
}

impl Optimizer {
    pub fn new() -> Self;
    pub fn default_passes() -> Self;
    pub fn add_pass(&mut self, pass: OptimizationPass);
    pub fn optimize(&self, graph: Graph) -> Graph;
}
```

### cache.rs

Function caching for compiled functions.

```rust
pub struct FunctionCache {
    cache: RwLock<HashMap<u64, CompiledFunction>>,
    max_size: usize,
}

impl FunctionCache {
    pub fn new(max_size: usize) -> Self;
    pub fn hash_graph(graph: &Graph) -> u64;
    pub fn get(&self, key: u64) -> Option<CompiledFunction>;
    pub fn insert(&self, key: u64, func: CompiledFunction);
    pub fn stats(&self) -> CacheStats;
}
```

### codegen.rs

JIT compiler and code generation.

```rust
pub struct CompiledFunction {
    graph: Arc<Graph>,
    // Compiled code or interpreter
}

impl CompiledFunction {
    pub fn run(&self, inputs: &[(&str, &[f32])]) -> JitResult<Vec<f32>>;
}

pub struct JitCompiler {
    optimizer: Optimizer,
    cache: FunctionCache,
}

impl JitCompiler {
    pub fn new() -> Self;
    pub fn compile(&self, graph: &Graph) -> JitResult<CompiledFunction>;
    pub fn cache_stats(&self) -> CacheStats;
    pub fn clear_cache(&self);
}
```

## Usage

### Basic Tracing

```rust
use axonml_jit::{trace, JitCompiler};

// Trace a computation
let graph = trace(|tracer| {
    let a = tracer.input("a", &[2, 3]);
    let b = tracer.input("b", &[2, 3]);
    let c = a.add(&b).relu();
    tracer.output("result", c)
});

// Compile and run
let compiler = JitCompiler::new();
let func = compiler.compile(&graph).unwrap();

let a_data = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
let b_data = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
let result = func.run(&[("a", &a_data), ("b", &b_data)]).unwrap();
// result = [2.0, 0.0, 4.0, 0.0, 6.0, 0.0]
```

### Complex Operations

```rust
let graph = trace(|tracer| {
    let x = tracer.input("x", &[4, 4]);

    // Chain of operations
    let y = x
        .relu()
        .mul_scalar(2.0)
        .add_scalar(1.0)
        .sigmoid();

    tracer.output("y", y)
});
```

### Matrix Multiplication

```rust
let graph = trace(|tracer| {
    let a = tracer.input("a", &[2, 3]);
    let b = tracer.input("b", &[3, 4]);
    let c = a.matmul(&b);  // Result: [2, 4]
    tracer.output("c", c)
});
```

### Reductions

```rust
let graph = trace(|tracer| {
    let x = tracer.input("x", &[2, 3, 4]);
    let sum = x.sum_axis(1, true);  // Shape: [2, 1, 4]
    let mean = x.mean();            // Shape: []
    tracer.output("result", sum)
});
```

### Custom Optimization

```rust
use axonml_jit::{Optimizer, OptimizationPass};

let mut optimizer = Optimizer::new();
optimizer.add_pass(OptimizationPass::ConstantFolding);
optimizer.add_pass(OptimizationPass::DeadCodeElimination);
optimizer.add_pass(OptimizationPass::AlgebraicSimplification);

let optimized = optimizer.optimize(graph);
```

## Error Handling

```rust
pub enum JitError {
    InvalidGraph(String),
    CompilationFailed(String),
    RuntimeError(String),
    ShapeMismatch { expected: Vec<usize>, found: Vec<usize> },
    InputNotFound(String),
    OutputNotFound(String),
    UnsupportedOp(String),
}
```

## Performance Tips

1. **Reuse compiled functions**: The JitCompiler caches compiled functions by graph hash
2. **Batch operations**: Process multiple samples together for better efficiency
3. **Use optimization passes**: Enable algebraic simplification and CSE for cleaner graphs
4. **Avoid small graphs**: Tracing overhead may exceed benefits for very small computations

## Feature Flags

- Default: Interpreter-based execution
- Future: `native` - Enable Cranelift native code generation

## Limitations

- Currently uses interpreter execution (native codegen planned)
- Only supports f32 data type for execution
- Matrix multiplication limited to 2D tensors in interpreter
