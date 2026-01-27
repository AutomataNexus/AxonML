//! Code Generation
//!
//! Generates native code from computation graphs using Cranelift.

use std::sync::Arc;

use crate::cache::FunctionCache;
use crate::error::{JitError, JitResult};
use crate::ir::{Graph, Node, NodeId, Op};
use crate::optimize::Optimizer;

/// A compiled function ready for execution.
#[derive(Clone)]
pub struct CompiledFunction {
    /// The original graph.
    graph: Arc<Graph>,
    /// Function kind.
    kind: CompiledKind,
}

#[derive(Clone)]
enum CompiledKind {
    /// Interpreted execution (fallback).
    Interpreted,
    /// Native code (future: Cranelift JIT).
    #[allow(dead_code)]
    Native {
        /// Pointer to compiled code.
        code_ptr: *const u8,
        /// Code size.
        code_size: usize,
    },
}

// Safety: The native code pointer is never dereferenced without proper synchronization
unsafe impl Send for CompiledKind {}
unsafe impl Sync for CompiledKind {}

impl CompiledFunction {
    /// Creates a placeholder compiled function (for testing).
    pub fn placeholder() -> Self {
        Self {
            graph: Arc::new(Graph::new()),
            kind: CompiledKind::Interpreted,
        }
    }

    /// Returns the graph.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Executes the compiled function with the given inputs.
    pub fn run(&self, inputs: &[(&str, &[f32])]) -> JitResult<Vec<f32>> {
        match &self.kind {
            CompiledKind::Interpreted => self.run_interpreted(inputs),
            CompiledKind::Native {
                code_ptr,
                code_size,
            } => {
                // Native execution via function pointer call
                // Safety: code_ptr points to valid compiled code from Cranelift
                unsafe {
                    let func: extern "C" fn(*const f32, *mut f32) = std::mem::transmute(code_ptr);
                    let flat_inputs: Vec<f32> =
                        inputs.iter().flat_map(|(_, d)| d.iter().copied()).collect();
                    let mut output = vec![0.0f32; self.graph.outputs().len() * 1024]; // Max output size
                    func(flat_inputs.as_ptr(), output.as_mut_ptr());
                    let _ = code_size; // Used for memory management
                    Ok(output)
                }
            }
        }
    }

    /// Interpreted execution.
    fn run_interpreted(&self, inputs: &[(&str, &[f32])]) -> JitResult<Vec<f32>> {
        let mut values: Vec<Option<Vec<f32>>> = vec![None; self.graph.len()];

        // Set input values
        for (name, data) in inputs {
            if let Some(id) = self.graph.input(name) {
                values[id.index()] = Some(data.to_vec());
            } else {
                return Err(JitError::InputNotFound(name.to_string()));
            }
        }

        // Execute in topological order
        for node in self.graph.nodes() {
            let result = self.eval_node(node, &values)?;
            values[node.id.index()] = Some(result);
        }

        // Get output value
        if let Some((_, output_id)) = self.graph.outputs().iter().next() {
            let output_node = self.graph.node(*output_id);
            if let Op::Output { input, .. } = &output_node.op {
                return Ok(values[input.index()].clone().unwrap_or_default());
            }
        }

        Err(JitError::OutputNotFound("no output".to_string()))
    }

    fn eval_node(&self, node: &Node, values: &[Option<Vec<f32>>]) -> JitResult<Vec<f32>> {
        let get = |id: NodeId| -> JitResult<&Vec<f32>> {
            values[id.index()]
                .as_ref()
                .ok_or_else(|| JitError::RuntimeError(format!("Node {:?} not computed", id)))
        };

        match &node.op {
            Op::Input { .. } => {
                // Already set
                Ok(values[node.id.index()].clone().unwrap_or_default())
            }

            Op::Output { input, .. } => Ok(get(*input)?.clone()),

            Op::Constant { value } => {
                let numel = node.shape.numel();
                Ok(vec![*value as f32; numel])
            }

            // Binary ops
            Op::Add { lhs, rhs } => {
                let a = get(*lhs)?;
                let b = get(*rhs)?;
                Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
            }

            Op::Sub { lhs, rhs } => {
                let a = get(*lhs)?;
                let b = get(*rhs)?;
                Ok(a.iter().zip(b.iter()).map(|(x, y)| x - y).collect())
            }

            Op::Mul { lhs, rhs } => {
                let a = get(*lhs)?;
                let b = get(*rhs)?;
                Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).collect())
            }

            Op::Div { lhs, rhs } => {
                let a = get(*lhs)?;
                let b = get(*rhs)?;
                Ok(a.iter().zip(b.iter()).map(|(x, y)| x / y).collect())
            }

            Op::Pow { base, exp } => {
                let a = get(*base)?;
                let b = get(*exp)?;
                Ok(a.iter().zip(b.iter()).map(|(x, y)| x.powf(*y)).collect())
            }

            Op::Max { lhs, rhs } => {
                let a = get(*lhs)?;
                let b = get(*rhs)?;
                Ok(a.iter().zip(b.iter()).map(|(x, y)| x.max(*y)).collect())
            }

            Op::Min { lhs, rhs } => {
                let a = get(*lhs)?;
                let b = get(*rhs)?;
                Ok(a.iter().zip(b.iter()).map(|(x, y)| x.min(*y)).collect())
            }

            // Scalar ops
            Op::AddScalar { input, scalar } => {
                let a = get(*input)?;
                Ok(a.iter().map(|x| x + *scalar as f32).collect())
            }

            Op::MulScalar { input, scalar } => {
                let a = get(*input)?;
                Ok(a.iter().map(|x| x * *scalar as f32).collect())
            }

            // Unary ops
            Op::Neg { input } => {
                let a = get(*input)?;
                Ok(a.iter().map(|x| -x).collect())
            }

            Op::Abs { input } => {
                let a = get(*input)?;
                Ok(a.iter().map(|x| x.abs()).collect())
            }

            Op::Sqrt { input } => {
                let a = get(*input)?;
                Ok(a.iter().map(|x| x.sqrt()).collect())
            }

            Op::Exp { input } => {
                let a = get(*input)?;
                Ok(a.iter().map(|x| x.exp()).collect())
            }

            Op::Log { input } => {
                let a = get(*input)?;
                Ok(a.iter().map(|x| x.ln()).collect())
            }

            Op::Sin { input } => {
                let a = get(*input)?;
                Ok(a.iter().map(|x| x.sin()).collect())
            }

            Op::Cos { input } => {
                let a = get(*input)?;
                Ok(a.iter().map(|x| x.cos()).collect())
            }

            Op::Tanh { input } => {
                let a = get(*input)?;
                Ok(a.iter().map(|x| x.tanh()).collect())
            }

            // Activations
            Op::Relu { input } => {
                let a = get(*input)?;
                Ok(a.iter().map(|x| x.max(0.0)).collect())
            }

            Op::Sigmoid { input } => {
                let a = get(*input)?;
                Ok(a.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect())
            }

            Op::Gelu { input } => {
                let a = get(*input)?;
                const SQRT_2_OVER_PI: f32 = 0.7978845608;
                Ok(a.iter()
                    .map(|x| 0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x.powi(3))).tanh()))
                    .collect())
            }

            Op::Silu { input } => {
                let a = get(*input)?;
                Ok(a.iter().map(|x| x / (1.0 + (-x).exp())).collect())
            }

            // Reductions
            Op::Sum { input } => {
                let a = get(*input)?;
                Ok(vec![a.iter().sum()])
            }

            Op::Mean { input } => {
                let a = get(*input)?;
                let sum: f32 = a.iter().sum();
                Ok(vec![sum / a.len() as f32])
            }

            Op::SumAxis {
                input,
                axis,
                keepdim,
            } => {
                // Simplified: just sum all for now
                let a = get(*input)?;
                let input_node = self.graph.node(*input);
                let input_shape = input_node.shape.dims();

                reduce_axis(a, input_shape, *axis, *keepdim, |x, y| x + y, 0.0)
            }

            Op::MeanAxis {
                input,
                axis,
                keepdim,
            } => {
                let a = get(*input)?;
                let input_node = self.graph.node(*input);
                let input_shape = input_node.shape.dims();
                let axis_size = input_shape[normalize_axis(*axis, input_shape.len())];

                let sum = reduce_axis(a, input_shape, *axis, *keepdim, |x, y| x + y, 0.0)?;
                Ok(sum.iter().map(|x| x / axis_size as f32).collect())
            }

            Op::MaxAxis {
                input,
                axis,
                keepdim,
            } => {
                let a = get(*input)?;
                let input_node = self.graph.node(*input);
                let input_shape = input_node.shape.dims();

                reduce_axis(a, input_shape, *axis, *keepdim, f32::max, f32::NEG_INFINITY)
            }

            // Shape ops - for interpreter, just pass through
            Op::Reshape { input, .. }
            | Op::Transpose { input, .. }
            | Op::Squeeze { input, .. }
            | Op::Unsqueeze { input, .. }
            | Op::Broadcast { input, .. }
            | Op::Contiguous { input } => Ok(get(*input)?.clone()),

            // Matrix multiplication
            Op::MatMul { lhs, rhs } => {
                let a = get(*lhs)?;
                let b = get(*rhs)?;
                let lhs_node = self.graph.node(*lhs);
                let rhs_node = self.graph.node(*rhs);

                let lhs_shape = lhs_node.shape.dims();
                let rhs_shape = rhs_node.shape.dims();

                matmul_impl(a, b, lhs_shape, rhs_shape)
            }

            // Comparisons
            Op::Gt { lhs, rhs } => {
                let a = get(*lhs)?;
                let b = get(*rhs)?;
                Ok(a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| if x > y { 1.0 } else { 0.0 })
                    .collect())
            }

            Op::Lt { lhs, rhs } => {
                let a = get(*lhs)?;
                let b = get(*rhs)?;
                Ok(a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| if x < y { 1.0 } else { 0.0 })
                    .collect())
            }

            Op::Eq { lhs, rhs } => {
                let a = get(*lhs)?;
                let b = get(*rhs)?;
                Ok(a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| {
                        if (x - y).abs() < f32::EPSILON {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect())
            }

            Op::Where { condition, x, y } => {
                let cond = get(*condition)?;
                let a = get(*x)?;
                let b = get(*y)?;
                Ok(cond
                    .iter()
                    .zip(a.iter().zip(b.iter()))
                    .map(|(c, (a, b))| if *c != 0.0 { *a } else { *b })
                    .collect())
            }

            Op::Cast { input, .. } => {
                // For f32, just pass through
                Ok(get(*input)?.clone())
            }
        }
    }
}

fn normalize_axis(axis: i32, ndim: usize) -> usize {
    if axis < 0 {
        (ndim as i32 + axis) as usize
    } else {
        axis as usize
    }
}

fn reduce_axis(
    data: &[f32],
    shape: &[usize],
    axis: i32,
    keepdim: bool,
    op: fn(f32, f32) -> f32,
    init: f32,
) -> JitResult<Vec<f32>> {
    let axis = normalize_axis(axis, shape.len());

    // Compute strides
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // Compute output shape
    let mut output_shape: Vec<usize> = shape.to_vec();
    if keepdim {
        output_shape[axis] = 1;
    } else {
        output_shape.remove(axis);
    }

    let output_numel: usize = output_shape.iter().product();
    let mut result = vec![init; output_numel];

    // Reduce
    for i in 0..data.len() {
        // Convert linear index to multi-index
        let mut multi_idx = vec![0usize; shape.len()];
        let mut idx = i;
        for d in 0..shape.len() {
            multi_idx[d] = idx / strides[d];
            idx %= strides[d];
        }

        // Compute output index
        let out_idx = if keepdim {
            let mut out_idx = 0;
            let mut temp_strides = vec![1usize; output_shape.len()];
            for d in (0..output_shape.len() - 1).rev() {
                temp_strides[d] = temp_strides[d + 1] * output_shape[d + 1];
            }
            for d in 0..output_shape.len() {
                let dim_idx = if d == axis { 0 } else { multi_idx[d] };
                out_idx += dim_idx * temp_strides[d];
            }
            out_idx
        } else {
            let mut out_idx = 0;
            let mut temp_strides = vec![1usize; output_shape.len()];
            if !output_shape.is_empty() {
                for d in (0..output_shape.len() - 1).rev() {
                    temp_strides[d] = temp_strides[d + 1] * output_shape[d + 1];
                }
            }
            let mut out_d = 0;
            for d in 0..shape.len() {
                if d == axis {
                    continue;
                }
                if out_d < temp_strides.len() {
                    out_idx += multi_idx[d] * temp_strides[out_d];
                }
                out_d += 1;
            }
            out_idx
        };

        if out_idx < result.len() {
            result[out_idx] = op(result[out_idx], data[i]);
        }
    }

    Ok(result)
}

fn matmul_impl(a: &[f32], b: &[f32], a_shape: &[usize], b_shape: &[usize]) -> JitResult<Vec<f32>> {
    // Simple 2D matmul
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(JitError::UnsupportedOp(
            "Only 2D matmul supported in interpreter".to_string(),
        ));
    }

    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    if k != b_shape[0] {
        return Err(JitError::ShapeMismatch {
            expected: vec![k],
            found: vec![b_shape[0]],
        });
    }

    let mut result = vec![0.0f32; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            result[i * n + j] = sum;
        }
    }

    Ok(result)
}

/// JIT compiler.
pub struct JitCompiler {
    optimizer: Optimizer,
    cache: FunctionCache,
    use_native: bool,
}

impl JitCompiler {
    /// Creates a new JIT compiler.
    pub fn new() -> Self {
        Self {
            optimizer: Optimizer::default_passes(),
            cache: FunctionCache::default_size(),
            use_native: false, // Fallback to interpreter for now
        }
    }

    /// Creates a compiler with custom optimizer.
    pub fn with_optimizer(optimizer: Optimizer) -> Self {
        Self {
            optimizer,
            cache: FunctionCache::default_size(),
            use_native: false,
        }
    }

    /// Compiles a graph into an executable function.
    pub fn compile(&self, graph: &Graph) -> JitResult<CompiledFunction> {
        // Check cache
        let cache_key = FunctionCache::hash_graph(graph);
        if let Some(cached) = self.cache.get(cache_key) {
            return Ok(cached);
        }

        // Validate graph
        graph.validate().map_err(JitError::InvalidGraph)?;

        // Optimize
        let optimized = self.optimizer.optimize(graph.clone());

        // Generate code
        let func = if self.use_native {
            self.compile_native(&optimized)?
        } else {
            self.compile_interpreted(&optimized)
        };

        // Cache result
        self.cache.insert(cache_key, func.clone());

        Ok(func)
    }

    fn compile_interpreted(&self, graph: &Graph) -> CompiledFunction {
        CompiledFunction {
            graph: Arc::new(graph.clone()),
            kind: CompiledKind::Interpreted,
        }
    }

    fn compile_native(&self, graph: &Graph) -> JitResult<CompiledFunction> {
        use cranelift::prelude::*;
        use cranelift_jit::{JITBuilder, JITModule};
        use cranelift_module::{Linkage, Module};

        // Initialize Cranelift JIT module
        let mut flag_builder = settings::builder();
        flag_builder.set("use_colocated_libcalls", "false").unwrap();
        flag_builder.set("is_pic", "false").unwrap();
        let isa_builder = cranelift_native::builder()
            .map_err(|e| JitError::CompilationFailed(format!("Failed to get native ISA: {}", e)))?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| JitError::CompilationFailed(format!("Failed to build ISA: {}", e)))?;

        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);

        // Create function signature: fn(inputs: *const f32, outputs: *mut f32)
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(types::I64)); // input ptr
        sig.params.push(AbiParam::new(types::I64)); // output ptr

        let func_id = module
            .declare_function("jit_kernel", Linkage::Export, &sig)
            .map_err(|e| {
                JitError::CompilationFailed(format!("Failed to declare function: {}", e))
            })?;

        let mut ctx = module.make_context();
        ctx.func.signature = sig;

        // Build function body
        let mut builder_ctx = FunctionBuilderContext::new();
        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let input_ptr = builder.block_params(entry_block)[0];
            let output_ptr = builder.block_params(entry_block)[1];

            // Generate code for each operation in the graph
            let mut values: Vec<Option<Value>> = vec![None; graph.len()];

            for node in graph.nodes() {
                let result = self.codegen_node(&mut builder, node, &values, input_ptr)?;
                values[node.id.index()] = Some(result);
            }

            // Store output
            if let Some((_, output_id)) = graph.outputs().iter().next() {
                let output_node = graph.node(*output_id);
                if let Op::Output { input, .. } = &output_node.op {
                    if let Some(val) = values[input.index()] {
                        builder.ins().store(MemFlags::new(), val, output_ptr, 0);
                    }
                }
            }

            builder.ins().return_(&[]);
            builder.finalize();
        }

        // Compile the function
        module.define_function(func_id, &mut ctx).map_err(|e| {
            JitError::CompilationFailed(format!("Failed to define function: {}", e))
        })?;
        module.clear_context(&mut ctx);
        module
            .finalize_definitions()
            .map_err(|e| JitError::CompilationFailed(format!("Failed to finalize: {:?}", e)))?;

        let code_ptr = module.get_finalized_function(func_id);
        let code_size = 0; // JITModule manages memory

        // Leak the module to keep the code alive
        std::mem::forget(module);

        Ok(CompiledFunction {
            graph: Arc::new(graph.clone()),
            kind: CompiledKind::Native {
                code_ptr: code_ptr as *const u8,
                code_size,
            },
        })
    }

    fn codegen_node(
        &self,
        builder: &mut cranelift::prelude::FunctionBuilder,
        node: &Node,
        values: &[Option<cranelift::prelude::Value>],
        input_ptr: cranelift::prelude::Value,
    ) -> JitResult<cranelift::prelude::Value> {
        use cranelift::prelude::*;

        let get = |id: NodeId| -> JitResult<Value> {
            values[id.index()]
                .ok_or_else(|| JitError::RuntimeError(format!("Node {:?} not compiled", id)))
        };

        match &node.op {
            Op::Input { name, .. } => {
                // Load from input pointer at appropriate offset
                let offset = self.get_input_offset(name);
                Ok(builder
                    .ins()
                    .load(types::F32, MemFlags::new(), input_ptr, offset))
            }

            Op::Output { input, .. } => get(*input),

            Op::Constant { value } => Ok(builder.ins().f32const(*value as f32)),

            Op::Add { lhs, rhs } => {
                let a = get(*lhs)?;
                let b = get(*rhs)?;
                Ok(builder.ins().fadd(a, b))
            }

            Op::Sub { lhs, rhs } => {
                let a = get(*lhs)?;
                let b = get(*rhs)?;
                Ok(builder.ins().fsub(a, b))
            }

            Op::Mul { lhs, rhs } => {
                let a = get(*lhs)?;
                let b = get(*rhs)?;
                Ok(builder.ins().fmul(a, b))
            }

            Op::Div { lhs, rhs } => {
                let a = get(*lhs)?;
                let b = get(*rhs)?;
                Ok(builder.ins().fdiv(a, b))
            }

            Op::Neg { input } => {
                let a = get(*input)?;
                Ok(builder.ins().fneg(a))
            }

            Op::Abs { input } => {
                let a = get(*input)?;
                Ok(builder.ins().fabs(a))
            }

            Op::Sqrt { input } => {
                let a = get(*input)?;
                Ok(builder.ins().sqrt(a))
            }

            Op::AddScalar { input, scalar } => {
                let a = get(*input)?;
                let s = builder.ins().f32const(*scalar as f32);
                Ok(builder.ins().fadd(a, s))
            }

            Op::MulScalar { input, scalar } => {
                let a = get(*input)?;
                let s = builder.ins().f32const(*scalar as f32);
                Ok(builder.ins().fmul(a, s))
            }

            // For operations not easily supported by Cranelift scalars,
            // fall back to interpreted execution for the whole graph
            _ => Err(JitError::UnsupportedOp(format!(
                "Operation {:?} not supported in native codegen, using interpreter",
                node.op
            ))),
        }
    }

    fn get_input_offset(&self, _name: &str) -> i32 {
        // Simple offset calculation - in practice would use a mapping
        0
    }

    /// Returns cache statistics.
    pub fn cache_stats(&self) -> crate::cache::CacheStats {
        self.cache.stats()
    }

    /// Clears the compilation cache.
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
}

impl Default for JitCompiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::trace;

    #[test]
    fn test_compile_simple() {
        let graph = trace(|tracer| {
            let a = tracer.input("a", &[4]);
            let b = tracer.input("b", &[4]);
            let c = a.add(&b);
            tracer.output("result", c)
        });

        let compiler = JitCompiler::new();
        let func = compiler.compile(&graph).unwrap();

        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let result = func.run(&[("a", &a), ("b", &b)]).unwrap();

        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_compile_chain() {
        let graph = trace(|tracer| {
            let x = tracer.input("x", &[4]);
            let y = x.relu().mul_scalar(2.0).add_scalar(1.0);
            tracer.output("y", y)
        });

        let compiler = JitCompiler::new();
        let func = compiler.compile(&graph).unwrap();

        let x = [-1.0, 0.0, 1.0, 2.0];
        let result = func.run(&[("x", &x)]).unwrap();

        // relu([-1, 0, 1, 2]) = [0, 0, 1, 2]
        // * 2 = [0, 0, 2, 4]
        // + 1 = [1, 1, 3, 5]
        assert_eq!(result, vec![1.0, 1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_compile_activations() {
        let graph = trace(|tracer| {
            let x = tracer.input("x", &[3]);
            let y = x.sigmoid();
            tracer.output("y", y)
        });

        let compiler = JitCompiler::new();
        let func = compiler.compile(&graph).unwrap();

        let x = [0.0, 1.0, -1.0];
        let result = func.run(&[("x", &x)]).unwrap();

        // sigmoid(0) = 0.5
        assert!((result[0] - 0.5).abs() < 0.01);
        // sigmoid(1) ≈ 0.731
        assert!((result[1] - 0.731).abs() < 0.01);
    }

    #[test]
    fn test_compile_matmul() {
        let graph = trace(|tracer| {
            let a = tracer.input("a", &[2, 3]);
            let b = tracer.input("b", &[3, 2]);
            let c = a.matmul(&b);
            tracer.output("c", c)
        });

        let compiler = JitCompiler::new();
        let func = compiler.compile(&graph).unwrap();

        // Identity-like matrices
        let a = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // 2x3
        let b = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 3x2
        let result = func.run(&[("a", &a), ("b", &b)]).unwrap();

        assert_eq!(result.len(), 4); // 2x2
    }

    #[test]
    fn test_caching() {
        let graph = trace(|tracer| {
            let x = tracer.input("x", &[4]);
            tracer.output("y", x.relu())
        });

        let compiler = JitCompiler::new();
        assert_eq!(compiler.cache_stats().entries, 0);

        let _ = compiler.compile(&graph).unwrap();
        assert_eq!(compiler.cache_stats().entries, 1);

        // Second compile should use cache
        let _ = compiler.compile(&graph).unwrap();
        assert_eq!(compiler.cache_stats().entries, 1);
    }
}
