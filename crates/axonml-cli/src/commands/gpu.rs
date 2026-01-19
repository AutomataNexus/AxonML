//! GPU - GPU Detection and Management
//!
//! Detect, list, and manage GPU devices for training and inference.
//! Uses wgpu for cross-platform GPU enumeration and compute.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::time::Instant;

use wgpu::{Backends, DeviceType, Instance, InstanceDescriptor, MemoryHints};

use super::utils::{print_header, print_info, print_kv, print_success, print_warning};
use crate::cli::{GpuArgs, GpuBenchArgs, GpuSelectArgs, GpuSubcommand};
use crate::error::{CliError, CliResult};

// =============================================================================
// GPU Info Structures
// =============================================================================

/// Information about a detected GPU
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub id: usize,
    pub name: String,
    pub vendor: String,
    pub device_type: String,
    pub backend: String,
    pub driver: String,
    pub memory_total: u64,
    pub is_available: bool,
}

// =============================================================================
// Execute Command
// =============================================================================

/// Execute the `gpu` command
pub fn execute(args: GpuArgs) -> CliResult<()> {
    match args.action {
        GpuSubcommand::List => execute_list(),
        GpuSubcommand::Info => execute_info(),
        GpuSubcommand::Select(select_args) => execute_select(select_args),
        GpuSubcommand::Bench(bench_args) => execute_bench(bench_args),
        GpuSubcommand::Memory => execute_memory(),
        GpuSubcommand::Status => execute_status(),
    }
}

// =============================================================================
// List Subcommand
// =============================================================================

fn execute_list() -> CliResult<()> {
    print_header("CUDA GPUs (NVIDIA)");
    println!();

    let gpus = detect_gpus()?;

    if gpus.is_empty() {
        print_warning("No NVIDIA CUDA GPUs detected.");
        println!();
        print_info("Ensure you have:");
        println!("  - NVIDIA GPU installed");
        println!("  - NVIDIA drivers with CUDA support");
        println!("  - For WSL2: GPU passthrough enabled");
        return Ok(());
    }

    println!(
        "{:<4} {:<35} {:<12} {:>12} {:>10}",
        "ID", "Name", "Backend", "Memory", "Type"
    );
    println!("{}", "-".repeat(77));

    for gpu in &gpus {
        println!(
            "{:<4} {:<35} {:<12} {:>12} {:>10}",
            gpu.id,
            truncate_string(&gpu.name, 34),
            gpu.backend,
            format_size(gpu.memory_total),
            gpu.device_type
        );
    }

    println!();
    print_kv("Total CUDA GPUs", &gpus.len().to_string());

    Ok(())
}

// =============================================================================
// Info Subcommand
// =============================================================================

fn execute_info() -> CliResult<()> {
    print_header("CUDA GPU Information");
    println!();

    // Detected CUDA devices
    let gpus = detect_gpus()?;

    if gpus.is_empty() {
        print_warning("No NVIDIA CUDA GPUs detected.");
        println!();
        print_info("Ensure you have:");
        println!("  - NVIDIA GPU installed");
        println!("  - NVIDIA drivers installed (with CUDA support)");
        println!("  - For WSL2: GPU passthrough enabled in Windows");
        return Ok(());
    }

    print_kv("CUDA Devices", &gpus.len().to_string());

    for gpu in &gpus {
        println!();
        print_header(&format!("CUDA Device {} - {}", gpu.id, gpu.name));
        print_kv("Vendor", &gpu.vendor);
        print_kv("Device Type", &gpu.device_type);
        print_kv("Backend", &gpu.backend);
        print_kv("Driver", &gpu.driver);
        print_kv("Memory", &format_size(gpu.memory_total));
        print_kv(
            "Status",
            if gpu.is_available {
                "Available"
            } else {
                "Busy"
            },
        );
    }

    Ok(())
}

// =============================================================================
// Select Subcommand
// =============================================================================

fn execute_select(args: GpuSelectArgs) -> CliResult<()> {
    print_header("GPU Selection");
    println!();

    let gpus = detect_gpus()?;

    if gpus.is_empty() {
        return Err(CliError::Gpu("No GPU devices available".to_string()));
    }

    // Parse device selection
    let device_id: usize = if args.device.to_lowercase() == "auto" {
        // Auto-select best available GPU (prefer discrete over integrated)
        gpus.iter()
            .filter(|g| g.device_type == "Discrete")
            .map(|g| g.id)
            .next()
            .unwrap_or(0)
    } else {
        args.device.parse().map_err(|_| {
            CliError::InvalidArgument(format!("Invalid device specifier: {}", args.device))
        })?
    };

    // Validate device exists
    let gpu = gpus
        .iter()
        .find(|g| g.id == device_id)
        .ok_or_else(|| CliError::Gpu(format!("GPU {device_id} not found")))?;

    // Save selection to config
    save_gpu_selection(device_id)?;

    print_success(&format!(
        "Selected GPU {}: {} ({})",
        device_id, gpu.name, gpu.backend
    ));

    if args.persistent {
        print_info("Selection saved to .axonml/gpu_config.json");
    }

    Ok(())
}

// =============================================================================
// Bench Subcommand
// =============================================================================

fn execute_bench(args: GpuBenchArgs) -> CliResult<()> {
    print_header("GPU Benchmark");
    println!();

    let gpus = detect_gpus()?;

    if gpus.is_empty() {
        return Err(CliError::Gpu(
            "No GPU devices available for benchmarking".to_string(),
        ));
    }

    // Determine which GPUs to benchmark
    let gpus_to_bench: Vec<&GpuInfo> = if args.all {
        gpus.iter().collect()
    } else if let Some(device) = &args.device {
        let id: usize = device
            .parse()
            .map_err(|_| CliError::InvalidArgument(format!("Invalid device ID: {device}")))?;
        gpus.iter().filter(|g| g.id == id).collect()
    } else {
        // Benchmark first available GPU
        gpus.iter().take(1).collect()
    };

    if gpus_to_bench.is_empty() {
        return Err(CliError::Gpu("No matching GPU found".to_string()));
    }

    print_kv("GPUs to benchmark", &gpus_to_bench.len().to_string());
    print_kv("Iterations", &args.iterations.to_string());
    println!();

    for gpu in gpus_to_bench {
        print_header(&format!("Benchmarking GPU {}: {}", gpu.id, gpu.name));
        println!();

        // Run actual GPU benchmarks using wgpu
        let results = run_gpu_benchmark(gpu.id, args.iterations)?;

        print_kv(
            "Buffer Copy (1MB)",
            &format!("{:.2} ms", results.buffer_copy_1mb_ms),
        );
        print_kv(
            "Buffer Copy (16MB)",
            &format!("{:.2} ms", results.buffer_copy_16mb_ms),
        );
        print_kv(
            "Buffer Copy (64MB)",
            &format!("{:.2} ms", results.buffer_copy_64mb_ms),
        );
        print_kv(
            "Compute Dispatch",
            &format!("{:.2} ms", results.compute_dispatch_ms),
        );

        println!();
        print_kv(
            "Effective Bandwidth (1MB)",
            &format!(
                "{}/s",
                format_size((1024.0 * 1024.0 / (results.buffer_copy_1mb_ms / 1000.0)) as u64)
            ),
        );
        print_kv(
            "Effective Bandwidth (16MB)",
            &format!(
                "{}/s",
                format_size(
                    (16.0 * 1024.0 * 1024.0 / (results.buffer_copy_16mb_ms / 1000.0)) as u64
                )
            ),
        );
        print_kv(
            "Effective Bandwidth (64MB)",
            &format!(
                "{}/s",
                format_size(
                    (64.0 * 1024.0 * 1024.0 / (results.buffer_copy_64mb_ms / 1000.0)) as u64
                )
            ),
        );

        println!();
    }

    print_success("Benchmark complete!");

    Ok(())
}

/// Benchmark results
struct BenchmarkResults {
    buffer_copy_1mb_ms: f64,
    buffer_copy_16mb_ms: f64,
    buffer_copy_64mb_ms: f64,
    compute_dispatch_ms: f64,
}

/// Run actual GPU benchmarks using wgpu
fn run_gpu_benchmark(gpu_id: usize, iterations: usize) -> CliResult<BenchmarkResults> {
    // Use all backends to support WSL2, native Windows, and native Linux
    let backends = Backends::all();

    let instance = Instance::new(InstanceDescriptor {
        backends,
        ..Default::default()
    });

    // Get only NVIDIA (CUDA) adapters
    // Check both vendor ID and name (for WSL2/D3D12 passthrough)
    let adapters: Vec<_> = instance
        .enumerate_adapters(backends)
        .into_iter()
        .filter(|a| {
            let info = a.get_info();
            let name_upper = info.name.to_uppercase();
            info.vendor == 0x10DE
                || name_upper.contains("NVIDIA")
                || name_upper.contains("GEFORCE")
                || name_upper.contains("RTX")
                || name_upper.contains("GTX")
                || name_upper.contains("QUADRO")
                || name_upper.contains("TESLA")
        })
        .collect();

    if gpu_id >= adapters.len() {
        return Err(CliError::Gpu(format!("CUDA GPU {gpu_id} not found")));
    }

    let adapter = &adapters[gpu_id];

    // Request device
    let (device, queue) = pollster::block_on(async {
        adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Benchmark Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: MemoryHints::Performance,
                },
                None,
            )
            .await
    })
    .map_err(|e| CliError::Gpu(format!("Failed to create device: {e}")))?;

    // Benchmark buffer copies at different sizes
    let buffer_copy_1mb_ms = benchmark_buffer_copy(&device, &queue, 1024 * 1024, iterations);
    let buffer_copy_16mb_ms = benchmark_buffer_copy(&device, &queue, 16 * 1024 * 1024, iterations);
    let buffer_copy_64mb_ms = benchmark_buffer_copy(&device, &queue, 64 * 1024 * 1024, iterations);

    // Benchmark compute dispatch
    let compute_dispatch_ms = benchmark_compute_dispatch(&device, &queue, iterations);

    Ok(BenchmarkResults {
        buffer_copy_1mb_ms,
        buffer_copy_16mb_ms,
        buffer_copy_64mb_ms,
        compute_dispatch_ms,
    })
}

/// Benchmark buffer copy operations
fn benchmark_buffer_copy(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    size: u64,
    iterations: usize,
) -> f64 {
    // Create source buffer with data
    let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

    let src_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Source Buffer"),
        size,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let dst_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Destination Buffer"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Write data to source buffer
    queue.write_buffer(&src_buffer, 0, &data);

    // Warmup
    for _ in 0..3 {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Copy Encoder"),
        });
        encoder.copy_buffer_to_buffer(&src_buffer, 0, &dst_buffer, 0, size);
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Copy Encoder"),
        });
        encoder.copy_buffer_to_buffer(&src_buffer, 0, &dst_buffer, 0, size);
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Benchmark compute shader dispatch
fn benchmark_compute_dispatch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    iterations: usize,
) -> f64 {
    // Simple compute shader that does vector addition
    let shader_source = r"
        @group(0) @binding(0) var<storage, read> input_a: array<f32>;
        @group(0) @binding(1) var<storage, read> input_b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> output: array<f32>;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let idx = global_id.x;
            if (idx < arrayLength(&output)) {
                output[idx] = input_a[idx] + input_b[idx];
            }
        }
    ";

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Compute Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let buffer_size: u64 = 1024 * 1024 * 4; // 1M floats = 4MB
    let num_elements = buffer_size / 4;

    // Create buffers
    let input_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Input A"),
        size: buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let input_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Input B"),
        size: buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output"),
        size: buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Initialize input buffers
    let data_a: Vec<f32> = (0..num_elements).map(|i| i as f32).collect();
    let data_b: Vec<f32> = (0..num_elements).map(|i| (i * 2) as f32).collect();
    queue.write_buffer(&input_a, 0, bytemuck::cast_slice(&data_a));
    queue.write_buffer(&input_b, 0, bytemuck::cast_slice(&data_b));

    // Create bind group layout and pipeline
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.as_entire_binding(),
            },
        ],
    });

    let workgroup_count = num_elements.div_ceil(256) as u32;

    // Warmup
    for _ in 0..3 {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

// =============================================================================
// Memory Subcommand
// =============================================================================

fn execute_memory() -> CliResult<()> {
    print_header("GPU Memory Information");
    println!();

    let gpus = detect_gpus()?;

    if gpus.is_empty() {
        print_warning("No GPU devices detected.");
        return Ok(());
    }

    println!("{:<4} {:<35} {:>15}", "ID", "Name", "Total Memory");
    println!("{}", "-".repeat(56));

    for gpu in &gpus {
        println!(
            "{:<4} {:<35} {:>15}",
            gpu.id,
            truncate_string(&gpu.name, 34),
            format_size(gpu.memory_total),
        );
    }

    println!();

    // Total across all GPUs
    let total_memory: u64 = gpus.iter().map(|g| g.memory_total).sum();
    print_kv("Total GPU Memory", &format_size(total_memory));

    // Note about memory reporting
    print_info("Note: wgpu reports total device memory. For real-time usage, use nvidia-smi or similar tools.");

    Ok(())
}

// =============================================================================
// Status Subcommand
// =============================================================================

fn execute_status() -> CliResult<()> {
    print_header("GPU Status");
    println!();

    // Current selection
    if let Some(device_id) = load_gpu_selection() {
        print_kv("Selected Device", &device_id.to_string());
    } else {
        print_kv("Selected Device", "None (auto-select)");
    }

    println!();

    // Show ALL adapters for diagnostics (not just NVIDIA)
    print_header("All Detected Adapters (Diagnostics)");
    println!();

    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::all(),
        ..Default::default()
    });

    let all_adapters = instance.enumerate_adapters(Backends::all());

    if all_adapters.is_empty() {
        print_warning("No GPU adapters detected by wgpu.");
        println!();
        print_info("This may indicate:");
        println!("  - Missing graphics drivers");
        println!("  - WSL2 GPU passthrough not configured");
        println!("  - No compatible GPU hardware");
        return Ok(());
    }

    println!(
        "{:<4} {:<30} {:>10} {:>12} {:>12}",
        "ID", "Name", "Vendor", "Backend", "Type"
    );
    println!("{}", "-".repeat(70));

    for (i, adapter) in all_adapters.iter().enumerate() {
        let info = adapter.get_info();
        let name_upper = info.name.to_uppercase();

        // Detect NVIDIA by vendor ID or name
        let is_nvidia = info.vendor == 0x10DE
            || name_upper.contains("NVIDIA")
            || name_upper.contains("GEFORCE")
            || name_upper.contains("RTX")
            || name_upper.contains("GTX")
            || name_upper.contains("QUADRO")
            || name_upper.contains("TESLA");

        let vendor = if is_nvidia {
            "NVIDIA"
        } else {
            match info.vendor {
                0x1002 => "AMD",
                0x10DE => "NVIDIA",
                0x8086 => "Intel",
                0x13B5 => "ARM",
                0x5143 => "Qualcomm",
                0x106B => "Apple",
                0x1414 => "Microsoft",
                _ => "Unknown",
            }
        };
        let backend = format!("{:?}", info.backend);
        let device_type = match info.device_type {
            DeviceType::DiscreteGpu => "Discrete",
            DeviceType::IntegratedGpu => "Integrated",
            DeviceType::VirtualGpu => "Virtual",
            DeviceType::Cpu => "CPU",
            DeviceType::Other => "Other",
        };
        let is_cuda = if is_nvidia { " [CUDA]" } else { "" };

        println!(
            "{:<4} {:<30} {:>10} {:>12} {:>12}{}",
            i,
            truncate_string(&info.name, 29),
            vendor,
            backend,
            device_type,
            is_cuda
        );
    }

    // Now show CUDA GPUs
    println!();
    let gpus = detect_gpus()?;

    if gpus.is_empty() {
        print_kv("CUDA GPU Count", "0");
        print_warning("No NVIDIA CUDA GPUs detected.");
    } else {
        print_kv("CUDA GPU Count", &gpus.len().to_string());

        // Recommendations
        println!();
        print_header("Recommendations");

        let best_gpu = gpus
            .iter()
            .filter(|g| g.device_type == "Discrete")
            .max_by_key(|g| g.memory_total);

        if let Some(gpu) = best_gpu {
            print_info(&format!(
                "Recommended device: GPU {} ({}) with {} memory",
                gpu.id,
                gpu.name,
                format_size(gpu.memory_total)
            ));
        } else if let Some(gpu) = gpus.first() {
            print_info(&format!(
                "Available device: GPU {} ({}) with {} memory",
                gpu.id,
                gpu.name,
                format_size(gpu.memory_total)
            ));
        }
    }

    Ok(())
}

// =============================================================================
// GPU Detection Functions
// =============================================================================

/// Detect CUDA-capable GPUs (NVIDIA only)
///
/// This function only returns NVIDIA GPUs which are CUDA-capable.
/// Uses all available backends to support various environments including WSL2.
fn detect_gpus() -> CliResult<Vec<GpuInfo>> {
    // Use all backends to support WSL2, native Windows, and native Linux
    // WSL2 exposes GPUs through various backends depending on driver setup
    let backends = Backends::all();

    let instance = Instance::new(InstanceDescriptor {
        backends,
        ..Default::default()
    });

    let adapters = instance.enumerate_adapters(backends);

    let mut gpus = Vec::new();
    let mut gpu_id = 0;

    for adapter in &adapters {
        let info = adapter.get_info();

        // Check if this is an NVIDIA GPU (CUDA-capable)
        // Method 1: Vendor ID 0x10DE = NVIDIA
        // Method 2: Name contains NVIDIA identifiers (for WSL2/D3D12 passthrough)
        let name_upper = info.name.to_uppercase();
        let is_nvidia = info.vendor == 0x10DE
            || name_upper.contains("NVIDIA")
            || name_upper.contains("GEFORCE")
            || name_upper.contains("RTX")
            || name_upper.contains("GTX")
            || name_upper.contains("QUADRO")
            || name_upper.contains("TESLA");

        if !is_nvidia {
            continue;
        }

        let device_type = match info.device_type {
            DeviceType::DiscreteGpu => "Discrete",
            DeviceType::IntegratedGpu => "Integrated",
            DeviceType::VirtualGpu => "Virtual",
            DeviceType::Cpu => "CPU",
            DeviceType::Other => "Other",
        };

        // Get the actual backend being used
        // For NVIDIA GPUs, all backends can use CUDA compute
        let backend = match info.backend {
            wgpu::Backend::Vulkan => "Vulkan/CUDA",
            wgpu::Backend::Dx12 => "DX12/CUDA",
            wgpu::Backend::Metal => "Metal",
            wgpu::Backend::Gl => "GL/CUDA", // WSL2 passthrough
            wgpu::Backend::BrowserWebGpu => "WebGPU",
            wgpu::Backend::Empty => "None",
        };

        // Get memory limits
        let limits = adapter.limits();
        let memory_total = limits.max_buffer_size;

        gpus.push(GpuInfo {
            id: gpu_id,
            name: info.name.clone(),
            vendor: "NVIDIA".to_string(),
            device_type: device_type.to_string(),
            backend: backend.to_string(),
            driver: info.driver.clone(),
            memory_total,
            is_available: true,
        });

        gpu_id += 1;
    }

    Ok(gpus)
}

// =============================================================================
// Helper Functions
// =============================================================================

fn save_gpu_selection(device_id: usize) -> CliResult<()> {
    let config_dir = std::path::PathBuf::from(".axonml");
    std::fs::create_dir_all(&config_dir)?;

    let config = serde_json::json!({
        "device_id": device_id,
    });

    std::fs::write(
        config_dir.join("gpu_config.json"),
        serde_json::to_string_pretty(&config)?,
    )?;

    Ok(())
}

fn load_gpu_selection() -> Option<usize> {
    let config_path = std::path::PathBuf::from(".axonml").join("gpu_config.json");
    let content = std::fs::read_to_string(config_path).ok()?;
    let config: serde_json::Value = serde_json::from_str(&content).ok()?;
    config.get("device_id")?.as_u64().map(|id| id as usize)
}

fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;

    if bytes >= TB {
        format!("{:.2} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

fn truncate_string(s: &str, max_len: usize) -> String {
    if max_len <= 3 {
        return "...".to_string();
    }
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_size() {
        assert!(format_size(500).contains('B'));
        assert!(format_size(1500).contains("KB"));
        assert!(format_size(1500000).contains("MB"));
        assert!(format_size(1500000000).contains("GB"));
    }

    #[test]
    fn test_truncate_string() {
        assert_eq!(truncate_string("short", 10), "short");
        // "a very long string"[..7] = "a very " -> "a very ..." (10 chars)
        assert_eq!(truncate_string("a very long string", 10), "a very ...");
        // Edge case: max_len <= 3
        assert_eq!(truncate_string("test", 3), "...");
        assert_eq!(truncate_string("test", 2), "...");
    }

    // Note: test_detect_gpus is intentionally omitted as it requires
    // actual GPU hardware and can cause SIGSEGV in test environments
    // without proper GPU drivers
}
