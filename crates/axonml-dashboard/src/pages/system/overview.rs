//! System Overview Page
//!
//! Shows system information, GPU detection, and benchmarking.

use leptos::*;

use crate::api;
use crate::components::{icons::*, spinner::*};
use crate::state::use_app_state;
use crate::types::*;

/// System overview page
#[component]
pub fn SystemOverviewPage() -> impl IntoView {
    let state = use_app_state();

    let (loading, set_loading) = create_signal(true);
    let (system_info, set_system_info) = create_signal::<Option<SystemInfo>>(None);
    let (gpu_list, set_gpu_list) = create_signal::<Option<GpuListResponse>>(None);
    let (benchmark_results, set_benchmark_results) =
        create_signal::<Option<BenchmarkResponse>>(None);
    let (benchmarking, set_benchmarking) = create_signal(false);

    let state_for_effect = state.clone();

    // Initial fetch
    create_effect(move |_| {
        let state = state_for_effect.clone();
        set_loading.set(true);
        spawn_local(async move {
            // Fetch system info
            match api::system::get_info().await {
                Ok(info) => set_system_info.set(Some(info)),
                Err(e) => state.toast_error("Error", e.message),
            }

            // Fetch GPU list
            match api::system::list_gpus().await {
                Ok(gpus) => set_gpu_list.set(Some(gpus)),
                Err(e) => state.toast_error("Error", e.message),
            }

            set_loading.set(false);
        });
    });

    view! {
        <div class="page system-overview-page">
            <div class="page-header">
                <div class="header-content">
                    <h1>"System Information"</h1>
                    <p class="page-subtitle">"Hardware, GPU detection, and performance benchmarks"</p>
                </div>
            </div>

            <Show when=move || loading.get()>
                <div class="loading-state">
                    <Spinner size=SpinnerSize::Lg />
                    <p>"Loading system information..."</p>
                </div>
            </Show>

            <Show when=move || !loading.get()>
                <div class="system-grid">
                    // System Info Card
                    <SystemInfoCard info=system_info />

                    // GPU List Card
                    <GpuListCard
                        gpu_list=gpu_list
                        benchmarking=benchmarking
                        set_benchmarking=set_benchmarking
                        set_benchmark_results=set_benchmark_results
                    />

                    // Benchmark Results Card
                    <BenchmarkResultsCard results=benchmark_results />
                </div>
            </Show>
        </div>
    }
}

/// System Info Card component
#[component]
fn SystemInfoCard(info: ReadSignal<Option<SystemInfo>>) -> impl IntoView {
    view! {
        {move || info.get().map(|info| view! {
            <div class="card system-info-card">
                <div class="card-header">
                    <h3>
                        <IconCpu size=IconSize::Md />
                        <span>"System Overview"</span>
                    </h3>
                </div>
                <div class="card-body">
                    <div class="info-grid">
                        <div class="info-item">
                            <span class="label">"Platform"</span>
                            <span class="value">{info.platform.clone()}</span>
                        </div>
                        <div class="info-item">
                            <span class="label">"Architecture"</span>
                            <span class="value">{info.arch.clone()}</span>
                        </div>
                        <div class="info-item">
                            <span class="label">"CPU Cores"</span>
                            <span class="value">{info.cpu_count}</span>
                        </div>
                        <div class="info-item">
                            <span class="label">"Total Memory"</span>
                            <span class="value">{format_bytes(info.total_memory_bytes)}</span>
                        </div>
                        <div class="info-item">
                            <span class="label">"Available Memory"</span>
                            <span class="value">{format_bytes(info.available_memory_bytes)}</span>
                        </div>
                        <div class="info-item">
                            <span class="label">"AxonML Version"</span>
                            <span class="value">{info.axonml_version.clone()}</span>
                        </div>
                        <div class="info-item">
                            <span class="label">"Rust Version"</span>
                            <span class="value">{info.rust_version.clone()}</span>
                        </div>
                    </div>
                </div>
            </div>
        })}
    }
}

/// GPU List Card component
#[component]
fn GpuListCard(
    gpu_list: ReadSignal<Option<GpuListResponse>>,
    benchmarking: ReadSignal<bool>,
    set_benchmarking: WriteSignal<bool>,
    set_benchmark_results: WriteSignal<Option<BenchmarkResponse>>,
) -> impl IntoView {
    let state = use_app_state();

    view! {
        {move || gpu_list.get().map(|gpus| {
            let cuda_available = gpus.cuda_available;
            let gpus_empty = gpus.gpus.is_empty();
            let total_gpu_memory = gpus.total_gpu_memory;
            let gpu_cards: Vec<_> = gpus.gpus.clone();
            let state = state.clone();

            view! {
                <GpuListCardInner
                    cuda_available=cuda_available
                    gpus_empty=gpus_empty
                    total_gpu_memory=total_gpu_memory
                    gpu_cards=gpu_cards
                    benchmarking=benchmarking
                    set_benchmarking=set_benchmarking
                    set_benchmark_results=set_benchmark_results
                    state=state
                />
            }
        })}
    }
}

/// GPU List Card Inner component
#[component]
fn GpuListCardInner(
    cuda_available: bool,
    gpus_empty: bool,
    total_gpu_memory: u64,
    gpu_cards: Vec<GpuInfo>,
    benchmarking: ReadSignal<bool>,
    set_benchmarking: WriteSignal<bool>,
    set_benchmark_results: WriteSignal<Option<BenchmarkResponse>>,
    state: crate::state::AppState,
) -> impl IntoView {
    let on_benchmark = move |_| {
        let state = state.clone();
        set_benchmarking.set(true);
        spawn_local(async move {
            match api::system::run_benchmark().await {
                Ok(results) => {
                    set_benchmark_results.set(Some(results));
                    state
                        .toast_success("Benchmark Complete", "GPU benchmark finished successfully");
                }
                Err(e) => {
                    state.toast_error("Benchmark Failed", e.message);
                }
            }
            set_benchmarking.set(false);
        });
    };

    view! {
        <div class="card gpu-list-card">
            <div class="card-header">
                <div class="header-content">
                    <h3>
                        <IconZap size=IconSize::Md />
                        <span>"GPU Devices"</span>
                    </h3>
                    <span class={if cuda_available { "badge badge-success" } else { "badge badge-warning" }}>
                        {if cuda_available { "GPU Available" } else { "CPU Only" }}
                    </span>
                </div>
                <button
                    class="btn btn-primary"
                    on:click=on_benchmark
                    disabled=move || benchmarking.get() || gpus_empty
                >
                    {move || if benchmarking.get() {
                        view! {
                            <Spinner size=SpinnerSize::Sm />
                            <span>"Running..."</span>
                        }.into_view()
                    } else {
                        view! {
                            <IconPlay size=IconSize::Sm />
                            <span>"Run Benchmark"</span>
                        }.into_view()
                    }}
                </button>
            </div>
            <div class="card-body">
                {if gpus_empty {
                    view! {
                        <div class="empty-state">
                            <IconCpu size=IconSize::Lg class="text-muted".to_string() />
                            <p>"No GPUs detected"</p>
                            <p class="text-muted">"Training will use CPU mode. If you have a GPU, ensure drivers are installed."</p>
                        </div>
                    }.into_view()
                } else {
                    view! {
                        <>
                            <div class="gpu-cards">
                                {gpu_cards.into_iter().map(|gpu| view! {
                                    <GpuCard gpu=gpu />
                                }).collect_view()}
                            </div>
                            <div class="gpu-summary">
                                <span class="label">"Total GPU Memory"</span>
                                <span class="value">{format_bytes(total_gpu_memory)}</span>
                            </div>
                        </>
                    }.into_view()
                }}
            </div>
        </div>
    }
}

/// Benchmark Results Card component
#[component]
fn BenchmarkResultsCard(results: ReadSignal<Option<BenchmarkResponse>>) -> impl IntoView {
    view! {
        {move || results.get().map(|results| {
            let timestamp = results.timestamp.clone();
            let benchmark_results: Vec<_> = results.results.clone();

            view! {
                <div class="card benchmark-card">
                    <div class="card-header">
                        <h3>
                            <IconActivity size=IconSize::Md />
                            <span>"Benchmark Results"</span>
                        </h3>
                        <span class="timestamp">{timestamp}</span>
                    </div>
                    <div class="card-body">
                        {benchmark_results.into_iter().map(|result| view! {
                            <BenchmarkResultCard result=result />
                        }).collect_view()}
                    </div>
                </div>
            }
        })}
    }
}

/// GPU Card component
#[component]
fn GpuCard(gpu: GpuInfo) -> impl IntoView {
    view! {
        <div class="gpu-card">
            <div class="gpu-header">
                <span class="gpu-name">{gpu.name.clone()}</span>
                <span class={if gpu.is_available { "badge badge-success" } else { "badge badge-error" }}>
                    {if gpu.is_available { "Available" } else { "Unavailable" }}
                </span>
            </div>
            <div class="gpu-details">
                <div class="detail-item">
                    <span class="label">"ID"</span>
                    <span class="value">{gpu.id}</span>
                </div>
                <div class="detail-item">
                    <span class="label">"Vendor"</span>
                    <span class="value">{gpu.vendor.clone()}</span>
                </div>
                <div class="detail-item">
                    <span class="label">"Type"</span>
                    <span class="value">{gpu.device_type.clone()}</span>
                </div>
                <div class="detail-item">
                    <span class="label">"Backend"</span>
                    <span class="value">{gpu.backend.clone()}</span>
                </div>
                <div class="detail-item">
                    <span class="label">"Driver"</span>
                    <span class="value">{gpu.driver.clone()}</span>
                </div>
                <div class="detail-item">
                    <span class="label">"Memory"</span>
                    <span class="value">{format_bytes(gpu.memory_total)}</span>
                </div>
            </div>
        </div>
    }
}

/// Benchmark Result Card component
#[component]
fn BenchmarkResultCard(result: BenchmarkResult) -> impl IntoView {
    view! {
        <div class="benchmark-result">
            <div class="result-header">
                <span class="gpu-name">{result.gpu_name.clone()}</span>
                <span class="gpu-id">"GPU "{result.gpu_id}</span>
            </div>
            <div class="result-grid">
                <div class="result-section">
                    <h4>"Buffer Copy Performance"</h4>
                    <div class="result-row">
                        <span class="label">"1 MB"</span>
                        <span class="value">{format!("{:.3} ms", result.buffer_copy_1mb_ms)}</span>
                        <span class="bandwidth">{result.effective_bandwidth_1mb.clone()}</span>
                    </div>
                    <div class="result-row">
                        <span class="label">"16 MB"</span>
                        <span class="value">{format!("{:.3} ms", result.buffer_copy_16mb_ms)}</span>
                        <span class="bandwidth">{result.effective_bandwidth_16mb.clone()}</span>
                    </div>
                    <div class="result-row">
                        <span class="label">"64 MB"</span>
                        <span class="value">{format!("{:.3} ms", result.buffer_copy_64mb_ms)}</span>
                        <span class="bandwidth">{result.effective_bandwidth_64mb.clone()}</span>
                    </div>
                </div>
                <div class="result-section">
                    <h4>"Compute Performance"</h4>
                    <div class="result-row">
                        <span class="label">"Dispatch Time"</span>
                        <span class="value">{format!("{:.3} ms", result.compute_dispatch_ms)}</span>
                    </div>
                </div>
            </div>
        </div>
    }
}

fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}
