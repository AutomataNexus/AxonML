//! Enhanced System Stats Admin Page with Advanced Visualizations
//!
//! Features:
//! - Real-time system metrics
//! - Radar chart for resource overview
//! - 3D scatter plot for process correlation
//! - Stacked waveform charts for time-series data
//! - GPU monitoring with temperature and power

use leptos::*;
use wasm_bindgen::prelude::*;

use crate::api;
use crate::components::icons::*;
use crate::components::spinner::*;
use crate::state::use_app_state;
use crate::types::*;

// JavaScript bindings for Chart.js and Three.js
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = Chart, js_name = Chart)]
    type ChartJs;

    #[wasm_bindgen(constructor, js_namespace = Chart)]
    fn new(ctx: &JsValue, config: &JsValue) -> ChartJs;

    #[wasm_bindgen(method)]
    fn update(this: &ChartJs);

    #[wasm_bindgen(method)]
    fn destroy(this: &ChartJs);

    #[wasm_bindgen(method, getter)]
    fn data(this: &ChartJs) -> JsValue;

    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[component]
pub fn SystemStatsPage() -> impl IntoView {
    let state = use_app_state();

    let (loading, set_loading) = create_signal(true);
    let (realtime_metrics, set_realtime_metrics) = create_signal::<Option<RealtimeMetrics>>(None);
    let (metrics_history, set_metrics_history) =
        create_signal::<Option<SystemMetricsHistory>>(None);
    let (correlation_data, set_correlation_data) = create_signal::<Option<CorrelationData>>(None);
    let (system_info, set_system_info) = create_signal::<Option<SystemInfo>>(None);
    let (gpu_list, set_gpu_list) = create_signal::<Option<GpuListResponse>>(None);

    // Initial data fetch
    let state_clone = state.clone();
    create_effect(move |_| {
        let _state = state_clone.clone();
        set_loading.set(true);
        spawn_local(async move {
            // Fetch all data in parallel
            let (info_res, gpu_res, metrics_res, history_res, corr_res) = futures::join!(
                api::system::get_info(),
                api::system::list_gpus(),
                api::system::get_realtime_metrics(),
                api::system::get_metrics_history(),
                api::system::get_correlation_data(),
            );

            if let Ok(info) = info_res {
                set_system_info.set(Some(info));
            }
            if let Ok(gpus) = gpu_res {
                set_gpu_list.set(Some(gpus));
            }
            if let Ok(metrics) = metrics_res {
                set_realtime_metrics.set(Some(metrics));
            }
            if let Ok(history) = history_res {
                set_metrics_history.set(Some(history));
            }
            if let Ok(corr) = corr_res {
                set_correlation_data.set(Some(corr));
            }

            set_loading.set(false);
        });
    });

    // Auto-refresh metrics every 2 seconds
    let _state_for_interval = state.clone();
    create_effect(move |_| {
        use gloo_timers::callback::Interval;

        let interval = Interval::new(2000, move || {
            spawn_local(async move {
                if let Ok(metrics) = api::system::get_realtime_metrics().await {
                    set_realtime_metrics.set(Some(metrics));
                }
            });
        });

        on_cleanup(move || drop(interval));
    });

    view! {
        <div class="admin-system-page">
            <div class="page-header">
                <div class="header-content">
                    <h1>
                        <IconActivity size=IconSize::Lg />
                        <span>"System Analytics"</span>
                    </h1>
                    <p class="page-subtitle">"Real-time system monitoring with advanced visualizations"</p>
                </div>
                <div class="header-actions">
                    <span class="live-indicator">
                        <span class="pulse"></span>
                        "Live"
                    </span>
                </div>
            </div>

            <Show when=move || loading.get()>
                <div class="loading-state">
                    <Spinner size=SpinnerSize::Lg />
                    <p>"Loading system analytics..."</p>
                </div>
            </Show>

            <Show when=move || !loading.get()>
                // Quick Stats Row
                <div class="stats-row">
                    {move || realtime_metrics.get().map(|m| {
                        let cpu_trend = if m.cpu_usage_percent > 80.0 { "high" } else if m.cpu_usage_percent > 50.0 { "medium" } else { "low" };
                        let mem_trend = if m.memory_percent > 80.0 { "high" } else if m.memory_percent > 50.0 { "medium" } else { "low" };
                        let disk_trend = if m.disk_percent > 80.0 { "high" } else if m.disk_percent > 50.0 { "medium" } else { "low" };
                        view! {
                            <>
                                <QuickStatCard
                                    icon="cpu"
                                    label="CPU Usage"
                                    value=format!("{:.1}%", m.cpu_usage_percent)
                                    trend=cpu_trend
                                />
                                <QuickStatCard
                                    icon="memory"
                                    label="Memory"
                                    value=format!("{:.1}%", m.memory_percent)
                                    trend=mem_trend
                                />
                                <QuickStatCard
                                    icon="disk"
                                    label="Disk"
                                    value=format!("{:.1}%", m.disk_percent)
                                    trend=disk_trend
                                />
                                <QuickStatCard
                                    icon="process"
                                    label="Processes"
                                    value=format!("{}", m.process_count)
                                    trend="normal"
                                />
                                <QuickStatCard
                                    icon="uptime"
                                    label="Uptime"
                                    value=format_uptime(m.uptime_seconds)
                                    trend="normal"
                                />
                            </>
                        }
                    })}
                </div>

                // Main Grid
                <div class="analytics-grid">
                    // Radar Chart - Resource Overview
                    <div class="chart-card radar-card">
                        <div class="card-header">
                            <h3>
                                <IconBarChart size=IconSize::Md />
                                <span>"Resource Overview"</span>
                            </h3>
                        </div>
                        <div class="card-body">
                            <RadarChart metrics=realtime_metrics />
                        </div>
                    </div>

                    // CPU Per Core Chart
                    <div class="chart-card cpu-cores-card">
                        <div class="card-header">
                            <h3>
                                <IconCpu size=IconSize::Md />
                                <span>"CPU Cores"</span>
                            </h3>
                        </div>
                        <div class="card-body">
                            <CpuCoresChart metrics=realtime_metrics />
                        </div>
                    </div>

                    // Time Series Waveform Chart
                    <div class="chart-card waveform-card">
                        <div class="card-header">
                            <h3>
                                <IconActivity size=IconSize::Md />
                                <span>"Performance Timeline"</span>
                            </h3>
                        </div>
                        <div class="card-body">
                            <WaveformChart history=metrics_history />
                        </div>
                    </div>

                    // 3D Scatter Plot
                    <div class="chart-card scatter-card">
                        <div class="card-header">
                            <h3>
                                <IconLayers size=IconSize::Md />
                                <span>"Process Correlation"</span>
                            </h3>
                            <span class="badge badge-info">"3D Interactive"</span>
                        </div>
                        <div class="card-body">
                            <ScatterPlot3D data=correlation_data />
                        </div>
                    </div>

                    // GPU Monitoring
                    <div class="chart-card gpu-card">
                        <div class="card-header">
                            <h3>
                                <IconZap size=IconSize::Md />
                                <span>"GPU Monitoring"</span>
                            </h3>
                        </div>
                        <div class="card-body">
                            <GpuMonitoringPanel metrics=realtime_metrics gpu_list=gpu_list />
                        </div>
                    </div>

                    // Network I/O Chart
                    <div class="chart-card network-card">
                        <div class="card-header">
                            <h3>
                                <IconGlobe size=IconSize::Md />
                                <span>"Network I/O"</span>
                            </h3>
                        </div>
                        <div class="card-body">
                            <NetworkChart history=metrics_history />
                        </div>
                    </div>

                    // System Info Panel
                    <div class="chart-card info-card">
                        <div class="card-header">
                            <h3>
                                <IconInfo size=IconSize::Md />
                                <span>"System Information"</span>
                            </h3>
                        </div>
                        <div class="card-body">
                            <SystemInfoPanel info=system_info metrics=realtime_metrics />
                        </div>
                    </div>

                    // Load Average Chart
                    <div class="chart-card load-card">
                        <div class="card-header">
                            <h3>
                                <IconTrendingUp size=IconSize::Md />
                                <span>"Load Average"</span>
                            </h3>
                        </div>
                        <div class="card-body">
                            <LoadAverageChart metrics=realtime_metrics />
                        </div>
                    </div>
                </div>
            </Show>
        </div>
    }
}

#[component]
fn QuickStatCard(
    icon: &'static str,
    label: &'static str,
    value: String,
    trend: &'static str,
) -> impl IntoView {
    let trend_class = match trend {
        "high" => "trend-high",
        "medium" => "trend-medium",
        "low" => "trend-low",
        _ => "trend-normal",
    };

    view! {
        <div class=format!("quick-stat-card {}", trend_class)>
            <div class="stat-icon">
                {match icon {
                    "cpu" => view! { <IconCpu size=IconSize::Md /> }.into_view(),
                    "memory" => view! { <IconDatabase size=IconSize::Md /> }.into_view(),
                    "disk" => view! { <IconDatabase size=IconSize::Md /> }.into_view(),
                    "process" => view! { <IconLayers size=IconSize::Md /> }.into_view(),
                    "uptime" => view! { <IconClock size=IconSize::Md /> }.into_view(),
                    _ => view! { <IconActivity size=IconSize::Md /> }.into_view(),
                }}
            </div>
            <div class="stat-content">
                <span class="stat-value">{value}</span>
                <span class="stat-label">{label}</span>
            </div>
        </div>
    }
}

#[component]
fn RadarChart(metrics: ReadSignal<Option<RealtimeMetrics>>) -> impl IntoView {
    let canvas_ref = create_node_ref::<leptos::html::Canvas>();

    create_effect(move |_| {
        if let (Some(canvas), Some(m)) = (canvas_ref.get(), metrics.get()) {
            let ctx = canvas.get_context("2d").ok().flatten();
            if let Some(ctx) = ctx {
                // Create radar chart using Chart.js
                let config = js_sys::Object::new();
                js_sys::Reflect::set(&config, &"type".into(), &"radar".into()).unwrap();

                let data = js_sys::Object::new();
                let labels = js_sys::Array::new();
                labels.push(&"CPU".into());
                labels.push(&"Memory".into());
                labels.push(&"Disk".into());
                labels.push(&"Network".into());
                labels.push(&"Load".into());
                js_sys::Reflect::set(&data, &"labels".into(), &labels).unwrap();

                let datasets = js_sys::Array::new();
                let dataset = js_sys::Object::new();
                js_sys::Reflect::set(&dataset, &"label".into(), &"Resource Usage".into()).unwrap();

                let values = js_sys::Array::new();
                values.push(&JsValue::from_f64(m.cpu_usage_percent));
                values.push(&JsValue::from_f64(m.memory_percent));
                values.push(&JsValue::from_f64(m.disk_percent));
                let net_usage =
                    ((m.network_rx_bytes + m.network_tx_bytes) as f64 / 1_000_000.0).min(100.0);
                values.push(&JsValue::from_f64(net_usage));
                values.push(&JsValue::from_f64(m.load_avg_1m * 10.0));
                js_sys::Reflect::set(&dataset, &"data".into(), &values).unwrap();

                js_sys::Reflect::set(
                    &dataset,
                    &"backgroundColor".into(),
                    &"rgba(20, 184, 166, 0.2)".into(),
                )
                .unwrap();
                js_sys::Reflect::set(
                    &dataset,
                    &"borderColor".into(),
                    &"rgba(20, 184, 166, 1)".into(),
                )
                .unwrap();
                js_sys::Reflect::set(
                    &dataset,
                    &"pointBackgroundColor".into(),
                    &"rgba(20, 184, 166, 1)".into(),
                )
                .unwrap();

                datasets.push(&dataset);
                js_sys::Reflect::set(&data, &"datasets".into(), &datasets).unwrap();
                js_sys::Reflect::set(&config, &"data".into(), &data).unwrap();

                let options = js_sys::Object::new();
                js_sys::Reflect::set(&options, &"maintainAspectRatio".into(), &false.into())
                    .unwrap();
                js_sys::Reflect::set(&options, &"responsive".into(), &true.into()).unwrap();

                // Scales with light theme
                let scales = js_sys::Object::new();
                let r = js_sys::Object::new();
                js_sys::Reflect::set(&r, &"beginAtZero".into(), &true.into()).unwrap();
                js_sys::Reflect::set(&r, &"max".into(), &JsValue::from_f64(100.0)).unwrap();

                // Grid styling for light theme
                let grid = js_sys::Object::new();
                js_sys::Reflect::set(&grid, &"color".into(), &"rgba(0, 0, 0, 0.1)".into()).unwrap();
                js_sys::Reflect::set(&r, &"grid".into(), &grid).unwrap();

                // Angle lines
                let angleLines = js_sys::Object::new();
                js_sys::Reflect::set(&angleLines, &"color".into(), &"rgba(0, 0, 0, 0.1)".into())
                    .unwrap();
                js_sys::Reflect::set(&r, &"angleLines".into(), &angleLines).unwrap();

                // Point labels (light theme - dark text)
                let pointLabels = js_sys::Object::new();
                js_sys::Reflect::set(&pointLabels, &"color".into(), &"#374151".into()).unwrap();
                let font = js_sys::Object::new();
                js_sys::Reflect::set(&font, &"size".into(), &JsValue::from_f64(12.0)).unwrap();
                js_sys::Reflect::set(&font, &"weight".into(), &"500".into()).unwrap();
                js_sys::Reflect::set(&pointLabels, &"font".into(), &font).unwrap();
                js_sys::Reflect::set(&r, &"pointLabels".into(), &pointLabels).unwrap();

                // Ticks (light theme)
                let ticks = js_sys::Object::new();
                js_sys::Reflect::set(&ticks, &"color".into(), &"#6b7280".into()).unwrap();
                js_sys::Reflect::set(&ticks, &"backdropColor".into(), &"transparent".into())
                    .unwrap();
                js_sys::Reflect::set(&r, &"ticks".into(), &ticks).unwrap();

                js_sys::Reflect::set(&scales, &"r".into(), &r).unwrap();
                js_sys::Reflect::set(&options, &"scales".into(), &scales).unwrap();

                // Plugins
                let plugins = js_sys::Object::new();
                let legend = js_sys::Object::new();
                js_sys::Reflect::set(&legend, &"display".into(), &false.into()).unwrap();
                js_sys::Reflect::set(&plugins, &"legend".into(), &legend).unwrap();
                js_sys::Reflect::set(&options, &"plugins".into(), &plugins).unwrap();

                js_sys::Reflect::set(&config, &"options".into(), &options).unwrap();

                let _ = ChartJs::new(&ctx, &config);
            }
        }
    });

    view! {
        <canvas node_ref=canvas_ref class="radar-canvas"></canvas>
    }
}

#[component]
fn CpuCoresChart(metrics: ReadSignal<Option<RealtimeMetrics>>) -> impl IntoView {
    view! {
        <div class="cpu-cores-grid">
            {move || metrics.get().map(|m| {
                m.cpu_per_core.iter().enumerate().map(|(i, &usage)| {
                    let bar_height = format!("{}%", usage.min(100.0));
                    let usage_class = if usage > 80.0 { "high" } else if usage > 50.0 { "medium" } else { "low" };
                    view! {
                        <div class="core-bar-container">
                            <div class=format!("core-bar {}", usage_class) style=format!("height: {}", bar_height)></div>
                            <span class="core-label">{format!("C{}", i)}</span>
                            <span class="core-value">{format!("{:.0}%", usage)}</span>
                        </div>
                    }
                }).collect_view()
            })}
        </div>
    }
}

#[component]
fn WaveformChart(history: ReadSignal<Option<SystemMetricsHistory>>) -> impl IntoView {
    let canvas_ref = create_node_ref::<leptos::html::Canvas>();

    create_effect(move |_| {
        if let (Some(canvas), Some(h)) = (canvas_ref.get(), history.get()) {
            let ctx = canvas.get_context("2d").ok().flatten();
            if let Some(ctx) = ctx {
                let config = js_sys::Object::new();
                js_sys::Reflect::set(&config, &"type".into(), &"line".into()).unwrap();

                let data = js_sys::Object::new();
                let labels =
                    js_sys::Array::from_iter(h.timestamps.iter().map(|s| JsValue::from_str(s)));
                js_sys::Reflect::set(&data, &"labels".into(), &labels).unwrap();

                let datasets = js_sys::Array::new();

                // CPU dataset
                let cpu_dataset = create_line_dataset(
                    "CPU",
                    &h.cpu_history,
                    "rgba(20, 184, 166, 1)",
                    "rgba(20, 184, 166, 0.3)",
                );
                datasets.push(&cpu_dataset);

                // Memory dataset
                let mem_dataset = create_line_dataset(
                    "Memory",
                    &h.memory_history,
                    "rgba(249, 115, 22, 1)",
                    "rgba(249, 115, 22, 0.3)",
                );
                datasets.push(&mem_dataset);

                js_sys::Reflect::set(&data, &"datasets".into(), &datasets).unwrap();
                js_sys::Reflect::set(&config, &"data".into(), &data).unwrap();

                let options = js_sys::Object::new();
                js_sys::Reflect::set(&options, &"maintainAspectRatio".into(), &false.into())
                    .unwrap();
                js_sys::Reflect::set(&options, &"responsive".into(), &true.into()).unwrap();

                // Scales with light theme
                let scales = js_sys::Object::new();

                // Y axis
                let y = js_sys::Object::new();
                js_sys::Reflect::set(&y, &"beginAtZero".into(), &true.into()).unwrap();
                js_sys::Reflect::set(&y, &"max".into(), &JsValue::from_f64(100.0)).unwrap();
                let y_grid = js_sys::Object::new();
                js_sys::Reflect::set(&y_grid, &"color".into(), &"rgba(0, 0, 0, 0.06)".into())
                    .unwrap();
                js_sys::Reflect::set(&y, &"grid".into(), &y_grid).unwrap();
                let y_ticks = js_sys::Object::new();
                js_sys::Reflect::set(&y_ticks, &"color".into(), &"#6b7280".into()).unwrap();
                js_sys::Reflect::set(&y, &"ticks".into(), &y_ticks).unwrap();
                js_sys::Reflect::set(&scales, &"y".into(), &y).unwrap();

                // X axis
                let x = js_sys::Object::new();
                let x_grid = js_sys::Object::new();
                js_sys::Reflect::set(&x_grid, &"display".into(), &false.into()).unwrap();
                js_sys::Reflect::set(&x, &"grid".into(), &x_grid).unwrap();
                let x_ticks = js_sys::Object::new();
                js_sys::Reflect::set(&x_ticks, &"color".into(), &"#6b7280".into()).unwrap();
                js_sys::Reflect::set(&x_ticks, &"maxRotation".into(), &JsValue::from_f64(0.0))
                    .unwrap();
                js_sys::Reflect::set(&x, &"ticks".into(), &x_ticks).unwrap();
                js_sys::Reflect::set(&scales, &"x".into(), &x).unwrap();

                js_sys::Reflect::set(&options, &"scales".into(), &scales).unwrap();

                // Plugins
                let plugins = js_sys::Object::new();
                let legend = js_sys::Object::new();
                js_sys::Reflect::set(&legend, &"position".into(), &"top".into()).unwrap();
                let legend_labels = js_sys::Object::new();
                js_sys::Reflect::set(&legend_labels, &"color".into(), &"#374151".into()).unwrap();
                js_sys::Reflect::set(&legend_labels, &"usePointStyle".into(), &true.into())
                    .unwrap();
                js_sys::Reflect::set(&legend_labels, &"padding".into(), &JsValue::from_f64(16.0))
                    .unwrap();
                js_sys::Reflect::set(&legend, &"labels".into(), &legend_labels).unwrap();
                js_sys::Reflect::set(&plugins, &"legend".into(), &legend).unwrap();
                js_sys::Reflect::set(&options, &"plugins".into(), &plugins).unwrap();

                // Interaction
                let interaction = js_sys::Object::new();
                js_sys::Reflect::set(&interaction, &"intersect".into(), &false.into()).unwrap();
                js_sys::Reflect::set(&interaction, &"mode".into(), &"index".into()).unwrap();
                js_sys::Reflect::set(&options, &"interaction".into(), &interaction).unwrap();

                js_sys::Reflect::set(&config, &"options".into(), &options).unwrap();

                let _ = ChartJs::new(&ctx, &config);
            }
        }
    });

    view! {
        <canvas node_ref=canvas_ref class="waveform-canvas"></canvas>
    }
}

fn create_line_dataset(label: &str, data: &[f64], border_color: &str, bg_color: &str) -> JsValue {
    let dataset = js_sys::Object::new();
    js_sys::Reflect::set(&dataset, &"label".into(), &label.into()).unwrap();
    let values = js_sys::Array::from_iter(data.iter().map(|&v| JsValue::from_f64(v)));
    js_sys::Reflect::set(&dataset, &"data".into(), &values).unwrap();
    js_sys::Reflect::set(&dataset, &"borderColor".into(), &border_color.into()).unwrap();
    js_sys::Reflect::set(&dataset, &"backgroundColor".into(), &bg_color.into()).unwrap();
    js_sys::Reflect::set(&dataset, &"fill".into(), &true.into()).unwrap();
    js_sys::Reflect::set(&dataset, &"tension".into(), &JsValue::from_f64(0.4)).unwrap();
    js_sys::Reflect::set(&dataset, &"pointRadius".into(), &JsValue::from_f64(0.0)).unwrap();
    dataset.into()
}

#[component]
fn ScatterPlot3D(data: ReadSignal<Option<CorrelationData>>) -> impl IntoView {
    let container_ref = create_node_ref::<leptos::html::Div>();

    create_effect(move |_| {
        if let (Some(container), Some(corr)) = (container_ref.get(), data.get()) {
            // Initialize Three.js scene
            let width = container.client_width() as f64;
            let height = 380.0;

            let init_code = format!(
                r#"
                (function() {{
                    const container = document.getElementById('scatter-3d-container');
                    if (!container || container.querySelector('canvas')) return;

                    const scene = new THREE.Scene();
                    scene.background = new THREE.Color(0xffffff);

                    const camera = new THREE.PerspectiveCamera(60, {width} / {height}, 0.1, 1000);
                    camera.position.set(80, 60, 80);

                    const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
                    renderer.setSize({width}, {height});
                    renderer.setPixelRatio(window.devicePixelRatio);
                    container.appendChild(renderer.domElement);

                    const controls = new THREE.OrbitControls(camera, renderer.domElement);
                    controls.enableDamping = true;
                    controls.dampingFactor = 0.05;
                    controls.autoRotate = true;
                    controls.autoRotateSpeed = 0.5;

                    // Add subtle ambient light
                    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
                    scene.add(ambientLight);
                    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
                    directionalLight.position.set(50, 100, 50);
                    scene.add(directionalLight);

                    // Create a nice ground plane with grid
                    const gridSize = 100;
                    const gridDivisions = 10;
                    const gridHelper = new THREE.GridHelper(gridSize, gridDivisions, 0xd1d5db, 0xe5e7eb);
                    gridHelper.position.y = 0;
                    scene.add(gridHelper);

                    // Axis lines with labels (X=CPU, Y=Runtime, Z=Memory)
                    const axisLength = 55;
                    const axisMaterial = new THREE.LineBasicMaterial({{ color: 0x9ca3af, linewidth: 1 }});

                    // X axis (red-ish for CPU)
                    const xAxisGeo = new THREE.BufferGeometry().setFromPoints([
                        new THREE.Vector3(0, 0, 0),
                        new THREE.Vector3(axisLength, 0, 0)
                    ]);
                    const xAxis = new THREE.Line(xAxisGeo, new THREE.LineBasicMaterial({{ color: 0xef4444 }}));
                    scene.add(xAxis);

                    // Y axis (teal for Runtime)
                    const yAxisGeo = new THREE.BufferGeometry().setFromPoints([
                        new THREE.Vector3(0, 0, 0),
                        new THREE.Vector3(0, axisLength, 0)
                    ]);
                    const yAxis = new THREE.Line(yAxisGeo, new THREE.LineBasicMaterial({{ color: 0x14b8a6 }}));
                    scene.add(yAxis);

                    // Z axis (orange for Memory)
                    const zAxisGeo = new THREE.BufferGeometry().setFromPoints([
                        new THREE.Vector3(0, 0, 0),
                        new THREE.Vector3(0, 0, axisLength)
                    ]);
                    const zAxis = new THREE.Line(zAxisGeo, new THREE.LineBasicMaterial({{ color: 0xf59e0b }}));
                    scene.add(zAxis);

                    // Add points using BufferGeometry for performance
                    const points = {points_json};
                    const positions = [];
                    const colors = [];
                    const colorMap = {{
                        'high-cpu': [0.94, 0.27, 0.27],    // #ef4444
                        'high-memory': [0.96, 0.62, 0.04], // #f59e0b
                        'normal': [0.08, 0.72, 0.65]       // #14b8a6
                    }};

                    // Scale factors to fit nicely in view
                    const scaleX = 0.5;  // CPU (0-100) -> 0-50
                    const scaleY = 0.05; // Runtime (0-1000 min) -> 0-50
                    const scaleZ = 0.025; // Memory (0-2000 MB) -> 0-50

                    points.forEach(p => {{
                        positions.push(p.x * scaleX, p.z * scaleY, p.y * scaleZ);
                        const c = colorMap[p.category] || colorMap['normal'];
                        colors.push(c[0], c[1], c[2]);
                    }});

                    const pointsGeometry = new THREE.BufferGeometry();
                    pointsGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
                    pointsGeometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

                    const pointsMaterial = new THREE.PointsMaterial({{
                        size: 3,
                        vertexColors: true,
                        sizeAttenuation: true,
                        transparent: true,
                        opacity: 0.85
                    }});

                    const pointCloud = new THREE.Points(pointsGeometry, pointsMaterial);
                    scene.add(pointCloud);

                    // Animation loop
                    function animate() {{
                        requestAnimationFrame(animate);
                        controls.update();
                        renderer.render(scene, camera);
                    }}
                    animate();

                    // Handle resize
                    window.addEventListener('resize', () => {{
                        const newWidth = container.clientWidth;
                        camera.aspect = newWidth / {height};
                        camera.updateProjectionMatrix();
                        renderer.setSize(newWidth, {height});
                    }});
                }})();
            "#,
                width = width,
                height = height,
                points_json = serde_json::to_string(&corr.points).unwrap_or_default()
            );

            let _ = js_sys::eval(&init_code);
        }
    });

    view! {
        <div node_ref=container_ref id="scatter-3d-container" class="scatter-3d-container">
            <div class="scatter-legend">
                <span class="legend-item"><span class="dot cpu"></span>"CPU (X)"</span>
                <span class="legend-item"><span class="dot memory"></span>"Memory (Z)"</span>
                <span class="legend-item"><span class="dot runtime"></span>"Runtime (Y)"</span>
            </div>
        </div>
    }
}

#[component]
fn GpuMonitoringPanel(
    metrics: ReadSignal<Option<RealtimeMetrics>>,
    gpu_list: ReadSignal<Option<GpuListResponse>>,
) -> impl IntoView {
    view! {
        <div class="gpu-monitoring">
            {move || {
                let gpu_metrics = metrics.get().map(|m| m.gpu_metrics).unwrap_or_default();
                let gpus = gpu_list.get().map(|g| g.gpus).unwrap_or_default();

                if gpus.is_empty() && gpu_metrics.is_empty() {
                    view! {
                        <div class="no-gpu">
                            <IconCpu size=IconSize::Lg />
                            <p>"No GPU detected"</p>
                        </div>
                    }.into_view()
                } else {
                    let metrics_map: std::collections::HashMap<usize, GpuMetrics> =
                        gpu_metrics.into_iter().map(|m| (m.id, m)).collect();

                    gpus.into_iter().map(|gpu| {
                        let metric = metrics_map.get(&gpu.id);
                        view! {
                            <div class="gpu-panel">
                                <div class="gpu-header">
                                    <span class="gpu-name">{gpu.name.clone()}</span>
                                    <span class="gpu-badge">{gpu.backend.clone()}</span>
                                </div>
                                <div class="gpu-metrics">
                                    <div class="metric-row">
                                        <span class="metric-label">"Utilization"</span>
                                        <div class="metric-bar">
                                            <div
                                                class="metric-fill utilization"
                                                style=format!("width: {}%", metric.map(|m| m.utilization_percent).unwrap_or(0.0))
                                            ></div>
                                        </div>
                                        <span class="metric-value">{format!("{:.0}%", metric.map(|m| m.utilization_percent).unwrap_or(0.0))}</span>
                                    </div>
                                    <div class="metric-row">
                                        <span class="metric-label">"Memory"</span>
                                        <div class="metric-bar">
                                            <div
                                                class="metric-fill memory"
                                                style=format!("width: {}%",
                                                    metric.map(|m| (m.memory_used_mb as f64 / m.memory_total_mb as f64) * 100.0).unwrap_or(0.0)
                                                )
                                            ></div>
                                        </div>
                                        <span class="metric-value">
                                            {format!("{} / {} MB",
                                                metric.map(|m| m.memory_used_mb).unwrap_or(0),
                                                metric.map(|m| m.memory_total_mb).unwrap_or(gpu.memory_total / 1024 / 1024)
                                            )}
                                        </span>
                                    </div>
                                    <div class="metric-row">
                                        <span class="metric-label">"Temperature"</span>
                                        <span class="metric-value temp">
                                            {format!("{:.0}°C", metric.map(|m| m.temperature_c).unwrap_or(0.0))}
                                        </span>
                                    </div>
                                    <div class="metric-row">
                                        <span class="metric-label">"Power"</span>
                                        <span class="metric-value power">
                                            {format!("{:.1}W", metric.map(|m| m.power_watts).unwrap_or(0.0))}
                                        </span>
                                    </div>
                                </div>
                            </div>
                        }
                    }).collect_view()
                }
            }}
        </div>
    }
}

#[component]
fn NetworkChart(history: ReadSignal<Option<SystemMetricsHistory>>) -> impl IntoView {
    let canvas_ref = create_node_ref::<leptos::html::Canvas>();

    create_effect(move |_| {
        if let (Some(canvas), Some(h)) = (canvas_ref.get(), history.get()) {
            let ctx = canvas.get_context("2d").ok().flatten();
            if let Some(ctx) = ctx {
                let config = js_sys::Object::new();
                js_sys::Reflect::set(&config, &"type".into(), &"line".into()).unwrap();

                let data = js_sys::Object::new();
                let labels =
                    js_sys::Array::from_iter(h.timestamps.iter().map(|s| JsValue::from_str(s)));
                js_sys::Reflect::set(&data, &"labels".into(), &labels).unwrap();

                let datasets = js_sys::Array::new();

                let rx_dataset = create_line_dataset(
                    "Download",
                    &h.network_rx,
                    "rgba(59, 130, 246, 1)",
                    "rgba(59, 130, 246, 0.3)",
                );
                datasets.push(&rx_dataset);

                let tx_dataset = create_line_dataset(
                    "Upload",
                    &h.network_tx,
                    "rgba(168, 85, 247, 1)",
                    "rgba(168, 85, 247, 0.3)",
                );
                datasets.push(&tx_dataset);

                js_sys::Reflect::set(&data, &"datasets".into(), &datasets).unwrap();
                js_sys::Reflect::set(&config, &"data".into(), &data).unwrap();

                let options = js_sys::Object::new();
                js_sys::Reflect::set(&options, &"maintainAspectRatio".into(), &false.into())
                    .unwrap();
                js_sys::Reflect::set(&options, &"responsive".into(), &true.into()).unwrap();

                // Scales with light theme
                let scales = js_sys::Object::new();

                let y = js_sys::Object::new();
                js_sys::Reflect::set(&y, &"beginAtZero".into(), &true.into()).unwrap();
                let y_grid = js_sys::Object::new();
                js_sys::Reflect::set(&y_grid, &"color".into(), &"rgba(0, 0, 0, 0.06)".into())
                    .unwrap();
                js_sys::Reflect::set(&y, &"grid".into(), &y_grid).unwrap();
                let y_ticks = js_sys::Object::new();
                js_sys::Reflect::set(&y_ticks, &"color".into(), &"#6b7280".into()).unwrap();
                js_sys::Reflect::set(&y, &"ticks".into(), &y_ticks).unwrap();
                js_sys::Reflect::set(&scales, &"y".into(), &y).unwrap();

                let x = js_sys::Object::new();
                let x_grid = js_sys::Object::new();
                js_sys::Reflect::set(&x_grid, &"display".into(), &false.into()).unwrap();
                js_sys::Reflect::set(&x, &"grid".into(), &x_grid).unwrap();
                let x_ticks = js_sys::Object::new();
                js_sys::Reflect::set(&x_ticks, &"color".into(), &"#6b7280".into()).unwrap();
                js_sys::Reflect::set(&x, &"ticks".into(), &x_ticks).unwrap();
                js_sys::Reflect::set(&scales, &"x".into(), &x).unwrap();

                js_sys::Reflect::set(&options, &"scales".into(), &scales).unwrap();

                // Plugins
                let plugins = js_sys::Object::new();
                let legend = js_sys::Object::new();
                js_sys::Reflect::set(&legend, &"position".into(), &"top".into()).unwrap();
                let legend_labels = js_sys::Object::new();
                js_sys::Reflect::set(&legend_labels, &"color".into(), &"#374151".into()).unwrap();
                js_sys::Reflect::set(&legend_labels, &"usePointStyle".into(), &true.into())
                    .unwrap();
                js_sys::Reflect::set(&legend, &"labels".into(), &legend_labels).unwrap();
                js_sys::Reflect::set(&plugins, &"legend".into(), &legend).unwrap();
                js_sys::Reflect::set(&options, &"plugins".into(), &plugins).unwrap();

                js_sys::Reflect::set(&config, &"options".into(), &options).unwrap();

                let _ = ChartJs::new(&ctx, &config);
            }
        }
    });

    view! {
        <canvas node_ref=canvas_ref class="network-canvas"></canvas>
    }
}

#[component]
fn SystemInfoPanel(
    info: ReadSignal<Option<SystemInfo>>,
    metrics: ReadSignal<Option<RealtimeMetrics>>,
) -> impl IntoView {
    view! {
        <div class="system-info-grid">
            {move || info.get().map(|i| view! {
                <>
                    <div class="info-row">
                        <span class="info-label">"Platform"</span>
                        <span class="info-value">{i.platform.clone()}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">"Architecture"</span>
                        <span class="info-value">{i.arch.clone()}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">"CPU Cores"</span>
                        <span class="info-value">{i.cpu_count}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">"Total Memory"</span>
                        <span class="info-value">{format_bytes(i.total_memory_bytes)}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">"AxonML Version"</span>
                        <span class="info-value">{i.axonml_version.clone()}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">"Rust Version"</span>
                        <span class="info-value">{i.rust_version.clone()}</span>
                    </div>
                </>
            })}
            {move || metrics.get().map(|m| view! {
                <>
                    <div class="info-row">
                        <span class="info-label">"Available Memory"</span>
                        <span class="info-value">{format_bytes(m.memory_total_bytes - m.memory_used_bytes)}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">"Network RX"</span>
                        <span class="info-value">{format_bytes(m.network_rx_bytes)}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">"Network TX"</span>
                        <span class="info-value">{format_bytes(m.network_tx_bytes)}</span>
                    </div>
                </>
            })}
        </div>
    }
}

#[component]
fn LoadAverageChart(metrics: ReadSignal<Option<RealtimeMetrics>>) -> impl IntoView {
    view! {
        <div class="load-average-display">
            {move || metrics.get().map(|m| view! {
                <>
                    <div class="load-item">
                        <span class="load-value">{format!("{:.2}", m.load_avg_1m)}</span>
                        <span class="load-label">"1 min"</span>
                    </div>
                    <div class="load-item">
                        <span class="load-value">{format!("{:.2}", m.load_avg_5m)}</span>
                        <span class="load-label">"5 min"</span>
                    </div>
                    <div class="load-item">
                        <span class="load-value">{format!("{:.2}", m.load_avg_15m)}</span>
                        <span class="load-label">"15 min"</span>
                    </div>
                </>
            })}
        </div>
    }
}

fn format_bytes(bytes: u64) -> String {
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
        format!("{} B", bytes)
    }
}

fn format_uptime(seconds: u64) -> String {
    let days = seconds / 86400;
    let hours = (seconds % 86400) / 3600;
    let mins = (seconds % 3600) / 60;

    if days > 0 {
        format!("{}d {}h", days, hours)
    } else if hours > 0 {
        format!("{}h {}m", hours, mins)
    } else {
        format!("{}m", mins)
    }
}
