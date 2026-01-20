//! SVG Chart Components
//!
//! Provides line charts, bar charts, and other visualizations.

use leptos::*;

/// Data point for charts
#[derive(Debug, Clone)]
pub struct DataPoint {
    pub x: f64,
    pub y: f64,
    pub label: Option<String>,
}

/// Chart series
#[derive(Debug, Clone)]
pub struct ChartSeries {
    pub name: String,
    pub data: Vec<DataPoint>,
    pub color: String,
}

/// Chart configuration
#[derive(Debug, Clone)]
pub struct ChartConfig {
    pub width: u32,
    pub height: u32,
    pub padding: u32,
    pub show_grid: bool,
    pub show_legend: bool,
    pub show_axis_labels: bool,
    pub y_min: Option<f64>,
    pub y_max: Option<f64>,
    pub x_label: Option<String>,
    pub y_label: Option<String>,
}

impl Default for ChartConfig {
    fn default() -> Self {
        Self {
            width: 600,
            height: 300,
            padding: 40,
            show_grid: true,
            show_legend: true,
            show_axis_labels: true,
            y_min: None,
            y_max: None,
            x_label: None,
            y_label: None,
        }
    }
}

/// Calculate min and max values from series
fn calculate_bounds(series: &[ChartSeries], config: &ChartConfig) -> (f64, f64, f64, f64) {
    let mut x_min = f64::MAX;
    let mut x_max = f64::MIN;
    let mut y_min = f64::MAX;
    let mut y_max = f64::MIN;

    for s in series {
        for p in &s.data {
            x_min = x_min.min(p.x);
            x_max = x_max.max(p.x);
            y_min = y_min.min(p.y);
            y_max = y_max.max(p.y);
        }
    }

    // Apply config overrides
    if let Some(min) = config.y_min {
        y_min = min;
    }
    if let Some(max) = config.y_max {
        y_max = max;
    }

    // Add some padding to y range
    let y_range = y_max - y_min;
    if y_range > 0.0 {
        y_min -= y_range * 0.05;
        y_max += y_range * 0.05;
    } else {
        y_min -= 1.0;
        y_max += 1.0;
    }

    (x_min, x_max, y_min, y_max)
}

/// Map data coordinates to SVG coordinates
fn map_to_svg(
    x: f64,
    y: f64,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    width: u32,
    height: u32,
    padding: u32,
) -> (f64, f64) {
    let chart_width = (width - 2 * padding) as f64;
    let chart_height = (height - 2 * padding) as f64;

    let x_range = x_max - x_min;
    let y_range = y_max - y_min;

    let svg_x = if x_range > 0.0 {
        padding as f64 + ((x - x_min) / x_range) * chart_width
    } else {
        padding as f64 + chart_width / 2.0
    };

    let svg_y = if y_range > 0.0 {
        padding as f64 + chart_height - ((y - y_min) / y_range) * chart_height
    } else {
        padding as f64 + chart_height / 2.0
    };

    (svg_x, svg_y)
}

/// Line chart component
#[component]
pub fn LineChart(
    #[prop(into)] series: MaybeSignal<Vec<ChartSeries>>,
    #[prop(default = ChartConfig::default())] config: ChartConfig,
    #[prop(optional, into)] class: String,
) -> impl IntoView {
    let width = config.width;
    let height = config.height;
    let padding = config.padding;
    let show_grid = config.show_grid;
    let show_legend = config.show_legend;

    let config_for_grid = config.clone();
    let config_for_lines = config.clone();

    let series_for_grid = series.clone();
    let series_for_lines = series.clone();
    let series_for_legend = series.clone();

    view! {
        <svg
            class=format!("chart line-chart {}", class)
            viewBox=format!("0 0 {} {}", width, height)
            preserveAspectRatio="xMidYMid meet"
        >
            // Background
            <rect
                x="0"
                y="0"
                width=width
                height=height
                fill="var(--card-bg)"
                rx="8"
            />

            // Grid lines
            <Show when=move || show_grid>
                {
                    let series = series_for_grid.clone();
                    let config = config_for_grid.clone();
                    move || {
                        let series_data = series.get();
                        if series_data.is_empty() {
                            return view! { <g></g> }.into_view();
                        }

                        let (x_min, x_max, y_min, y_max) = calculate_bounds(&series_data, &config);
                        let chart_height = height - 2 * padding;
                        let chart_width = width - 2 * padding;

                        // Generate horizontal grid lines
                        let h_lines: Vec<_> = (0..=5).map(|i| {
                            let y = padding as f64 + (i as f64 / 5.0) * chart_height as f64;
                            let value = y_max - (i as f64 / 5.0) * (y_max - y_min);
                            view! {
                                <g>
                                    <line
                                        x1=padding
                                        y1=y
                                        x2=width - padding
                                        y2=y
                                        stroke="var(--text-muted)"
                                        stroke-width="0.5"
                                        stroke-dasharray="4,4"
                                        opacity="0.3"
                                    />
                                    <text
                                        x=padding - 8
                                        y=y + 4.0
                                        text-anchor="end"
                                        font-size="10"
                                        fill="var(--text-muted)"
                                    >
                                        {format!("{:.2}", value)}
                                    </text>
                                </g>
                            }
                        }).collect();

                        // Generate vertical grid lines
                        let v_lines: Vec<_> = (0..=5).map(|i| {
                            let x = padding as f64 + (i as f64 / 5.0) * chart_width as f64;
                            let value = x_min + (i as f64 / 5.0) * (x_max - x_min);
                            view! {
                                <g>
                                    <line
                                        x1=x
                                        y1=padding
                                        x2=x
                                        y2=height - padding
                                        stroke="var(--text-muted)"
                                        stroke-width="0.5"
                                        stroke-dasharray="4,4"
                                        opacity="0.2"
                                    />
                                    <text
                                        x=x
                                        y=height - padding + 16
                                        text-anchor="middle"
                                        font-size="10"
                                        fill="var(--text-muted)"
                                    >
                                        {format!("{:.1}", value)}
                                    </text>
                                </g>
                            }
                        }).collect();

                        view! { <g class="grid">{h_lines}{v_lines}</g> }.into_view()
                    }
                }
            </Show>

            // Chart lines
            {
                let series = series_for_lines.clone();
                let config = config_for_lines.clone();
                move || {
                    let series_data = series.get();
                    if series_data.is_empty() {
                        return view! { <g></g> }.into_view();
                    }

                    let (x_min, x_max, y_min, y_max) = calculate_bounds(&series_data, &config);

                let lines: Vec<_> = series_data.iter().map(|s| {
                    if s.data.is_empty() {
                        return view! { <g></g> }.into_view();
                    }

                    // Build path
                    let mut path = String::new();
                    for (i, point) in s.data.iter().enumerate() {
                        let (svg_x, svg_y) = map_to_svg(
                            point.x, point.y, x_min, x_max, y_min, y_max,
                            width, height, padding
                        );
                        if i == 0 {
                            path.push_str(&format!("M {} {}", svg_x, svg_y));
                        } else {
                            path.push_str(&format!(" L {} {}", svg_x, svg_y));
                        }
                    }

                    // Build area (filled region under line)
                    let mut area_path = path.clone();
                    if let Some(last) = s.data.last() {
                        let (last_x, _) = map_to_svg(
                            last.x, last.y, x_min, x_max, y_min, y_max,
                            width, height, padding
                        );
                        let bottom_y = height - padding;
                        let (first_x, _) = map_to_svg(
                            s.data[0].x, s.data[0].y, x_min, x_max, y_min, y_max,
                            width, height, padding
                        );
                        area_path.push_str(&format!(" L {} {} L {} {} Z", last_x, bottom_y, first_x, bottom_y));
                    }

                    let color = s.color.clone();
                    let color2 = s.color.clone();

                    // Data points
                    let points: Vec<_> = s.data.iter().map(|point| {
                        let (svg_x, svg_y) = map_to_svg(
                            point.x, point.y, x_min, x_max, y_min, y_max,
                            width, height, padding
                        );
                        let color3 = s.color.clone();
                        view! {
                            <circle
                                cx=svg_x
                                cy=svg_y
                                r="4"
                                fill=color3
                                stroke="var(--card-bg)"
                                stroke-width="2"
                                class="chart-point"
                            >
                                <title>{format!("({:.2}, {:.4})", point.x, point.y)}</title>
                            </circle>
                        }
                    }).collect();

                    view! {
                        <g class="series">
                            // Area fill
                            <path
                                d=area_path
                                fill=color.clone()
                                opacity="0.1"
                            />
                            // Line
                            <path
                                d=path
                                fill="none"
                                stroke=color2
                                stroke-width="2"
                                stroke-linecap="round"
                                stroke-linejoin="round"
                            />
                            // Points
                            {points}
                        </g>
                    }.into_view()
                }).collect();

                    view! { <g class="series-group">{lines}</g> }.into_view()
                }
            }

            // Legend
            <Show when=move || show_legend>
                {
                    let series = series_for_legend.clone();
                    move || {
                        let series_data = series.get();
                        let legend_items: Vec<_> = series_data.iter().enumerate().map(|(i, s)| {
                            let x = padding + (i as u32 * 100);
                            let color = s.color.clone();
                            let name = s.name.clone();
                            view! {
                                <g transform=format!("translate({}, {})", x, height - 15)>
                                    <rect x="0" y="0" width="12" height="12" fill=color rx="2" />
                                    <text x="16" y="10" font-size="11" fill="var(--text-secondary)">{name}</text>
                                </g>
                            }
                        }).collect();
                        view! { <g class="legend">{legend_items}</g> }
                    }
                }
            </Show>
        </svg>
    }
}

/// Bar chart component
#[component]
pub fn BarChart(
    #[prop(into)] data: MaybeSignal<Vec<(String, f64)>>,
    #[prop(default = "var(--teal)".to_string())] color: String,
    #[prop(default = ChartConfig::default())] config: ChartConfig,
    #[prop(optional, into)] class: String,
) -> impl IntoView {
    let width = config.width;
    let height = config.height;
    let padding = config.padding;

    view! {
        <svg
            class=format!("chart bar-chart {}", class)
            viewBox=format!("0 0 {} {}", width, height)
            preserveAspectRatio="xMidYMid meet"
        >
            // Background
            <rect
                x="0"
                y="0"
                width=width
                height=height
                fill="var(--card-bg)"
                rx="8"
            />

            // Bars
            {move || {
                let data_vec = data.get();
                if data_vec.is_empty() {
                    return view! { <g></g> }.into_view();
                }

                let max_value = data_vec.iter().map(|(_, v)| *v).fold(0.0f64, f64::max);
                let chart_height = (height - 2 * padding) as f64;
                let chart_width = (width - 2 * padding) as f64;
                let bar_width = chart_width / data_vec.len() as f64 * 0.7;
                let gap = chart_width / data_vec.len() as f64 * 0.15;

                let bars: Vec<_> = data_vec.iter().enumerate().map(|(i, (label, value))| {
                    let bar_height = if max_value > 0.0 {
                        (*value / max_value) * chart_height
                    } else {
                        0.0
                    };
                    let x = padding as f64 + (i as f64 * (bar_width + 2.0 * gap)) + gap;
                    let y = padding as f64 + chart_height - bar_height;
                    let color_clone = color.clone();
                    let label_clone = label.clone();

                    view! {
                        <g class="bar">
                            <rect
                                x=x
                                y=y
                                width=bar_width
                                height=bar_height
                                fill=color_clone
                                rx="4"
                                class="bar-rect"
                            >
                                <title>{format!("{}: {:.2}", label_clone, value)}</title>
                            </rect>
                            <text
                                x=x + bar_width / 2.0
                                y=height as f64 - padding as f64 + 15.0
                                text-anchor="middle"
                                font-size="10"
                                fill="var(--text-muted)"
                            >
                                {label.clone()}
                            </text>
                        </g>
                    }
                }).collect();

                view! { <g class="bars">{bars}</g> }.into_view()
            }}
        </svg>
    }
}

/// Sparkline chart (small inline chart)
#[component]
pub fn Sparkline(
    #[prop(into)] data: MaybeSignal<Vec<f64>>,
    #[prop(default = 100)] width: u32,
    #[prop(default = 30)] height: u32,
    #[prop(default = "var(--teal)".to_string())] color: String,
    #[prop(optional, into)] class: String,
) -> impl IntoView {
    view! {
        <svg
            class=format!("sparkline {}", class)
            viewBox=format!("0 0 {} {}", width, height)
            preserveAspectRatio="none"
        >
            {move || {
                let values = data.get();
                if values.is_empty() {
                    return view! { <path d="" /> }.into_view();
                }

                let min = values.iter().cloned().fold(f64::MAX, f64::min);
                let max = values.iter().cloned().fold(f64::MIN, f64::max);
                let range = if max > min { max - min } else { 1.0 };

                let mut path = String::new();
                let step = width as f64 / (values.len().max(2) - 1) as f64;

                for (i, value) in values.iter().enumerate() {
                    let x = i as f64 * step;
                    let y = height as f64 - ((value - min) / range * height as f64);
                    if i == 0 {
                        path.push_str(&format!("M {} {}", x, y));
                    } else {
                        path.push_str(&format!(" L {} {}", x, y));
                    }
                }

                let color_clone = color.clone();
                view! {
                    <path
                        d=path
                        fill="none"
                        stroke=color_clone
                        stroke-width="2"
                        stroke-linecap="round"
                        stroke-linejoin="round"
                    />
                }.into_view()
            }}
        </svg>
    }
}

/// Donut chart component
#[component]
pub fn DonutChart(
    #[prop(into)] data: MaybeSignal<Vec<(String, f64, String)>>,
    #[prop(default = 150)] size: u32,
    #[prop(default = 20)] thickness: u32,
    #[prop(optional, into)] class: String,
) -> impl IntoView {
    let radius = (size / 2 - thickness) as f64;
    let center = size as f64 / 2.0;

    view! {
        <svg
            class=format!("chart donut-chart {}", class)
            viewBox=format!("0 0 {} {}", size, size)
            preserveAspectRatio="xMidYMid meet"
        >
            {
                let data = data.clone();
                move || {
                    let data_vec = data.get();
                    let total: f64 = data_vec.iter().map(|(_, v, _)| *v).sum();
                    if total == 0.0 {
                        return view! { <g></g> }.into_view();
                    }

                    let mut start_angle = -90.0_f64;
                    let segments: Vec<_> = data_vec.iter().map(|(label, value, color)| {
                        let angle = (*value / total) * 360.0;
                        let end_angle = start_angle + angle;

                        let start_rad = start_angle.to_radians();
                        let end_rad = end_angle.to_radians();

                        let x1 = center + radius * start_rad.cos();
                        let y1 = center + radius * start_rad.sin();
                        let x2 = center + radius * end_rad.cos();
                        let y2 = center + radius * end_rad.sin();

                        let large_arc = if angle > 180.0 { 1 } else { 0 };

                        let path = format!(
                            "M {} {} A {} {} 0 {} 1 {} {}",
                            x1, y1, radius, radius, large_arc, x2, y2
                        );

                        let color_clone = color.clone();
                        let label_clone = label.clone();
                        let percentage = (*value / total) * 100.0;

                        start_angle = end_angle;

                        view! {
                            <path
                                d=path
                                fill="none"
                                stroke=color_clone
                                stroke-width=thickness
                                stroke-linecap="round"
                                class="donut-segment"
                            >
                                <title>{format!("{}: {:.1}%", label_clone, percentage)}</title>
                            </path>
                        }
                    }).collect();

                    view! { <g class="segments">{segments}</g> }.into_view()
                }
            }

            // Center text
            <text
                x=center
                y=center
                text-anchor="middle"
                dominant-baseline="middle"
                font-size="14"
                font-weight="600"
                fill="var(--text-primary)"
            >
                {
                    let data = data.clone();
                    move || {
                        let data_vec = data.get();
                        let total: f64 = data_vec.iter().map(|(_, v, _)| *v).sum();
                        format!("{:.0}", total)
                    }
                }
            </text>
        </svg>
    }
}

/// Gauge chart component
#[component]
pub fn GaugeChart(
    #[prop(into)] value: MaybeSignal<f64>,
    #[prop(default = 0.0)] min: f64,
    #[prop(default = 100.0)] max: f64,
    #[prop(default = 150)] size: u32,
    #[prop(default = "var(--teal)".to_string())] color: String,
    #[prop(optional, into)] label: String,
    #[prop(optional, into)] class: String,
) -> impl IntoView {
    let radius = (size / 2 - 15) as f64;
    let center = size as f64 / 2.0;

    let label_empty = label.is_empty();
    let label_display = label.clone();

    view! {
        <svg
            class=format!("chart gauge-chart {}", class)
            viewBox=format!("0 0 {} {}", size, size)
            preserveAspectRatio="xMidYMid meet"
        >
            // Background arc
            <path
                d={
                    let start_angle = -135.0_f64.to_radians();
                    let end_angle = -45.0_f64.to_radians();
                    let x1 = center + radius * start_angle.cos();
                    let y1 = center + radius * start_angle.sin();
                    let x2 = center + radius * end_angle.cos();
                    let y2 = center + radius * end_angle.sin();
                    format!("M {} {} A {} {} 0 1 1 {} {}", x1, y1, radius, radius, x2, y2)
                }
                fill="none"
                stroke="var(--slate-bg)"
                stroke-width="12"
                stroke-linecap="round"
            />

            // Value arc
            {move || {
                let v = value.get().clamp(min, max);
                let percentage = (v - min) / (max - min);
                let angle_range = 270.0; // degrees
                let value_angle = percentage * angle_range;

                let start_angle = -135.0_f64.to_radians();
                let end_angle = (-135.0 + value_angle).to_radians();

                let x1 = center + radius * start_angle.cos();
                let y1 = center + radius * start_angle.sin();
                let x2 = center + radius * end_angle.cos();
                let y2 = center + radius * end_angle.sin();

                let large_arc = if value_angle > 180.0 { 1 } else { 0 };
                let color_clone = color.clone();

                view! {
                    <path
                        d=format!("M {} {} A {} {} 0 {} 1 {} {}", x1, y1, radius, radius, large_arc, x2, y2)
                        fill="none"
                        stroke=color_clone
                        stroke-width="12"
                        stroke-linecap="round"
                    />
                }
            }}

            // Value text
            <text
                x=center
                y=center
                text-anchor="middle"
                dominant-baseline="middle"
                font-size="24"
                font-weight="700"
                fill="var(--text-primary)"
            >
                {move || format!("{:.1}", value.get())}
            </text>

            // Label
            <Show when=move || !label_empty>
                <text
                    x=center
                    y=center + 20.0
                    text-anchor="middle"
                    font-size="11"
                    fill="var(--text-muted)"
                >
                    {label_display.clone()}
                </text>
            </Show>
        </svg>
    }
}
