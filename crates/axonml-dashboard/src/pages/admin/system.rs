//! System Stats Admin Page

use leptos::*;
use crate::components::icons::*;

#[component]
pub fn SystemStatsPage() -> impl IntoView {
    view! {
        <div class="page-container">
            <div class="page-header">
                <div class="page-header-content">
                    <h1 class="page-title">
                        <IconCpu />
                        <span>"System Stats"</span>
                    </h1>
                    <p class="page-description">
                        "Monitor system performance, resource usage, and health metrics"
                    </p>
                </div>
            </div>

            <div class="page-content">
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-icon">
                            <IconCpu />
                        </div>
                        <div class="stat-content">
                            <div class="stat-label">"CPU Usage"</div>
                            <div class="stat-value">"--"</div>
                        </div>
                    </div>

                    <div class="stat-card">
                        <div class="stat-icon">
                            <IconActivity />
                        </div>
                        <div class="stat-content">
                            <div class="stat-label">"Memory Usage"</div>
                            <div class="stat-value">"--"</div>
                        </div>
                    </div>

                    <div class="stat-card">
                        <div class="stat-icon">
                            <IconDatabase />
                        </div>
                        <div class="stat-content">
                            <div class="stat-label">"Storage Usage"</div>
                            <div class="stat-value">"--"</div>
                        </div>
                    </div>

                    <div class="stat-card">
                        <div class="stat-icon">
                            <IconZap />
                        </div>
                        <div class="stat-content">
                            <div class="stat-label">"Active Connections"</div>
                            <div class="stat-value">"--"</div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <h3>"System Information"</h3>
                    </div>
                    <div class="card-body">
                        <div class="empty-state">
                            <IconCpu size=IconSize::Xl />
                            <h3>"System Monitoring"</h3>
                            <p>"Connect to server for real-time metrics"</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    }
}
