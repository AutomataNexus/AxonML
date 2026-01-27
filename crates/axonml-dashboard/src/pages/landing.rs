//! Public Landing Page

use leptos::*;
use leptos_router::*;

use crate::components::{icons::*, navbar::PublicNavbar};

/// Landing page for non-authenticated users
#[component]
pub fn LandingPage() -> impl IntoView {
    view! {
        <div class="landing-page">
            <PublicNavbar />

            // Hero Section
            <section class="hero">
                <div class="hero-content">
                    <h1 class="hero-title">
                        "Machine Learning"
                        <br />
                        <span class="gradient-text">"Made Simple"</span>
                    </h1>
                    <p class="hero-subtitle">
                        "Train, deploy, and monitor ML models with a beautiful, intuitive dashboard. Built for researchers and engineers who want to focus on what matters."
                    </p>
                    <div class="hero-actions">
                        <A href="/register" class="btn btn-primary btn-lg">
                            "Get Started"
                            <IconArrowRight size=IconSize::Sm />
                        </A>
                        <a href="https://github.com/AutomataNexus/AxonML" class="btn btn-secondary btn-lg" target="_blank">
                            "View on GitHub"
                        </a>
                    </div>
                </div>
                <div class="hero-visual">
                    <div class="dashboard-preview">
                        <img src="/assets/AxonML-logo.png" alt="AxonML Dashboard Preview" class="preview-image" />
                    </div>
                </div>
            </section>

            // Features Section
            <section id="features" class="features-section">
                <div class="section-header">
                    <h2>"Everything You Need"</h2>
                    <p>"A complete platform for your machine learning workflow"</p>
                </div>

                <div class="features-grid">
                    <FeatureCard
                        icon=view! { <IconActivity size=IconSize::Lg /> }
                        title="Training Monitor"
                        description="Real-time metrics, loss curves, and GPU utilization. Watch your models train with live WebSocket updates."
                    />
                    <FeatureCard
                        icon=view! { <IconBox size=IconSize::Lg /> }
                        title="Model Registry"
                        description="Version control for your models. Track experiments, compare versions, and deploy with confidence."
                    />
                    <FeatureCard
                        icon=view! { <IconServer size=IconSize::Lg /> }
                        title="Inference Server"
                        description="Deploy models instantly. Auto-scaling endpoints with latency metrics and health monitoring."
                    />
                    <FeatureCard
                        icon=view! { <IconShield size=IconSize::Lg /> }
                        title="Secure by Default"
                        description="Enterprise-grade security with MFA, WebAuthn, and role-based access control."
                    />
                    <FeatureCard
                        icon=view! { <IconZap size=IconSize::Lg /> }
                        title="Blazing Fast"
                        description="Built with Rust and WebAssembly. Sub-millisecond response times for the smoothest experience."
                    />
                    <FeatureCard
                        icon=view! { <IconDatabase size=IconSize::Lg /> }
                        title="Aegis-DB Powered"
                        description="Time-series metrics, document storage, and key-value cache all in one database."
                    />
                </div>
            </section>

            // Stats Section
            <section class="stats-section">
                <div class="stats-grid">
                    <StatCard value="<1ms" label="Dashboard Latency" />
                    <StatCard value="10x" label="Faster Deployments" />
                    <StatCard value="100%" label="Open Source" />
                    <StatCard value="24/7" label="Monitoring" />
                </div>
            </section>

            // CTA Section
            <section class="cta-section">
                <div class="cta-content">
                    <h2>"Ready to Get Started?"</h2>
                    <p>"Join developers and researchers who trust AxonML for their ML workflows."</p>
                    <A href="/register" class="btn btn-primary btn-lg">
                        "Create Free Account"
                        <IconArrowRight size=IconSize::Sm />
                    </A>
                </div>
            </section>

            // Footer
            <footer class="landing-footer">
                <div class="footer-content">
                    <div class="footer-grid">
                        // Brand & Description
                        <div class="footer-brand-section">
                            <div class="footer-brand">
                                <img src="/assets/AxonML-logo.png" alt="AxonML" class="footer-logo" />
                                <span>"AxonML"</span>
                            </div>
                            <p class="footer-description">
                                "Machine learning made simple. Train, deploy, and monitor ML models with ease."
                            </p>
                            <div class="footer-social">
                                <a href="https://github.com/AutomataNexus/AxonML" target="_blank" rel="noopener" class="social-link">
                                    <IconGithub size=IconSize::Md />
                                </a>
                            </div>
                        </div>

                        // Product Links
                        <div class="footer-section">
                            <h4 class="footer-section-title">"Product"</h4>
                            <div class="footer-links">
                                <a href="#features">"Features"</a>
                                <a href="#docs">"Documentation"</a>
                                <a href="https://github.com/AutomataNexus/AxonML" target="_blank">"GitHub"</a>
                            </div>
                        </div>

                        // Company Links
                        <div class="footer-section">
                            <h4 class="footer-section-title">"Company"</h4>
                            <div class="footer-links">
                                <a href="https://automatanexus.com">"AutomataNexus"</a>
                                <a href="https://automatanexus.com/contact">"Contact"</a>
                            </div>
                        </div>

                        // Related Products
                        <div class="footer-section">
                            <h4 class="footer-section-title">"Related Products"</h4>
                            <div class="footer-links">
                                <a href="https://aegis-db.automatanexus.com" target="_blank">"Aegis-DB"</a>
                                <a href="https://automatanexus.com/products">"All Products"</a>
                            </div>
                        </div>
                    </div>

                    <div class="footer-bottom">
                        <p class="footer-copyright">
                            "© 2026 AutomataNexus, LLC. Built with Rust, Leptos, and Aegis-DB."
                        </p>
                        <div class="footer-legal">
                            <a href="https://automatanexus.com/privacy">"Privacy"</a>
                            <a href="https://automatanexus.com/terms">"Terms"</a>
                        </div>
                    </div>
                </div>
            </footer>
        </div>
    }
}

/// Feature card component
#[component]
fn FeatureCard(
    icon: View,
    #[prop(into)] title: String,
    #[prop(into)] description: String,
) -> impl IntoView {
    view! {
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <h3 class="feature-title">{title}</h3>
            <p class="feature-description">{description}</p>
        </div>
    }
}

/// Stat card component
#[component]
fn StatCard(#[prop(into)] value: String, #[prop(into)] label: String) -> impl IntoView {
    view! {
        <div class="stat-card">
            <div class="stat-value">{value}</div>
            <div class="stat-label">{label}</div>
        </div>
    }
}
