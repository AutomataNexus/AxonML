//! Inference metrics for AxonML
//!
//! Tracks request latency and throughput for inference endpoints.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Latency histogram bucket
#[derive(Debug, Clone)]
pub struct LatencyBucket {
    pub le: f64,     // Less than or equal to (milliseconds)
    pub count: u64,
}

/// Endpoint metrics
#[derive(Debug, Clone)]
pub struct EndpointMetrics {
    pub endpoint_id: String,
    pub requests_total: u64,
    pub requests_success: u64,
    pub requests_error: u64,
    pub latencies: Vec<f64>,  // Last N latencies for percentile calculation
    pub created_at: Instant,
    pub last_request_at: Option<Instant>,
}

impl EndpointMetrics {
    pub fn new(endpoint_id: &str) -> Self {
        Self {
            endpoint_id: endpoint_id.to_string(),
            requests_total: 0,
            requests_success: 0,
            requests_error: 0,
            latencies: Vec::with_capacity(1000),
            created_at: Instant::now(),
            last_request_at: None,
        }
    }

    /// Record a successful request
    pub fn record_success(&mut self, latency_ms: f64) {
        self.requests_total += 1;
        self.requests_success += 1;
        self.last_request_at = Some(Instant::now());

        // Keep last 1000 latencies for percentile calculation
        if self.latencies.len() >= 1000 {
            self.latencies.remove(0);
        }
        self.latencies.push(latency_ms);
    }

    /// Record a failed request
    pub fn record_error(&mut self, latency_ms: f64) {
        self.requests_total += 1;
        self.requests_error += 1;
        self.last_request_at = Some(Instant::now());

        if self.latencies.len() >= 1000 {
            self.latencies.remove(0);
        }
        self.latencies.push(latency_ms);
    }

    /// Calculate percentile latency
    pub fn percentile(&self, p: f64) -> f64 {
        if self.latencies.is_empty() {
            return 0.0;
        }

        let mut sorted = self.latencies.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let index = ((p / 100.0) * sorted.len() as f64).ceil() as usize - 1;
        let index = index.min(sorted.len() - 1);

        sorted[index]
    }

    /// Get p50 latency
    pub fn p50(&self) -> f64 {
        self.percentile(50.0)
    }

    /// Get p95 latency
    pub fn p95(&self) -> f64 {
        self.percentile(95.0)
    }

    /// Get p99 latency
    pub fn p99(&self) -> f64 {
        self.percentile(99.0)
    }

    /// Get average latency
    pub fn avg_latency(&self) -> f64 {
        if self.latencies.is_empty() {
            return 0.0;
        }
        self.latencies.iter().sum::<f64>() / self.latencies.len() as f64
    }

    /// Get uptime since metrics were created
    pub fn uptime(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Get endpoint ID
    pub fn id(&self) -> &str {
        &self.endpoint_id
    }

    /// Get latency histogram buckets for Prometheus-style metrics
    pub fn latency_histogram(&self) -> Vec<LatencyBucket> {
        let buckets = [1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0];
        buckets.iter().map(|&le| {
            let count = self.latencies.iter().filter(|&&l| l <= le).count() as u64;
            LatencyBucket { le, count }
        }).collect()
    }

    /// Get requests per second
    pub fn rps(&self) -> f64 {
        let elapsed = self.created_at.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.requests_total as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Get error rate
    pub fn error_rate(&self) -> f64 {
        if self.requests_total > 0 {
            self.requests_error as f64 / self.requests_total as f64
        } else {
            0.0
        }
    }
}

/// Inference metrics collector
pub struct InferenceMetrics {
    metrics: Arc<RwLock<HashMap<String, EndpointMetrics>>>,
}

impl InferenceMetrics {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Initialize metrics for an endpoint
    pub async fn init(&self, endpoint_id: &str) {
        let mut metrics = self.metrics.write().await;
        metrics.insert(endpoint_id.to_string(), EndpointMetrics::new(endpoint_id));
    }

    /// Remove metrics for an endpoint
    pub async fn remove(&self, endpoint_id: &str) {
        let mut metrics = self.metrics.write().await;
        metrics.remove(endpoint_id);
    }

    /// Record a successful request
    pub async fn record_success(&self, endpoint_id: &str, latency_ms: f64) {
        let mut metrics = self.metrics.write().await;
        if let Some(m) = metrics.get_mut(endpoint_id) {
            m.record_success(latency_ms);
        }
    }

    /// Record a failed request
    pub async fn record_error(&self, endpoint_id: &str, latency_ms: f64) {
        let mut metrics = self.metrics.write().await;
        if let Some(m) = metrics.get_mut(endpoint_id) {
            m.record_error(latency_ms);
        }
    }

    /// Get metrics for an endpoint
    pub async fn get(&self, endpoint_id: &str) -> Option<EndpointMetrics> {
        let metrics = self.metrics.read().await;
        metrics.get(endpoint_id).cloned()
    }

    /// Get metrics summary for all endpoints
    pub async fn summary(&self) -> MetricsSummary {
        let metrics = self.metrics.read().await;

        let total_requests: u64 = metrics.values().map(|m| m.requests_total).sum();
        let total_success: u64 = metrics.values().map(|m| m.requests_success).sum();
        let total_errors: u64 = metrics.values().map(|m| m.requests_error).sum();

        let avg_latency: f64 = if !metrics.is_empty() {
            metrics.values().map(|m| m.avg_latency()).sum::<f64>() / metrics.len() as f64
        } else {
            0.0
        };

        MetricsSummary {
            endpoints_count: metrics.len(),
            total_requests,
            total_success,
            total_errors,
            avg_latency,
        }
    }

    /// Time a request and record metrics
    pub fn time_request(&self, endpoint_id: String) -> RequestTimer {
        RequestTimer {
            endpoint_id,
            start: Instant::now(),
            metrics: self.metrics.clone(),
        }
    }
}

impl Default for InferenceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Metrics summary
#[derive(Debug, Clone)]
pub struct MetricsSummary {
    pub endpoints_count: usize,
    pub total_requests: u64,
    pub total_success: u64,
    pub total_errors: u64,
    pub avg_latency: f64,
}

/// Request timer for automatic latency recording
pub struct RequestTimer {
    endpoint_id: String,
    start: Instant,
    metrics: Arc<RwLock<HashMap<String, EndpointMetrics>>>,
}

impl RequestTimer {
    /// Finish the request as successful
    pub async fn finish_success(self) {
        let latency_ms = self.start.elapsed().as_secs_f64() * 1000.0;
        let mut metrics = self.metrics.write().await;
        if let Some(m) = metrics.get_mut(&self.endpoint_id) {
            m.record_success(latency_ms);
        }
    }

    /// Finish the request as failed
    pub async fn finish_error(self) {
        let latency_ms = self.start.elapsed().as_secs_f64() * 1000.0;
        let mut metrics = self.metrics.write().await;
        if let Some(m) = metrics.get_mut(&self.endpoint_id) {
            m.record_error(latency_ms);
        }
    }

    /// Get elapsed time without finishing
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_endpoint_metrics() {
        let mut metrics = EndpointMetrics::new("test-endpoint");

        for i in 1..=100 {
            metrics.record_success(i as f64);
        }

        assert_eq!(metrics.requests_total, 100);
        assert_eq!(metrics.requests_success, 100);
        assert_eq!(metrics.requests_error, 0);

        // p50 should be around 50
        assert!((metrics.p50() - 50.0).abs() < 5.0);

        // p95 should be around 95
        assert!((metrics.p95() - 95.0).abs() < 5.0);
    }

    #[tokio::test]
    async fn test_inference_metrics() {
        let collector = InferenceMetrics::new();

        collector.init("ep-1").await;
        collector.record_success("ep-1", 10.0).await;
        collector.record_success("ep-1", 20.0).await;
        collector.record_error("ep-1", 5.0).await;

        let metrics = collector.get("ep-1").await.unwrap();
        assert_eq!(metrics.requests_total, 3);
        assert_eq!(metrics.requests_success, 2);
        assert_eq!(metrics.requests_error, 1);
    }
}
