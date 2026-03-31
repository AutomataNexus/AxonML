//! Rate Limiting Middleware
//!
//! # File
//! `crates/axonml-server/src/auth/rate_limit.rs`
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Updated
//! March 30, 2026
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use axum::{
    body::Body,
    extract::ConnectInfo,
    http::{Request, StatusCode},
    middleware::Next,
    response::Response,
};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Instant;

// =============================================================================
// Rate Limiter
// =============================================================================

/// Simple sliding-window rate limiter keyed by IP address.
///
/// Tracks request timestamps per IP. Requests exceeding `max_requests` within
/// `window_secs` are rejected with 429 Too Many Requests.
#[derive(Clone)]
pub struct RateLimiter {
    /// Map of IP -> list of request timestamps
    state: Arc<Mutex<HashMap<String, Vec<Instant>>>>,
    /// Maximum requests per window
    max_requests: usize,
    /// Window duration in seconds
    window_secs: u64,
}

impl RateLimiter {
    /// Creates a new rate limiter.
    ///
    /// # Arguments
    /// * `max_requests` - Maximum number of requests allowed per window
    /// * `window_secs` - Window duration in seconds
    pub fn new(max_requests: usize, window_secs: u64) -> Self {
        Self {
            state: Arc::new(Mutex::new(HashMap::new())),
            max_requests,
            window_secs,
        }
    }

    /// Creates a rate limiter suitable for auth endpoints.
    ///
    /// Default: 10 requests per 60 seconds per IP.
    pub fn auth_default() -> Self {
        Self::new(10, 60)
    }

    /// Check if a request from the given IP is allowed.
    /// Returns true if allowed, false if rate limited.
    pub fn check(&self, ip: &str) -> bool {
        let mut state = self.state.lock().unwrap();
        let now = Instant::now();
        let window = std::time::Duration::from_secs(self.window_secs);

        let timestamps = state.entry(ip.to_string()).or_default();

        // Remove expired timestamps
        timestamps.retain(|&t| now.duration_since(t) < window);

        if timestamps.len() >= self.max_requests {
            return false;
        }

        timestamps.push(now);
        true
    }
}

/// Rate limiting middleware for axum.
///
/// Extract the client IP from `ConnectInfo` or `x-forwarded-for` header,
/// then check the rate limiter.
pub async fn rate_limit_middleware(
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    request: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    // Extract rate limiter from extensions
    let limiter = request
        .extensions()
        .get::<RateLimiter>()
        .cloned();

    let ip = request
        .headers()
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.split(',').next().unwrap_or("").trim().to_string())
        .unwrap_or_else(|| addr.ip().to_string());

    if let Some(limiter) = limiter {
        if !limiter.check(&ip) {
            tracing::warn!(ip = %ip, "Rate limited");
            return Err(StatusCode::TOO_MANY_REQUESTS);
        }
    }

    Ok(next.run(request).await)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limiter_allows_within_limit() {
        let limiter = RateLimiter::new(5, 60);

        for _ in 0..5 {
            assert!(limiter.check("127.0.0.1"));
        }
    }

    #[test]
    fn test_rate_limiter_blocks_over_limit() {
        let limiter = RateLimiter::new(3, 60);

        assert!(limiter.check("127.0.0.1"));
        assert!(limiter.check("127.0.0.1"));
        assert!(limiter.check("127.0.0.1"));
        // 4th request should be blocked
        assert!(!limiter.check("127.0.0.1"));
    }

    #[test]
    fn test_rate_limiter_per_ip() {
        let limiter = RateLimiter::new(2, 60);

        assert!(limiter.check("1.1.1.1"));
        assert!(limiter.check("1.1.1.1"));
        assert!(!limiter.check("1.1.1.1"));

        // Different IP should be independent
        assert!(limiter.check("2.2.2.2"));
        assert!(limiter.check("2.2.2.2"));
        assert!(!limiter.check("2.2.2.2"));
    }

    #[test]
    fn test_rate_limiter_window_expiry() {
        let limiter = RateLimiter::new(2, 0); // 0-second window = immediately expired

        assert!(limiter.check("127.0.0.1"));
        // With 0-second window, previous request is immediately expired
        assert!(limiter.check("127.0.0.1"));
    }
}
