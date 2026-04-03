//! Timeline Profiling Module
//!
//! # File
//! `crates/axonml-profile/src/timeline.rs`
//!
//! # Author
//! Andrew Jewell Sr - AutomataNexus
//!
//! # Updated
//! March 8, 2026
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Type of event in the timeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventType {
    /// Start of an operation
    Start,
    /// End of an operation
    End,
    /// Instantaneous event (marker)
    Instant,
    /// Memory allocation
    Alloc,
    /// Memory deallocation
    Free,
    /// Custom event
    Custom,
}

/// A single event in the timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// Name/category of the event
    pub name: String,
    /// Type of event
    pub event_type: EventType,
    /// Timestamp in nanoseconds from profiler start
    pub timestamp_ns: u64,
    /// Optional thread ID
    pub thread_id: Option<u64>,
    /// Optional additional data
    pub metadata: Option<String>,
}

/// Timeline profiler for recording events with timestamps.
#[derive(Debug)]
pub struct TimelineProfiler {
    /// All recorded events
    events: Vec<Event>,
    /// Start time for relative timestamps
    start_time: Instant,
    /// Maximum events to store (ring buffer behavior)
    max_events: Option<usize>,
}

impl Default for TimelineProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl TimelineProfiler {
    /// Creates a new timeline profiler.
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            start_time: Instant::now(),
            max_events: None,
        }
    }

    /// Creates a new timeline profiler with a maximum event capacity.
    pub fn with_capacity(max_events: usize) -> Self {
        Self {
            events: Vec::with_capacity(max_events.min(10000)),
            start_time: Instant::now(),
            max_events: Some(max_events),
        }
    }

    /// Records an event.
    pub fn record(&mut self, name: &str, event_type: EventType) {
        let timestamp_ns = self.start_time.elapsed().as_nanos() as u64;

        let event = Event {
            name: name.to_string(),
            event_type,
            timestamp_ns,
            thread_id: Some(Self::current_thread_id()),
            metadata: None,
        };

        self.add_event(event);
    }

    /// Records an event with metadata.
    pub fn record_with_metadata(&mut self, name: &str, event_type: EventType, metadata: &str) {
        let timestamp_ns = self.start_time.elapsed().as_nanos() as u64;

        let event = Event {
            name: name.to_string(),
            event_type,
            timestamp_ns,
            thread_id: Some(Self::current_thread_id()),
            metadata: Some(metadata.to_string()),
        };

        self.add_event(event);
    }

    /// Records a start event.
    pub fn start(&mut self, name: &str) {
        self.record(name, EventType::Start);
    }

    /// Records an end event.
    pub fn end(&mut self, name: &str) {
        self.record(name, EventType::End);
    }

    /// Records an instant event (marker).
    pub fn instant(&mut self, name: &str) {
        self.record(name, EventType::Instant);
    }

    /// Adds an event, respecting max capacity.
    fn add_event(&mut self, event: Event) {
        if let Some(max) = self.max_events {
            if self.events.len() >= max {
                // Drop oldest events to make room
                let excess = self.events.len() - max + 1;
                self.events.drain(..excess);
            }
        }
        self.events.push(event);
    }

    /// Gets a simple thread ID (hash of thread debug representation).
    fn current_thread_id() -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let thread_id = std::thread::current().id();
        let mut hasher = DefaultHasher::new();
        thread_id.hash(&mut hasher);
        hasher.finish()
    }

    /// Returns all recorded events.
    pub fn events(&self) -> &[Event] {
        &self.events
    }

    /// Returns events filtered by name.
    pub fn events_by_name(&self, name: &str) -> Vec<&Event> {
        self.events.iter().filter(|e| e.name == name).collect()
    }

    /// Returns events filtered by type.
    pub fn events_by_type(&self, event_type: EventType) -> Vec<&Event> {
        self.events
            .iter()
            .filter(|e| e.event_type == event_type)
            .collect()
    }

    /// Calculates cumulative duration for all matched start/end pairs of a given name.
    ///
    /// For operations called multiple times, returns the sum of all individual durations
    /// (not the wall time from first start to last end).
    pub fn duration(&self, name: &str) -> Option<Duration> {
        let events: Vec<_> = self.events_by_name(name);
        let starts: Vec<_> = events
            .iter()
            .filter(|e| e.event_type == EventType::Start)
            .collect();
        let ends: Vec<_> = events
            .iter()
            .filter(|e| e.event_type == EventType::End)
            .collect();

        if starts.is_empty() || ends.is_empty() {
            return None;
        }

        // Sum matched pairs (start[i], end[i])
        let mut total_ns = 0u64;
        for (s, e) in starts.iter().zip(ends.iter()) {
            if e.timestamp_ns >= s.timestamp_ns {
                total_ns += e.timestamp_ns - s.timestamp_ns;
            }
        }

        if total_ns > 0 {
            Some(Duration::from_nanos(total_ns))
        } else {
            None
        }
    }

    /// Returns the total timeline duration.
    pub fn total_duration(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Returns the number of events.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Exports events to Chrome trace format (JSON).
    #[cfg(feature = "chrome-trace")]
    pub fn to_chrome_trace(&self) -> String {
        let trace_events: Vec<_> = self
            .events
            .iter()
            .map(|e| {
                let ph = match e.event_type {
                    EventType::Start => "B",
                    EventType::End => "E",
                    EventType::Instant => "i",
                    _ => "i",
                };

                serde_json::json!({
                    "name": e.name,
                    "cat": "profile",
                    "ph": ph,
                    "ts": e.timestamp_ns / 1000, // Convert to microseconds
                    "pid": 1,
                    "tid": e.thread_id.unwrap_or(1),
                })
            })
            .collect();

        serde_json::json!({
            "traceEvents": trace_events,
            "displayTimeUnit": "ms"
        })
        .to_string()
    }

    /// Exports events to a simple JSON format.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.events)
    }

    /// Resets the timeline.
    pub fn reset(&mut self) {
        self.events.clear();
        self.start_time = Instant::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_recording() {
        let mut profiler = TimelineProfiler::new();

        profiler.start("test_op");
        std::thread::sleep(Duration::from_millis(10));
        profiler.end("test_op");

        assert_eq!(profiler.event_count(), 2);

        let duration = profiler.duration("test_op").unwrap();
        assert!(duration >= Duration::from_millis(10));
    }

    #[test]
    fn test_event_filtering() {
        let mut profiler = TimelineProfiler::new();

        profiler.start("op1");
        profiler.start("op2");
        profiler.end("op1");
        profiler.end("op2");

        let op1_events = profiler.events_by_name("op1");
        assert_eq!(op1_events.len(), 2);

        let start_events = profiler.events_by_type(EventType::Start);
        assert_eq!(start_events.len(), 2);
    }

    #[test]
    fn test_capacity_limit() {
        let mut profiler = TimelineProfiler::with_capacity(3);

        profiler.instant("event1");
        profiler.instant("event2");
        profiler.instant("event3");
        profiler.instant("event4");

        assert_eq!(profiler.event_count(), 3);
        assert_eq!(profiler.events()[0].name, "event2");
    }

    #[test]
    fn test_json_export() {
        let mut profiler = TimelineProfiler::new();
        profiler.instant("test");

        let json = profiler.to_json().unwrap();
        assert!(json.contains("test"));
    }
}
