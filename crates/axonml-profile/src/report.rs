//! Report Generation Module
//!
//! Generates formatted profiling reports in various output formats.

use std::fmt;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use serde::{Serialize, Deserialize};

use crate::{Profiler, BottleneckAnalyzer, Bottleneck};
use crate::memory::MemoryProfiler;
use crate::error::{ProfileResult, ProfileError};

/// Output format for profiling reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportFormat {
    /// Plain text format
    Text,
    /// JSON format
    Json,
    /// Markdown format
    Markdown,
    /// HTML format
    Html,
}

/// A complete profiling report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileReport {
    /// Report title
    pub title: String,
    /// Total profiling duration in seconds
    pub total_duration_secs: f64,
    /// Memory statistics
    pub memory: MemorySummary,
    /// Compute statistics
    pub compute: ComputeSummary,
    /// Detected bottlenecks
    pub bottlenecks: Vec<Bottleneck>,
}

/// Summary of memory profiling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySummary {
    /// Current memory usage in bytes
    pub current_usage: usize,
    /// Peak memory usage in bytes
    pub peak_usage: usize,
    /// Total bytes allocated
    pub total_allocated: usize,
    /// Total bytes freed
    pub total_freed: usize,
    /// Number of allocations
    pub allocation_count: usize,
}

/// Summary of compute profiling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeSummary {
    /// Number of unique operations profiled
    pub operation_count: usize,
    /// Total compute time in nanoseconds
    pub total_time_ns: u64,
    /// Top operations by time
    pub top_operations: Vec<OperationSummary>,
}

/// Summary of a single operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationSummary {
    /// Operation name
    pub name: String,
    /// Total time in nanoseconds
    pub total_time_ns: u64,
    /// Number of calls
    pub call_count: usize,
    /// Average time in nanoseconds
    pub avg_time_ns: u64,
    /// Percentage of total time
    pub time_percentage: f64,
}

impl ProfileReport {
    /// Generates a report from a profiler.
    pub fn generate(profiler: &Profiler) -> Self {
        let memory_profiler = profiler.memory.read();
        let compute_profiler = profiler.compute.read();
        let timeline_profiler = profiler.timeline.read();

        let memory_stats = memory_profiler.stats();
        let compute_stats = compute_profiler.all_stats();

        // Calculate total compute time
        let total_time_ns: u64 = compute_stats.values().map(|s| s.total_time_ns).sum();

        // Get top operations
        let mut top_ops: Vec<_> = compute_stats.values().collect();
        top_ops.sort_by(|a, b| b.total_time_ns.cmp(&a.total_time_ns));
        let top_operations: Vec<OperationSummary> = top_ops
            .into_iter()
            .take(10)
            .map(|op| {
                let time_pct = if total_time_ns > 0 {
                    (op.total_time_ns as f64 / total_time_ns as f64) * 100.0
                } else {
                    0.0
                };
                OperationSummary {
                    name: op.name.clone(),
                    total_time_ns: op.total_time_ns,
                    call_count: op.call_count,
                    avg_time_ns: if op.call_count > 0 {
                        op.total_time_ns / op.call_count as u64
                    } else {
                        0
                    },
                    time_percentage: time_pct,
                }
            })
            .collect();

        // Analyze bottlenecks
        let analyzer = BottleneckAnalyzer::new();
        let bottlenecks = analyzer.analyze(&compute_stats, &memory_stats);

        Self {
            title: "Axonml Profile Report".to_string(),
            total_duration_secs: timeline_profiler.total_duration().as_secs_f64(),
            memory: MemorySummary {
                current_usage: memory_stats.current_usage,
                peak_usage: memory_stats.peak_usage,
                total_allocated: memory_stats.total_allocated,
                total_freed: memory_stats.total_freed,
                allocation_count: memory_stats.allocation_count,
            },
            compute: ComputeSummary {
                operation_count: compute_stats.len(),
                total_time_ns,
                top_operations,
            },
            bottlenecks,
        }
    }

    /// Exports the report to a file.
    pub fn export(&self, path: &Path, format: ReportFormat) -> ProfileResult<()> {
        let content = match format {
            ReportFormat::Text => self.to_text(),
            ReportFormat::Json => self.to_json()?,
            ReportFormat::Markdown => self.to_markdown(),
            ReportFormat::Html => self.to_html(),
        };

        let mut file = File::create(path)?;
        file.write_all(content.as_bytes())?;

        Ok(())
    }

    /// Converts report to plain text.
    pub fn to_text(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!("═══════════════════════════════════════════════════════════════\n"));
        output.push_str(&format!("                    {}\n", self.title));
        output.push_str(&format!("═══════════════════════════════════════════════════════════════\n\n"));

        output.push_str(&format!("Total Duration: {:.3} seconds\n\n", self.total_duration_secs));

        // Memory section
        output.push_str("─── Memory Statistics ─────────────────────────────────────────\n");
        output.push_str(&format!("  Current Usage:    {}\n", MemoryProfiler::format_bytes(self.memory.current_usage)));
        output.push_str(&format!("  Peak Usage:       {}\n", MemoryProfiler::format_bytes(self.memory.peak_usage)));
        output.push_str(&format!("  Total Allocated:  {}\n", MemoryProfiler::format_bytes(self.memory.total_allocated)));
        output.push_str(&format!("  Total Freed:      {}\n", MemoryProfiler::format_bytes(self.memory.total_freed)));
        output.push_str(&format!("  Allocations:      {}\n\n", self.memory.allocation_count));

        // Compute section
        output.push_str("─── Compute Statistics ────────────────────────────────────────\n");
        output.push_str(&format!("  Operations Profiled: {}\n", self.compute.operation_count));
        output.push_str(&format!("  Total Compute Time:  {}\n\n", Self::format_duration_ns(self.compute.total_time_ns)));

        if !self.compute.top_operations.is_empty() {
            output.push_str("  Top Operations by Time:\n");
            output.push_str("  ┌─────────────────────────────┬────────────┬──────────┬───────────┐\n");
            output.push_str("  │ Operation                   │ Total Time │ Calls    │ % Time    │\n");
            output.push_str("  ├─────────────────────────────┼────────────┼──────────┼───────────┤\n");

            for op in &self.compute.top_operations {
                let name = if op.name.len() > 27 {
                    format!("{}...", &op.name[..24])
                } else {
                    op.name.clone()
                };
                output.push_str(&format!(
                    "  │ {:<27} │ {:>10} │ {:>8} │ {:>8.1}% │\n",
                    name,
                    Self::format_duration_ns(op.total_time_ns),
                    op.call_count,
                    op.time_percentage
                ));
            }
            output.push_str("  └─────────────────────────────┴────────────┴──────────┴───────────┘\n\n");
        }

        // Bottlenecks section
        if !self.bottlenecks.is_empty() {
            output.push_str("─── Bottlenecks Detected ──────────────────────────────────────\n");
            for (i, b) in self.bottlenecks.iter().enumerate() {
                let severity = match b.severity {
                    crate::bottleneck::Severity::Critical => "CRITICAL",
                    crate::bottleneck::Severity::High => "HIGH",
                    crate::bottleneck::Severity::Medium => "MEDIUM",
                    crate::bottleneck::Severity::Low => "LOW",
                };
                output.push_str(&format!("\n  {}. [{}] {}\n", i + 1, severity, b.name));
                output.push_str(&format!("     {}\n", b.description));
                output.push_str(&format!("     → {}\n", b.suggestion));
            }
            output.push_str("\n");
        } else {
            output.push_str("─── Bottlenecks ───────────────────────────────────────────────\n");
            output.push_str("  No bottlenecks detected.\n\n");
        }

        output.push_str("═══════════════════════════════════════════════════════════════\n");

        output
    }

    /// Converts report to JSON.
    pub fn to_json(&self) -> ProfileResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| ProfileError::SerializationError(e.to_string()))
    }

    /// Converts report to Markdown.
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!("# {}\n\n", self.title));
        output.push_str(&format!("**Total Duration:** {:.3} seconds\n\n", self.total_duration_secs));

        // Memory section
        output.push_str("## Memory Statistics\n\n");
        output.push_str("| Metric | Value |\n");
        output.push_str("|--------|-------|\n");
        output.push_str(&format!("| Current Usage | {} |\n", MemoryProfiler::format_bytes(self.memory.current_usage)));
        output.push_str(&format!("| Peak Usage | {} |\n", MemoryProfiler::format_bytes(self.memory.peak_usage)));
        output.push_str(&format!("| Total Allocated | {} |\n", MemoryProfiler::format_bytes(self.memory.total_allocated)));
        output.push_str(&format!("| Total Freed | {} |\n", MemoryProfiler::format_bytes(self.memory.total_freed)));
        output.push_str(&format!("| Allocations | {} |\n\n", self.memory.allocation_count));

        // Compute section
        output.push_str("## Compute Statistics\n\n");
        output.push_str(&format!("- **Operations Profiled:** {}\n", self.compute.operation_count));
        output.push_str(&format!("- **Total Compute Time:** {}\n\n", Self::format_duration_ns(self.compute.total_time_ns)));

        if !self.compute.top_operations.is_empty() {
            output.push_str("### Top Operations by Time\n\n");
            output.push_str("| Operation | Total Time | Calls | % Time |\n");
            output.push_str("|-----------|------------|-------|--------|\n");

            for op in &self.compute.top_operations {
                output.push_str(&format!(
                    "| {} | {} | {} | {:.1}% |\n",
                    op.name,
                    Self::format_duration_ns(op.total_time_ns),
                    op.call_count,
                    op.time_percentage
                ));
            }
            output.push_str("\n");
        }

        // Bottlenecks section
        output.push_str("## Bottlenecks\n\n");
        if !self.bottlenecks.is_empty() {
            for (i, b) in self.bottlenecks.iter().enumerate() {
                let severity = match b.severity {
                    crate::bottleneck::Severity::Critical => "🔴 CRITICAL",
                    crate::bottleneck::Severity::High => "🟠 HIGH",
                    crate::bottleneck::Severity::Medium => "🟡 MEDIUM",
                    crate::bottleneck::Severity::Low => "🟢 LOW",
                };
                output.push_str(&format!("### {}. {} - {}\n\n", i + 1, severity, b.name));
                output.push_str(&format!("{}\n\n", b.description));
                output.push_str(&format!("**Suggestion:** {}\n\n", b.suggestion));
            }
        } else {
            output.push_str("No bottlenecks detected.\n");
        }

        output
    }

    /// Converts report to HTML.
    pub fn to_html(&self) -> String {
        let mut output = String::new();

        output.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        output.push_str("<meta charset=\"UTF-8\">\n");
        output.push_str(&format!("<title>{}</title>\n", self.title));
        output.push_str("<style>\n");
        output.push_str("body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }\n");
        output.push_str("h1 { color: #333; }\n");
        output.push_str("h2 { color: #555; border-bottom: 1px solid #ddd; padding-bottom: 5px; }\n");
        output.push_str("table { border-collapse: collapse; width: 100%; margin: 20px 0; }\n");
        output.push_str("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n");
        output.push_str("th { background: #f5f5f5; }\n");
        output.push_str(".bottleneck { margin: 15px 0; padding: 15px; border-radius: 5px; }\n");
        output.push_str(".critical { background: #fee; border-left: 4px solid #c00; }\n");
        output.push_str(".high { background: #fff3e0; border-left: 4px solid #f80; }\n");
        output.push_str(".medium { background: #fff9c4; border-left: 4px solid #fc0; }\n");
        output.push_str(".low { background: #e8f5e9; border-left: 4px solid #4c4; }\n");
        output.push_str("</style>\n</head>\n<body>\n");

        output.push_str(&format!("<h1>{}</h1>\n", self.title));
        output.push_str(&format!("<p><strong>Total Duration:</strong> {:.3} seconds</p>\n", self.total_duration_secs));

        // Memory section
        output.push_str("<h2>Memory Statistics</h2>\n");
        output.push_str("<table>\n<tr><th>Metric</th><th>Value</th></tr>\n");
        output.push_str(&format!("<tr><td>Current Usage</td><td>{}</td></tr>\n", MemoryProfiler::format_bytes(self.memory.current_usage)));
        output.push_str(&format!("<tr><td>Peak Usage</td><td>{}</td></tr>\n", MemoryProfiler::format_bytes(self.memory.peak_usage)));
        output.push_str(&format!("<tr><td>Total Allocated</td><td>{}</td></tr>\n", MemoryProfiler::format_bytes(self.memory.total_allocated)));
        output.push_str(&format!("<tr><td>Total Freed</td><td>{}</td></tr>\n", MemoryProfiler::format_bytes(self.memory.total_freed)));
        output.push_str(&format!("<tr><td>Allocations</td><td>{}</td></tr>\n", self.memory.allocation_count));
        output.push_str("</table>\n");

        // Compute section
        output.push_str("<h2>Compute Statistics</h2>\n");
        if !self.compute.top_operations.is_empty() {
            output.push_str("<table>\n");
            output.push_str("<tr><th>Operation</th><th>Total Time</th><th>Calls</th><th>% Time</th></tr>\n");
            for op in &self.compute.top_operations {
                output.push_str(&format!(
                    "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.1}%</td></tr>\n",
                    op.name,
                    Self::format_duration_ns(op.total_time_ns),
                    op.call_count,
                    op.time_percentage
                ));
            }
            output.push_str("</table>\n");
        }

        // Bottlenecks
        output.push_str("<h2>Bottlenecks</h2>\n");
        if !self.bottlenecks.is_empty() {
            for b in &self.bottlenecks {
                let class = match b.severity {
                    crate::bottleneck::Severity::Critical => "critical",
                    crate::bottleneck::Severity::High => "high",
                    crate::bottleneck::Severity::Medium => "medium",
                    crate::bottleneck::Severity::Low => "low",
                };
                output.push_str(&format!("<div class=\"bottleneck {}\">\n", class));
                output.push_str(&format!("<strong>{}</strong>\n", b.name));
                output.push_str(&format!("<p>{}</p>\n", b.description));
                output.push_str(&format!("<p><em>Suggestion: {}</em></p>\n", b.suggestion));
                output.push_str("</div>\n");
            }
        } else {
            output.push_str("<p>No bottlenecks detected.</p>\n");
        }

        output.push_str("</body>\n</html>\n");

        output
    }

    /// Formats nanoseconds into a human-readable string.
    fn format_duration_ns(ns: u64) -> String {
        if ns >= 1_000_000_000 {
            format!("{:.2}s", ns as f64 / 1e9)
        } else if ns >= 1_000_000 {
            format!("{:.2}ms", ns as f64 / 1e6)
        } else if ns >= 1_000 {
            format!("{:.2}µs", ns as f64 / 1e3)
        } else {
            format!("{}ns", ns)
        }
    }
}

impl fmt::Display for ProfileReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_text())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Profiler;

    #[test]
    fn test_report_generation() {
        let profiler = Profiler::new();
        profiler.start("test_op");
        std::thread::sleep(std::time::Duration::from_millis(10));
        profiler.stop("test_op");
        profiler.record_alloc("tensor", 1024);

        let report = ProfileReport::generate(&profiler);

        assert_eq!(report.compute.operation_count, 1);
        assert!(report.memory.total_allocated >= 1024);
    }

    #[test]
    fn test_text_format() {
        let profiler = Profiler::new();
        profiler.start("op1");
        profiler.stop("op1");

        let report = ProfileReport::generate(&profiler);
        let text = report.to_text();

        assert!(text.contains("Axonml Profile Report"));
        assert!(text.contains("Memory Statistics"));
        assert!(text.contains("Compute Statistics"));
    }

    #[test]
    fn test_json_format() {
        let profiler = Profiler::new();
        let report = ProfileReport::generate(&profiler);
        let json = report.to_json().unwrap();

        assert!(json.contains("title"));
        assert!(json.contains("memory"));
        assert!(json.contains("compute"));
    }

    #[test]
    fn test_markdown_format() {
        let profiler = Profiler::new();
        let report = ProfileReport::generate(&profiler);
        let md = report.to_markdown();

        assert!(md.contains("# Axonml Profile Report"));
        assert!(md.contains("## Memory Statistics"));
    }

    #[test]
    fn test_html_format() {
        let profiler = Profiler::new();
        let report = ProfileReport::generate(&profiler);
        let html = report.to_html();

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("<title>Axonml Profile Report</title>"));
    }
}
