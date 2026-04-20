//! claude_trace_export — Claude Code session traces → DeepSeek-R1 SFT corpus.
//!
//! Walks `~/.claude/projects/<project>/*.jsonl`, reconstructs each session
//! from its event log, filters to sessions worth training on, and renders
//! the turn sequence into DeepSeek-R1's native chat template with inline
//! `<tool_use>{json}</tool_use>` blocks matching the nexus-serve parser
//! convention (task #52 / #20 in nexus-serve).
//!
//! The output is two JSONL files — `corpus_train.jsonl` + `corpus_val.jsonl`,
//! 95/5 split by session — each line a `{"text": "<full rendered prompt>"}`
//! object ready for PEFT/Unsloth LoRA fine-tuning of
//! `DeepSeek-R1-Distill-Qwen-7B` on an A100 80GB.
//!
//! ## Why DeepSeek-R1 template at corpus-build time?
//!
//! We are training on top of R1-Distill, which was SFT'd with DeepSeek's
//! chat template (full-width-pipe `<｜User｜>` / `<｜Assistant｜>` markers,
//! `<｜begin▁of▁sentence｜>` BOS, `<｜end▁of▁sentence｜>` turn terminator).
//! nexus-serve already dispatches to `render_deepseek_r1` when the GGUF's
//! `general.name` contains "DeepSeek" + "R1"/"Distill", so if we bake the
//! same template into the SFT corpus the trained LoRA → merged → GGUF will
//! deploy with no additional post-processing.
//!
//! ## Why inline `<tool_use>{json}</tool_use>` blocks?
//!
//! nexus-serve's Messages-API parser recognizes `<tool_use>{json}</tool_use>`
//! as a signal to emit a `tool_use` content block with `stop_reason="tool_use"`.
//! Training on this exact surface form means the trained Oracle speaks
//! nexus-serve's dialect natively — no template-mismatch footgun at deploy
//! time. Tool *results* are rendered as plain text inside the `<｜User｜>`
//! turn (matching how nexus-serve's `build_prompt` flattens tool_result
//! blocks today).
//!
//! ## Quality filters
//!
//! - Drop sessions with fewer than 2 user + 2 assistant turns.
//! - Drop sessions with no `tool_use` blocks — goal is to teach **agentic**
//!   tool-calling, not pure chat.
//! - Drop sidechain events (`isSidechain=true` — subagent threads that
//!   aren't the main agent loop).
//! - Truncate individual tool_result contents to 8 KB (a single cat of a
//!   100 MB log file would otherwise dominate one training sample).
//! - Drop sessions whose rendered text exceeds `--max-chars` (default 64 KB,
//!   ≈ 16K tokens at ~4 chars/tok — fits R1-Distill's 32K context with
//!   room for loss masking & padding).
//!
//! ## Usage
//!
//! ```bash
//! claude_trace_export \
//!   --projects-dir ~/.claude/projects \
//!   --output-dir   /opt/datasets/oracle-lora \
//!   --val-split    0.05 \
//!   --seed         42
//! ```
//!
//! Emits:
//! - `corpus_train.jsonl` + `corpus_val.jsonl`
//! - `corpus_stats.json` — `{sessions_total, sessions_kept, turns_total,
//!   tool_uses_total, chars_train, chars_val, rejected: {reason: count}}`.
//!
//! # File
//! `llm-training/src/bin/claude_trace_export.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Created
//! April 19, 2026
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of
//! any kind, express or implied. The author and AutomataNexus shall not be
//! held liable for any damages arising from the use of this software.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use rand::prelude::*;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use walkdir::WalkDir;

// =============================================================================
// CLI
// =============================================================================

#[derive(Parser, Debug)]
#[command(about = "Export Claude Code session traces to DeepSeek-R1 SFT JSONL")]
struct Args {
    /// Root of Claude Code's per-project trace directory. Files live at
    /// `{projects_dir}/<project-slug>/<session-uuid>.jsonl`.
    #[arg(long, default_value = "/home/devops/.claude/projects")]
    projects_dir: PathBuf,

    /// Where to write `corpus_train.jsonl`, `corpus_val.jsonl`, and
    /// `corpus_stats.json`. Created if it doesn't exist.
    #[arg(long)]
    output_dir: PathBuf,

    /// Fraction of sessions routed to the validation split.
    #[arg(long, default_value_t = 0.05)]
    val_split: f64,

    /// RNG seed for the train/val split. Deterministic for reproducible
    /// dataset builds across re-runs.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Reject sessions whose rendered prompt exceeds this many characters.
    /// Default 64 KB ≈ 16K tokens at 4 chars/token, leaving R1-Distill's
    /// 32K context headroom for loss-masking & padding during training.
    #[arg(long, default_value_t = 131_072)]
    max_chars: usize,

    /// Truncate any single tool_result payload to this many characters
    /// before rendering. Protects against Read-of-giant-file sessions
    /// blowing a single training sample out of the context window.
    #[arg(long, default_value_t = 8_192)]
    max_tool_result_chars: usize,

    /// Drop sessions whose last assistant turn contains "[Request
    /// interrupted by user]" or similar interrupt markers. Those are
    /// unfinished behaviors we don't want to teach.
    #[arg(long, default_value_t = true)]
    drop_interrupted: bool,
}

// =============================================================================
// Raw event shape (subset we actually consume).
// =============================================================================
//
// Claude Code's JSONL event log has many fields; we only deserialize what we
// need to reconstruct the conversation. `#[serde(default)]` on every field
// makes the parser tolerant to schema drift across Claude Code versions
// (2.1.x + 2.2.x have slightly different event types).

#[derive(Deserialize, Debug)]
struct RawEvent {
    #[serde(default, rename = "type")]
    event_type: String,
    #[serde(default)]
    message: Option<Value>,
    #[serde(default, rename = "isSidechain")]
    is_sidechain: bool,
}

// =============================================================================
// Normalized turn model.
// =============================================================================

#[derive(Debug)]
enum Block {
    Text(String),
    ToolUse {
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

#[derive(Debug)]
struct Turn {
    role: Role,
    blocks: Vec<Block>,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Role {
    User,
    Assistant,
}

// =============================================================================
// Per-session stats rolled up into the global stats file.
// =============================================================================

#[derive(Default, Serialize)]
struct Stats {
    sessions_total: usize,
    sessions_kept: usize,
    turns_total: usize,
    tool_uses_total: usize,
    chars_train: usize,
    chars_val: usize,
    rejected: BTreeMap<String, usize>,
}

impl Stats {
    fn reject(&mut self, reason: &str) {
        *self.rejected.entry(reason.to_string()).or_insert(0) += 1;
    }
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<()> {
    let args = Args::parse();
    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("creating {}", args.output_dir.display()))?;

    let mut stats = Stats::default();
    let mut samples: Vec<String> = Vec::new();

    // Walk the full tree — Claude Code stores top-level session files at
    // `<project>/<uuid>.jsonl` (depth 2) AND subagent traces at
    // `<project>/<uuid>/subagents/<file>.jsonl` (depth 4). Both are
    // legitimate agentic conversations to train on.
    for entry in WalkDir::new(&args.projects_dir)
        .min_depth(2)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("jsonl") {
            continue;
        }
        stats.sessions_total += 1;

        match process_session(path, &args, &mut stats) {
            Ok(Some(rendered)) => {
                samples.push(rendered);
                stats.sessions_kept += 1;
            }
            Ok(None) => {} // rejected with reason already recorded
            Err(e) => {
                eprintln!("warn: {} — {e}", path.display());
                stats.reject("parse_error");
            }
        }
    }

    // Deterministic train/val split.
    let mut rng = StdRng::seed_from_u64(args.seed);
    samples.shuffle(&mut rng);
    let val_n = ((samples.len() as f64) * args.val_split).round() as usize;
    let val_n = val_n.max(1).min(samples.len().saturating_sub(1));
    let (val, train) = samples.split_at(val_n);

    let train_path = args.output_dir.join("corpus_train.jsonl");
    let val_path = args.output_dir.join("corpus_val.jsonl");
    stats.chars_train = write_jsonl(&train_path, train)?;
    stats.chars_val = write_jsonl(&val_path, val)?;

    let stats_path = args.output_dir.join("corpus_stats.json");
    fs::write(&stats_path, serde_json::to_string_pretty(&stats)? + "\n")?;

    println!(
        "done: {} sessions → {} train + {} val (rejected {}); see {}",
        stats.sessions_total,
        train.len(),
        val.len(),
        stats.sessions_total - stats.sessions_kept,
        stats_path.display()
    );
    for (reason, n) in &stats.rejected {
        println!("  reject[{reason}] = {n}");
    }
    Ok(())
}

// =============================================================================
// Session parsing & filtering
// =============================================================================

fn process_session(
    path: &Path,
    args: &Args,
    stats: &mut Stats,
) -> Result<Option<String>> {
    let f = File::open(path)?;
    let rdr = BufReader::new(f);

    // Subagent traces (under `<session>/subagents/<file>.jsonl`) have every
    // event flagged `isSidechain=true` because the subagent is a sidechain
    // relative to its parent session — but inside the subagent file those
    // events ARE the main conversation. Only filter sidechain events out
    // of top-level session files, where they're duplicates of the subagent
    // file's content.
    let is_subagent_file = path
        .components()
        .any(|c| c.as_os_str() == "subagents");

    // Walk the file line-by-line and collapse consecutive same-role events
    // into single semantic turns. Claude Code emits one JSON event per
    // streamed block, so a single assistant response with three tool_use
    // calls shows up as three separate assistant events — but they belong
    // to ONE assistant turn from the model's perspective.
    let mut turns: Vec<Turn> = Vec::new();
    for line in rdr.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let Ok(evt) = serde_json::from_str::<RawEvent>(&line) else {
            continue; // malformed single line — skip, don't abort the session
        };
        if evt.is_sidechain && !is_subagent_file {
            continue; // duplicate of what's in the subagents/ file
        }
        let role = match evt.event_type.as_str() {
            "user" => Role::User,
            "assistant" => Role::Assistant,
            _ => continue, // attachment, system, custom-title, last-prompt
        };
        let Some(msg) = evt.message else { continue };
        let blocks = extract_blocks(&msg, args.max_tool_result_chars);
        if blocks.is_empty() {
            continue; // empty content (e.g. pure thinking turns)
        }
        match turns.last_mut() {
            Some(last) if last.role == role => last.blocks.extend(blocks),
            _ => turns.push(Turn { role, blocks }),
        }
    }

    // ── filters ─────────────────────────────────────────────────────────
    // Need at least one user/assistant exchange; the tool_use check below
    // is what actually ensures the session is agentic. Subagent traces
    // often have only 1 user + 1 assistant turn (single focused tool
    // call) — valid agentic signal, don't drop them.
    let user_turns = turns.iter().filter(|t| t.role == Role::User).count();
    let asst_turns = turns.iter().filter(|t| t.role == Role::Assistant).count();
    if user_turns < 1 || asst_turns < 1 {
        stats.reject("too_short");
        return Ok(None);
    }
    let tool_uses: usize = turns
        .iter()
        .flat_map(|t| t.blocks.iter())
        .filter(|b| matches!(b, Block::ToolUse { .. }))
        .count();
    if tool_uses == 0 {
        stats.reject("no_tool_use");
        return Ok(None);
    }
    if args.drop_interrupted && is_interrupted(&turns) {
        stats.reject("interrupted");
        return Ok(None);
    }
    // Drop sessions whose FIRST turn isn't a user turn — R1 template
    // requires the user turn as the conversation opener, and without it
    // we'd render `<｜Assistant｜>` as the first non-system content,
    // which is out-of-distribution for the base model.
    if turns.first().map(|t| t.role) != Some(Role::User) {
        stats.reject("bad_opener");
        return Ok(None);
    }

    // ── render ──────────────────────────────────────────────────────────
    let rendered = render_deepseek_r1(&turns, tool_uses);
    if rendered.len() > args.max_chars {
        stats.reject("too_long");
        return Ok(None);
    }

    stats.turns_total += turns.len();
    stats.tool_uses_total += tool_uses;
    Ok(Some(rendered))
}

/// Extract our three block variants from a raw `message` Value. Handles
/// both shapes Claude Code emits:
///   - user text: `{"role":"user","content":"hello"}`
///   - user tool_result: `{"role":"user","content":[{"type":"tool_result",...}]}`
///   - assistant: `{"role":"assistant","content":[{"type":"text",...},
///                  {"type":"tool_use",...}]}`
fn extract_blocks(msg: &Value, max_tool_result_chars: usize) -> Vec<Block> {
    let mut out = Vec::new();
    let content = msg.get("content");
    match content {
        Some(Value::String(s)) => {
            if !s.trim().is_empty() {
                out.push(Block::Text(s.clone()));
            }
        }
        Some(Value::Array(items)) => {
            for item in items {
                let Some(obj) = item.as_object() else { continue };
                match obj.get("type").and_then(|v| v.as_str()) {
                    Some("text") => {
                        if let Some(t) = obj.get("text").and_then(|v| v.as_str()) {
                            if !t.trim().is_empty() {
                                out.push(Block::Text(t.to_string()));
                            }
                        }
                    }
                    Some("tool_use") => {
                        let name = obj
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let input = obj
                            .get("input")
                            .cloned()
                            .unwrap_or(Value::Object(Default::default()));
                        if !name.is_empty() {
                            out.push(Block::ToolUse { name, input });
                        }
                    }
                    Some("tool_result") => {
                        let id = obj
                            .get("tool_use_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let text = flatten_result_content(
                            obj.get("content"),
                            max_tool_result_chars,
                        );
                        out.push(Block::ToolResult {
                            tool_use_id: id,
                            content: text,
                        });
                    }
                    _ => {} // thinking / image / other — ignored
                }
            }
        }
        _ => {}
    }
    out
}

/// Tool_result content may itself be a string OR a list of content blocks
/// (each with `{"type":"text","text":...}`). Flatten both to a single
/// string, truncating to `max_chars` with an explicit `[truncated]` marker
/// so the model learns to handle finite context rather than expecting
/// unlimited tool output.
fn flatten_result_content(content: Option<&Value>, max_chars: usize) -> String {
    let raw = match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|it| {
                it.as_object()
                    .and_then(|o| o.get("text"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    };
    if raw.len() <= max_chars {
        raw
    } else {
        // Step BACK from max_chars until we hit a UTF-8 char boundary, so
        // we never slice mid-codepoint (panics on `&raw[..n]` otherwise).
        let mut cut = max_chars;
        while cut > 0 && !raw.is_char_boundary(cut) {
            cut -= 1;
        }
        let mut t = raw[..cut].to_string();
        t.push_str("\n[truncated]");
        t
    }
}

fn is_interrupted(turns: &[Turn]) -> bool {
    let Some(last) = turns.last() else { return false };
    last.blocks.iter().any(|b| match b {
        Block::Text(t) => {
            t.contains("[Request interrupted by user]")
                || t.contains("Request interrupted")
        }
        _ => false,
    })
}

// =============================================================================
// DeepSeek-R1 rendering
// =============================================================================

/// Static system prompt baked into every training sample. Mirrors the
/// intent of nexus-serve's `tool_use_system_preamble` so the trained
/// model's expectations about tool-calling format match what it'll
/// see at inference time on nexus-serve. Keep this in sync with
/// `nexus-serve/src/api/messages.rs::tool_use_system_preamble` if the
/// inference-side preamble changes format.
const SYSTEM_PROMPT: &str = "You are Oracle, a software engineering assistant that can call tools \
to inspect the filesystem, read code, and run commands. When you decide to call a tool, \
emit exactly one `<tool_use>{\"name\":\"<tool_name>\",\"input\":{...}}</tool_use>` block \
and nothing else in that turn. The runtime will execute the tool and return a tool_result \
in the next user turn; only then continue your response. Never invent tool results — if \
you need information, call a tool. Be concise.";

/// Render a session into DeepSeek-R1's chat-template wire form. Full-width
/// pipes (U+FF5C) are load-bearing — regular ASCII `|` tokenizes to
/// different IDs and produces gibberish on R1-Distill.
///
/// Shape:
/// ```text
/// <｜begin▁of▁sentence｜>{SYSTEM_PROMPT}
/// <｜User｜>{user1}<｜Assistant｜>{asst1}<｜end▁of▁sentence｜>
/// <｜User｜>{user2}<｜Assistant｜>{asst2}<｜end▁of▁sentence｜>
/// ...
/// ```
///
/// The final `<｜end▁of▁sentence｜>` IS present (unlike inference-time
/// generation-prompt mode) because during SFT we want the model to learn
/// to emit EOS at the end of every assistant turn.
fn render_deepseek_r1(turns: &[Turn], _tool_uses: usize) -> String {
    let mut out = String::with_capacity(4096);
    out.push_str("<｜begin▁of▁sentence｜>");
    out.push_str(SYSTEM_PROMPT);

    for turn in turns {
        match turn.role {
            Role::User => {
                out.push_str("<｜User｜>");
                render_user_blocks(&turn.blocks, &mut out);
            }
            Role::Assistant => {
                out.push_str("<｜Assistant｜>");
                render_assistant_blocks(&turn.blocks, &mut out);
                out.push_str("<｜end▁of▁sentence｜>");
            }
        }
    }
    out
}

/// User-turn content: free text plus flattened tool_result blocks. The
/// result is tagged so the model can distinguish user words from
/// tool-provided data, mirroring how Claude's API presents tool_result
/// to the next assistant turn.
fn render_user_blocks(blocks: &[Block], out: &mut String) {
    let mut first = true;
    for b in blocks {
        if !first {
            out.push_str("\n\n");
        }
        first = false;
        match b {
            Block::Text(t) => out.push_str(t),
            Block::ToolResult { tool_use_id, content } => {
                out.push_str("<tool_result id=\"");
                out.push_str(tool_use_id);
                out.push_str("\">\n");
                out.push_str(content);
                out.push_str("\n</tool_result>");
            }
            // Shouldn't happen — user turns don't have tool_use blocks —
            // but if they do, serialize defensively rather than lose data.
            Block::ToolUse { name, input } => {
                out.push_str("<tool_use>");
                out.push_str(&serde_json::json!({ "name": name, "input": input }).to_string());
                out.push_str("</tool_use>");
            }
        }
    }
}

/// Assistant-turn content: free text plus inline `<tool_use>{json}</tool_use>`
/// blocks. This exact surface form is what nexus-serve's Messages-API
/// parser looks for when converting streamed text → `stop_reason="tool_use"`
/// + `tool_use` content block, so SFT'ing on it gives zero-friction deploy.
fn render_assistant_blocks(blocks: &[Block], out: &mut String) {
    let mut first = true;
    for b in blocks {
        if !first {
            out.push_str("\n");
        }
        first = false;
        match b {
            Block::Text(t) => out.push_str(t),
            Block::ToolUse { name, input } => {
                out.push_str("<tool_use>");
                out.push_str(
                    &serde_json::json!({ "name": name, "input": input }).to_string(),
                );
                out.push_str("</tool_use>");
            }
            // tool_result doesn't belong in an assistant turn; skip.
            Block::ToolResult { .. } => {}
        }
    }
}

// =============================================================================
// Output
// =============================================================================

fn write_jsonl(path: &Path, samples: &[String]) -> Result<usize> {
    let f = File::create(path)
        .with_context(|| format!("creating {}", path.display()))?;
    let mut w = BufWriter::new(f);
    let mut total = 0usize;
    for s in samples {
        let obj = serde_json::json!({ "text": s });
        let line = obj.to_string();
        total += line.len();
        writeln!(w, "{line}")?;
    }
    w.flush()?;
    Ok(total)
}
