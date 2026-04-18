//! spec_bench — Speculative decoding benchmark
//!
//! Loads a target model and a (smaller, same-tokenizer) draft model, runs a
//! speculative-decoding loop against a prompt, and reports tokens/sec plus
//! draft acceptance rate.
//!
//! Usage:
//!   cargo run --release --features cuda --bin spec_bench -- \
//!     --target /path/to/target.gguf \
//!     --draft  /path/to/draft.gguf \
//!     --prompt "Write a paragraph about the history of computing." \
//!     --max-tokens 128 \
//!     --gamma 4
//!
//! Algorithm (greedy / temperature=0, the simplest correct form):
//!
//!   1. Prefill both models with the prompt. Commit `t_last` = target's
//!      greedy first-token prediction from prefill logits.
//!   2. Speculative round:
//!        a. Draft runs `forward_one(t_last)` → logit → d_1 (greedy).
//!           Then `forward_one(d_1)` → d_2, ..., up to d_{gamma}.
//!           Draft's KV cache grew by gamma rows.
//!        b. Target runs `forward_batch([t_last, d_1, ..., d_{gamma-1}])`
//!           in a single call. Output: gamma logit rows, one per input
//!           position. Target's KV cache grew by gamma rows.
//!        c. For i in 0..gamma-1: if argmax(target_logits[i]) == d_{i+1},
//!           accept d_{i+1}. On first mismatch, substitute
//!           argmax(target_logits[i]) and stop accepting further drafts.
//!           If all gamma-1 drafts accepted, emit the "bonus token" =
//!           argmax(target_logits[gamma-1]).
//!        d. Let `accepted` = number of draft tokens confirmed;
//!           `emitted` = accepted + 1 (substitution or bonus).
//!           Commit those `emitted` tokens. Set `t_last` to the last one.
//!        e. Roll back any unused draft / target KV rows to the committed
//!           position so the next round starts from a clean state.
//!
//! Acceptance rate reported as `accepted_tokens / (speculative_rounds *
//! (gamma-1))`.
//!
//! # File
//! `nexus-serve/src/bin/spec_bench.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060

use std::path::{Path, PathBuf};
use std::time::Instant;

use nexus_serve::model::gguf::GgufFile;
use nexus_serve::model::inference::{argmax, InferenceEngine, KvCache, MappedGguf};
use nexus_serve::tokenizer::Tokenizer;

#[cfg(feature = "cuda")]
use axonml_core::Device;

// =============================================================================
// CLI
// =============================================================================

struct Args {
    target: PathBuf,
    draft: PathBuf,
    prompt: String,
    max_tokens: usize,
    gamma: usize,
    /// Run target-only baseline for comparison, no spec.
    baseline_only: bool,
    /// Suppress generated text printing.
    quiet: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        target: PathBuf::new(),
        draft: PathBuf::new(),
        prompt: "Write a paragraph about the history of computing.".to_string(),
        max_tokens: 128,
        gamma: 4,
        baseline_only: false,
        quiet: false,
    };
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--target" => { i += 1; a.target = PathBuf::from(&argv[i]); }
            "--draft"  => { i += 1; a.draft  = PathBuf::from(&argv[i]); }
            "--prompt" => { i += 1; a.prompt = argv[i].clone(); }
            "--max-tokens" | "-n" => { i += 1; a.max_tokens = argv[i].parse().unwrap(); }
            "--gamma" | "-g" => { i += 1; a.gamma = argv[i].parse().unwrap(); }
            "--baseline-only" => { a.baseline_only = true; }
            "--quiet" | "-q" => { a.quiet = true; }
            "--help" | "-h" => {
                println!("spec_bench --target PATH --draft PATH [--prompt STR] [-n N] [-g G] [--baseline-only] [-q]");
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }
    if a.target.as_os_str().is_empty() || a.draft.as_os_str().is_empty() {
        eprintln!("--target and --draft are required");
        std::process::exit(1);
    }
    if a.gamma < 2 { a.gamma = 2; }
    a
}

// =============================================================================
// Load a model + tokenizer
// =============================================================================

struct Loaded {
    engine: InferenceEngine,
    tok: Tokenizer,
    // Held so the memory-map stays live for the engine's weight pointers.
    _mapped: MappedGguf,
    _gguf: GgufFile,
}

fn load_model(path: &Path, label: &str) -> Loaded {
    println!("[{label}] loading {}", path.display());
    let gguf = GgufFile::open(path).expect("gguf open");
    let mapped = MappedGguf::open(path, &gguf).expect("mmap");
    let mut engine =
        InferenceEngine::from_gguf_with_mode(&gguf, &mapped, /*quantized=*/ true)
            .expect("engine");
    #[cfg(feature = "cuda")]
    {
        if axonml_core::backends::cuda::is_available() {
            engine.to_device(Device::Cuda(0));
            println!("[{label}]   device: Cuda(0)");
        }
    }
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    println!(
        "[{label}]   arch={} hidden={} layers={} heads={} kv_heads={} vocab={}",
        engine.architecture(),
        engine.config.hidden_size,
        engine.config.num_layers,
        engine.config.num_heads,
        engine.config.num_kv_heads,
        engine.config.vocab_size,
    );
    Loaded { engine, tok, _mapped: mapped, _gguf: gguf }
}

// =============================================================================
// Baselines and speculative loop
// =============================================================================

fn logits_tail<'a>(logits: &'a [f32], vocab: usize, i_from_end: usize) -> &'a [f32] {
    // forward_batch returns [seq_len, vocab] flat; row i starts at i*vocab.
    // `i_from_end` counts from the END of the sequence (0 = last row).
    let total_rows = logits.len() / vocab;
    let row = total_rows - 1 - i_from_end;
    &logits[row * vocab..(row + 1) * vocab]
}

/// `use_batch_for_decode=true` loops `forward_batch([tok])` instead of
/// `forward_one(tok)` so every token is produced on the SAME numerical
/// path that spec's verify uses (CPU activations, GPU matmul). With this
/// flag off the baseline uses the faster `forward_one_gpu_resident` path
/// but its logits will NOT be numerically identical to spec's verify —
/// which tanks spec's acceptance rate. Pass `true` when the question is
/// "does spec decoding help when numerics match" and `false` to compare
/// against the production decode speed.
fn baseline_decode(
    loaded: &Loaded,
    prompt_ids: &[u32],
    max_tokens: usize,
    use_batch_for_decode: bool,
) -> (Vec<u32>, std::time::Duration) {
    let mut kv = KvCache::new(loaded.engine.config.num_layers);
    let prefill_logits = loaded.engine.forward_batch(prompt_ids, &mut kv);
    let vocab = loaded.engine.config.vocab_size;
    let mut t = argmax(logits_tail(&prefill_logits, vocab, 0)) as u32;
    let stop: Vec<u32> = loaded.engine.stop_tokens().to_vec();
    let mut out = Vec::with_capacity(max_tokens);
    if stop.contains(&t) {
        return (out, std::time::Duration::from_nanos(0));
    }
    out.push(t);
    let start = Instant::now();
    for _ in 1..max_tokens {
        let logits = if use_batch_for_decode {
            loaded.engine.forward_batch(&[t], &mut kv)
        } else {
            loaded.engine.forward_one(t, &mut kv)
        };
        // forward_batch returns [seq_len, vocab] flat; take last row.
        let row = &logits[logits.len() - vocab..];
        t = argmax(row) as u32;
        if stop.contains(&t) { break; }
        out.push(t);
    }
    (out, start.elapsed())
}

/// Speculative decoding with gamma-wide draft window, greedy sampling.
///
/// Returns (generated_tokens, decode_elapsed, accepted_drafts, draft_rounds).
/// Time excludes prefill (like the baseline).
fn speculative_decode(
    target: &Loaded,
    draft: &Loaded,
    prompt_ids: &[u32],
    max_tokens: usize,
    gamma: usize,
) -> (Vec<u32>, std::time::Duration, usize, usize) {
    let vocab = target.engine.config.vocab_size;
    let draft_vocab = draft.engine.config.vocab_size;
    // Target's vocab may be wider than draft's (e.g. DeepSeek-R1 adds 128
    // chat tokens on top of Qwen2.5). The overlap (first min(vt, vd)
    // tokens) must match between the two tokenizers for speculative
    // decoding to be meaningful. When `t_last` lands in target-only
    // territory, we skip drafting for that step and let target decode
    // solo — keeps correctness intact at the cost of one draft round.
    let min_vocab = vocab.min(draft_vocab);
    println!(
        "vocab: target={} draft={} overlap={}",
        vocab, draft_vocab, min_vocab
    );

    let mut t_kv = KvCache::new(target.engine.config.num_layers);
    let mut d_kv = KvCache::new(draft.engine.config.num_layers);

    // Prefill both on the prompt (out of the measured region).
    let prefill_logits = target.engine.forward_batch(prompt_ids, &mut t_kv);
    let _ = draft.engine.forward_batch(prompt_ids, &mut d_kv);
    let mut t_last = argmax(logits_tail(&prefill_logits, vocab, 0)) as u32;

    let stop_t: Vec<u32> = target.engine.stop_tokens().to_vec();
    let mut out = Vec::with_capacity(max_tokens);
    if stop_t.contains(&t_last) {
        return (out, std::time::Duration::from_nanos(0), 0, 0);
    }
    out.push(t_last);

    let mut accepted_total = 0usize;
    let mut rounds = 0usize;

    let start = Instant::now();
    while out.len() < max_tokens {
        // If the prior round committed a token that's in target-only
        // vocab (draft has no embedding for it), skip this spec round
        // and do one solo target forward_one. Then retry the loop.
        if (t_last as usize) >= draft_vocab {
            let logits = target.engine.forward_one(t_last, &mut t_kv);
            // Advance draft's cache too with the closest in-vocab token
            // so the two caches stay in positional sync. Using 0 (pad)
            // is safe here — drafts made in rounds AFTER this step will
            // be re-computed from t_last anyway. We just need the row
            // count aligned.
            let _ = draft.engine.forward_one(0, &mut d_kv);
            let t = argmax(&logits) as u32;
            if stop_t.contains(&t) { break; }
            out.push(t);
            t_last = t;
            continue;
        }

        rounds += 1;

        // ---- Draft: γ speculative tokens (autoregressive on draft KV) ----
        let mut drafts: Vec<u32> = Vec::with_capacity(gamma);
        let mut drafts_input = t_last;
        for _ in 0..gamma {
            let dlogits = draft.engine.forward_one(drafts_input, &mut d_kv);
            let next = argmax(&dlogits) as u32;
            drafts.push(next);
            drafts_input = next;
        }

        // ---- Target verify: [t_last, d_0, d_1, ..., d_{γ-2}] in one batch ----
        // Feeding γ tokens produces γ logit rows; row i predicts what comes
        // AFTER position (t_kv.len before batch) + i, which is compared to
        // drafts[i] for i in 0..γ-1, with drafts[γ-1]'s target prediction
        // serving as the bonus token.
        let pre_t_len = t_kv.len;
        let mut batch: Vec<u32> = Vec::with_capacity(gamma);
        batch.push(t_last);
        for d in drafts.iter().take(gamma - 1) { batch.push(*d); }
        let t_logits = target.engine.forward_batch(&batch, &mut t_kv);

        // ---- Accept / reject ----
        let mut accepted_this_round = 0usize;
        let mut substituted: Option<u32> = None;
        for i in 0..gamma {
            let row = &t_logits[i * vocab..(i + 1) * vocab];
            let t_argmax = argmax(row) as u32;
            if i < gamma - 1 {
                if t_argmax == drafts[i] {
                    accepted_this_round += 1;
                } else {
                    substituted = Some(t_argmax);
                    break;
                }
            } else {
                // All γ-1 drafts accepted → bonus token from last logit row.
                substituted = Some(t_argmax);
            }
        }
        accepted_total += accepted_this_round;

        // Commit accepted + 1 substitution/bonus:
        //   out += drafts[0..accepted_this_round] then push substituted
        let mut committed = 0usize;
        for d in drafts.iter().take(accepted_this_round) {
            out.push(*d);
            committed += 1;
            if out.len() >= max_tokens { break; }
        }
        if out.len() < max_tokens {
            if let Some(sub) = substituted {
                out.push(sub);
                committed += 1;
                t_last = sub;
            } else {
                // Shouldn't happen — substituted always set above.
                break;
            }
        } else if accepted_this_round > 0 {
            t_last = out[out.len() - 1];
        }

        // Stop-token check on just-emitted tokens; drop them from `out`
        // and break if we hit one.
        let start_check = out.len().saturating_sub(committed);
        for &t in &out[start_check..] {
            if stop_t.contains(&t) {
                out.truncate(out.len() - (out.len() - start_check - out[start_check..].iter().position(|&x| x == t).unwrap()));
                return (out, start.elapsed(), accepted_total, rounds);
            }
        }

        // ---- Roll back KV caches to match the committed position ----
        // Layout: before this round, kv.len == pre_t_len, with K/V for
        // positions [0..pre_t_len-1]. Target verify added γ rows with
        // INPUTS [t_last, d_0, ..., d_{γ-2}] at positions [pre_t_len..
        // pre_t_len+γ-1]. Rows where the input matched what we commit
        // (t_last at pre_t_len, then accepted drafts) hold correct K/V.
        // Rows beyond the first accepted-draft-count+1 positions are
        // stale (input was a rejected draft). Truncate to keep the
        // correct prefix: pre_t_len + (accepted drafts) + 1 (for t_last).
        // Since `committed` = accepted + 1 (accepted drafts + substitution),
        // the correct length is pre_t_len + committed. The substituted
        // token itself has NOT been processed yet (it will be the first
        // input on the next round), so we stop right before it.
        let new_len = pre_t_len + committed;
        // Target's K/V for positions [pre_t_len..new_len-1] is correct.
        // Note: target's row at (new_len-1) corresponds to the last
        // accepted draft; the substitution is emitted but not yet kv'd.
        t_kv.truncate(new_len);
        // Draft's cache grew by γ identically — keep the same prefix.
        // Draft's row at position new_len-1 is correct if and only if
        // the committed token at that position matches what draft fed
        // (it does, because it's either t_last or an accepted draft).
        d_kv.truncate(new_len);

        if committed == 0 { break; }
    }

    (out, start.elapsed(), accepted_total, rounds)
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    let args = parse_args();
    println!("=== spec_bench ===");
    let target = load_model(&args.target, "target");
    let draft = if args.baseline_only {
        // Not used but simpler to still load so comparison prints align.
        load_model(&args.draft, "draft")
    } else {
        load_model(&args.draft, "draft")
    };

    // Prompt tokenization (same vocab, so either tokenizer works).
    let prompt_ids = target.tok.encode(&args.prompt);
    println!("\nprompt: {:?}", args.prompt);
    println!("prompt_ids: {} tokens", prompt_ids.len());

    // ---- Baseline A: forward_one path (production decode speed). ----
    println!("\n--- baseline A: target solo (forward_one, production path) ---");
    let (ba_tokens, ba_elapsed) =
        baseline_decode(&target, &prompt_ids, args.max_tokens, false);
    let ba_tps = ba_tokens.len() as f64 / ba_elapsed.as_secs_f64();
    println!(
        "baseline A: {} tokens in {:.3}s = {:.2} tok/s",
        ba_tokens.len(), ba_elapsed.as_secs_f64(), ba_tps
    );

    // ---- Baseline B: forward_batch path (numerically matches spec's verify). ----
    println!("\n--- baseline B: target solo (forward_batch loop — spec-verify-equivalent) ---");
    let (b_tokens, b_elapsed) =
        baseline_decode(&target, &prompt_ids, args.max_tokens, true);
    let b_tps = b_tokens.len() as f64 / b_elapsed.as_secs_f64();
    println!(
        "baseline B: {} tokens in {:.3}s = {:.2} tok/s",
        b_tokens.len(), b_elapsed.as_secs_f64(), b_tps
    );
    if !args.quiet {
        let text = target.tok.decode(&b_tokens);
        let snippet: String = text.chars().take(200).collect();
        println!("baseline text: {:?}", snippet);
    }

    if args.baseline_only {
        return;
    }
    let _ = ba_tokens;  // silence unused if baseline-only path returns

    // ---- Speculative ----
    println!("\n--- speculative (γ={}) ---", args.gamma);
    let (s_tokens, s_elapsed, accepted, rounds) =
        speculative_decode(&target, &draft, &prompt_ids, args.max_tokens, args.gamma);
    let s_tps = s_tokens.len() as f64 / s_elapsed.as_secs_f64();
    let accept_rate = if rounds > 0 {
        accepted as f64 / (rounds as f64 * (args.gamma - 1) as f64)
    } else { 0.0 };
    println!(
        "spec: {} tokens in {:.3}s = {:.2} tok/s  (rounds={}, accepted_drafts={}/{} = {:.1}%)",
        s_tokens.len(), s_elapsed.as_secs_f64(), s_tps,
        rounds, accepted, rounds * (args.gamma - 1), accept_rate * 100.0,
    );
    if !args.quiet {
        let text = target.tok.decode(&s_tokens);
        let snippet: String = text.chars().take(200).collect();
        println!("spec text: {:?}", snippet);
    }

    // ---- Comparison ----
    println!("\n--- summary ---");
    println!("baseline A (forward_one, production):       {:.2} tok/s", ba_tps);
    println!("baseline B (forward_batch, verify-match):   {:.2} tok/s", b_tps);
    println!(
        "spec(γ={}):                                   {:.2} tok/s  ({:.2}× vs A, {:.2}× vs B)",
        args.gamma, s_tps, s_tps / ba_tps, s_tps / b_tps
    );
    println!("acceptance rate: {:.1}% of draft slots", accept_rate * 100.0);

    // Sanity-check: do the generated token sequences match on the first N
    // tokens they share? (They may diverge if spec's verify cache diverges
    // from solo, but for greedy decoding on the same model they should
    // match up to a prefix. A long mismatched prefix signals a bug.)
    let common = b_tokens.len().min(s_tokens.len());
    let mut first_divergence = common;
    for i in 0..common {
        if b_tokens[i] != s_tokens[i] {
            first_divergence = i;
            break;
        }
    }
    println!(
        "first divergence between baseline and spec outputs: token {} of {} common",
        first_divergence, common,
    );
}
