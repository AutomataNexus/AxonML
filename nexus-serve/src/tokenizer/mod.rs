//! tokenizer — Encode / Decode Dispatcher Across Three Backends
//!
//! Unified [`Tokenizer`] enum with three variants chosen at load time:
//! - `HuggingFace(tokenizers::Tokenizer)`: from a `tokenizer.json` file
//!   ([`Tokenizer::from_file`]) — first-choice backend, delegates to the
//!   upstream `tokenizers` crate.
//! - `GgufBpe { id_to_token, token_to_id, merges }`: built from the
//!   GGUF-embedded `tokenizer.ggml.tokens` / `.merges` metadata
//!   ([`Tokenizer::from_gguf`]). Used as the authoritative path for LLaMA-3
//!   / BitNet-2B / Qwen2 GGUFs that don't ship a sidecar `tokenizer.json`.
//! - `CharLevel { chars, char_to_id }`: last-resort fallback built from a
//!   printable ASCII corpus ([`Tokenizer::char_level`]). Produces garbage
//!   against a real model but keeps the server from failing to start.
//!
//! Priority when loading: `tokenizer.json` > GGUF-embedded > char-level.
//!
//! Encoding path:
//! - HuggingFace: `tok.encode(text, true)` — `add_special_tokens=true` so
//!   ChatML `<|im_start|>` and LLaMA-3 header ids resolve to their single
//!   token ID rather than being broken into byte fragments.
//! - GgufBpe: [`collect_special_tokens`] + [`split_on_specials`] pre-split
//!   exact `<|…|>` / `<s>` / `</s>` markers so the BPE merges see only
//!   non-special chunks. Each chunk goes through [`bpe_encode`] (byte-level
//!   GPT-2 remapping via [`byte_encode_map`] so `0x20 → Ġ` before merges
//!   run) or [`greedy_encode`] if merges are absent.
//! - CharLevel: direct char → id table lookup.
//!
//! Decoding path:
//! - HuggingFace: `tok.decode(ids, true)`.
//! - GgufBpe: reverses the byte-level permutation via [`byte_decode_map`],
//!   handles `<0xNN>` sentinels for raw bytes, strips known special
//!   markers, and SentencePiece `▁ → space`. Concatenates byte output and
//!   `String::from_utf8_lossy`s it.
//! - CharLevel: direct id → char.
//!
//! [`decode_bpe_token`] is a legacy per-token helper (referenced by tests);
//! production decoding uses the byte-bundle path above because it's the only
//! way to correctly stitch multi-byte UTF-8 glyphs that span multiple tokens.
//!
//! # File
//! `nexus-serve/src/tokenizer/mod.rs`
//!
//! # Author
//! Andrew Jewell Sr. — AutomataNexus LLC
//! ORCID: 0009-0005-2158-7060
//!
//! # Updated
//! April 16, 2026 11:15 PM EST
//!
//! # Disclaimer
//! Use at own risk. This software is provided "as is", without warranty of any
//! kind, express or implied. The author and AutomataNexus shall not be held
//! liable for any damages arising from the use of this software.

// =============================================================================
// Imports
// =============================================================================

use std::collections::HashMap;
use std::path::Path;

use crate::model::gguf::{GgufFile, GgufValue};

// =============================================================================
// Tokenizer Enum + Core Impl
// =============================================================================

/// A loaded tokenizer that can encode/decode text.
pub enum Tokenizer {
    /// HuggingFace tokenizers (BPE, WordPiece, etc.)
    HuggingFace(tokenizers::Tokenizer),
    /// BPE tokenizer built from GGUF-embedded vocabulary.
    GgufBpe {
        /// Token ID → token string
        id_to_token: Vec<String>,
        /// Token string → token ID
        token_to_id: HashMap<String, u32>,
        /// BPE merge pairs (if available)
        merges: Vec<(String, String)>,
    },
    /// Simple character-level fallback for AxonML-trained models.
    CharLevel {
        chars: Vec<char>,
        char_to_id: HashMap<char, u32>,
    },
}

impl Tokenizer {
    /// Load a HuggingFace tokenizer.json file.
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let tok = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        Ok(Self::HuggingFace(tok))
    }

    /// Build a tokenizer from GGUF-embedded vocabulary metadata.
    ///
    /// GGUF files contain:
    /// - `tokenizer.ggml.tokens`: Array of token strings (the vocabulary)
    /// - `tokenizer.ggml.scores`: Array of f32 merge priorities (optional)
    /// - `tokenizer.ggml.merges`: Array of merge rule strings "token_a token_b" (optional)
    /// - `tokenizer.ggml.token_type`: Array of i32 token types (optional)
    pub fn from_gguf(gguf: &GgufFile) -> Option<Self> {
        let tokens = gguf.get_meta("tokenizer.ggml.tokens")?;
        let token_strings: Vec<String> = match tokens {
            GgufValue::Array(arr) => arr
                .iter()
                .map(|v| match v {
                    GgufValue::String(s) => s.clone(),
                    _ => String::new(),
                })
                .collect(),
            _ => return None,
        };

        if token_strings.is_empty() {
            return None;
        }

        let mut token_to_id = HashMap::with_capacity(token_strings.len());
        for (i, tok) in token_strings.iter().enumerate() {
            token_to_id.insert(tok.clone(), i as u32);
        }

        // Load merges if available
        let merges = gguf
            .get_meta("tokenizer.ggml.merges")
            .and_then(|v| match v {
                GgufValue::Array(arr) => {
                    let pairs: Vec<(String, String)> = arr
                        .iter()
                        .filter_map(|v| match v {
                            GgufValue::String(s) => {
                                let parts: Vec<&str> = s.splitn(2, ' ').collect();
                                if parts.len() == 2 {
                                    Some((parts[0].to_string(), parts[1].to_string()))
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        })
                        .collect();
                    Some(pairs)
                }
                _ => None,
            })
            .unwrap_or_default();

        Some(Self::GgufBpe {
            id_to_token: token_strings,
            token_to_id,
            merges,
        })
    }

    /// Build a char-level tokenizer from a corpus string.
    pub fn char_level(corpus: &str) -> Self {
        use std::collections::BTreeSet;
        let mut chars: Vec<char> = corpus.chars().collect::<BTreeSet<_>>().into_iter().collect();
        if !chars.contains(&'\0') {
            chars.insert(0, '\0');
        }
        let mut char_to_id = HashMap::with_capacity(chars.len());
        for (i, &c) in chars.iter().enumerate() {
            char_to_id.insert(c, i as u32);
        }
        Self::CharLevel { chars, char_to_id }
    }

    /// Encode text to token IDs.
    ///
    /// Passes `add_special_tokens=true` so that ChatML markers like
    /// `<|im_start|>` and `<|im_end|>` get their proper single-token IDs
    /// when present in the tokenizer's added_tokens list.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        match self {
            Self::HuggingFace(tok) => tok
                .encode(text, true)
                .map(|enc| enc.get_ids().to_vec())
                .unwrap_or_default(),

            Self::GgufBpe {
                token_to_id,
                merges,
                id_to_token,
                ..
            } => {
                // Special tokens (`<|begin_of_text|>`, `<|eot_id|>`, etc.)
                // must be matched exactly BEFORE byte-level BPE runs. If
                // we let the BPE merges break them into character-level
                // tokens, the model never sees the actual special-token
                // ID in the prompt and tends to echo the pattern back as
                // raw bytes in its output. Split the input on special
                // tokens, BPE-encode each non-special chunk, then
                // interleave the special-token IDs.
                let specials = collect_special_tokens(token_to_id);
                let mut out: Vec<u32> = Vec::new();
                for segment in split_on_specials(text, &specials) {
                    match segment {
                        Segment::Special(id) => out.push(id),
                        Segment::Literal(s) => {
                            if s.is_empty() {
                                continue;
                            }
                            if !merges.is_empty() {
                                out.extend(bpe_encode(s, token_to_id, merges, id_to_token));
                            } else {
                                out.extend(greedy_encode(s, token_to_id));
                            }
                        }
                    }
                }
                out
            }

            Self::CharLevel { char_to_id, .. } => text
                .chars()
                .map(|c| *char_to_id.get(&c).unwrap_or(&0))
                .collect(),
        }
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        match self {
            Self::HuggingFace(tok) => tok.decode(ids, true).unwrap_or_default(),

            Self::GgufBpe { id_to_token, .. } => {
                // Byte-level BPE (GPT-2 / LLaMA-3 / BitNet-2B): each character
                // in the token string maps to one underlying byte via the
                // `bytes_to_unicode()` permutation. We concatenate the bytes
                // across all tokens and interpret the result as UTF-8. This
                // correctly reverses `Ġ → 0x20 (space)`, `Ċ → 0x0A (newline)`,
                // raw multi-byte UTF-8 glyphs, and the `<0xNN>` sentinel tokens
                // that some GGUF vocabs use for control bytes.
                let map = byte_decode_map();
                let mut bytes: Vec<u8> = Vec::with_capacity(ids.len() * 4);
                for &id in ids {
                    let Some(tok) = id_to_token.get(id as usize) else { continue };
                    // Skip special tokens that shouldn't appear in output.
                    // Includes DeepSeek R1's full-width-pipe variants
                    // (`<｜begin▁of▁sentence｜>` etc.) alongside the ASCII
                    // ChatML / LLaMA-3 specials.
                    if matches!(tok.as_str(),
                        "<s>" | "</s>" | "<|endoftext|>"
                        | "<|im_start|>" | "<|im_end|>"
                        | "<|begin_of_text|>" | "<|end_of_text|>"
                        | "<|start_header_id|>" | "<|end_header_id|>"
                        | "<|eot_id|>"
                        | "<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}>"
                        | "<\u{FF5C}end\u{2581}of\u{2581}sentence\u{FF5C}>"
                        | "<\u{FF5C}User\u{FF5C}>"
                        | "<\u{FF5C}Assistant\u{FF5C}>") {
                        continue;
                    }
                    // `<0xNN>` sentinel → one raw byte.
                    if tok.starts_with("<0x") && tok.ends_with('>') && tok.len() == 6 {
                        if let Ok(b) = u8::from_str_radix(&tok[3..5], 16) {
                            bytes.push(b);
                            continue;
                        }
                    }
                    // Otherwise map each character through byte_decode_map.
                    // Unknown characters fall through as-is (UTF-8 bytes of
                    // the char itself), which is the graceful path for
                    // SentencePiece-style vocabs that never took the byte-
                    // level remapping (we handle `▁` explicitly).
                    for c in tok.chars() {
                        if c == '\u{2581}' {
                            bytes.push(b' ');
                        } else if let Some(&b) = map.get(&c) {
                            bytes.push(b);
                        } else {
                            // Emit the char's own UTF-8.
                            let mut tmp = [0u8; 4];
                            bytes.extend_from_slice(c.encode_utf8(&mut tmp).as_bytes());
                        }
                    }
                }
                String::from_utf8_lossy(&bytes).into_owned()
            }

            Self::CharLevel { chars, .. } => ids
                .iter()
                .map(|&id| chars.get(id as usize).copied().unwrap_or('\0'))
                .collect(),
        }
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        match self {
            Self::HuggingFace(tok) => tok.get_vocab_size(true),
            Self::GgufBpe { id_to_token, .. } => id_to_token.len(),
            Self::CharLevel { chars, .. } => chars.len(),
        }
    }

    /// Get the variant name.
    pub fn variant(&self) -> &'static str {
        match self {
            Self::HuggingFace(_) => "HuggingFace BPE",
            Self::GgufBpe { merges, .. } => {
                if merges.is_empty() {
                    "GGUF vocab (greedy)"
                } else {
                    "GGUF BPE"
                }
            }
            Self::CharLevel { .. } => "char-level fallback",
        }
    }
}

// =============================================================================
// BPE encoding
// =============================================================================

/// BPE encode using merge rules from GGUF.
///
/// Applies the GPT-2 / LLaMA-3 byte-level permutation on the input UTF-8
/// bytes so every byte becomes exactly one unicode character from the
/// canonical set. This is what BitNet and LLaMA-3 vocabs expect:
/// `0x20` (space) must become `Ġ` (U+0120) **before** BPE merges run, or
/// no merge rule will ever match (`Ġhello` is in the vocab, `" hello"`
/// with literal space is not). Falling back to `<0xNN>` sentinels for
/// non-printable bytes is kept for SentencePiece-style vocabs that don't
/// use the byte-level map.
fn bpe_encode(
    text: &str,
    token_to_id: &HashMap<String, u32>,
    merges: &[(String, String)],
    _vocab: &[String],
) -> Vec<u32> {
    let forward = byte_encode_map();
    let bytes = text.as_bytes();
    let mut symbols: Vec<String> = bytes
        .iter()
        .map(|&b| {
            // Byte-level BPE: every byte has a unique unicode char.
            if let Some(&c) = forward.get(&b) {
                String::from(c)
            } else if b.is_ascii() && !b.is_ascii_control() {
                // Shouldn't happen (map covers 256), but keep the old
                // fallback for defensiveness.
                String::from(b as char)
            } else {
                format!("<0x{:02X}>", b)
            }
        })
        .collect();

    // Apply merges greedily
    for (left, right) in merges {
        let mut i = 0;
        while i + 1 < symbols.len() {
            if symbols[i] == *left && symbols[i + 1] == *right {
                let merged = format!("{}{}", left, right);
                symbols[i] = merged;
                symbols.remove(i + 1);
                // Don't increment i — check if the merged token can merge again
            } else {
                i += 1;
            }
        }
    }

    // Map to token IDs
    symbols
        .iter()
        .map(|s| *token_to_id.get(s).unwrap_or(&0))
        .collect()
}

/// Greedy longest-match encoding (when no merge rules available).
fn greedy_encode(text: &str, token_to_id: &HashMap<String, u32>) -> Vec<u32> {
    let mut ids = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let mut best_len = 0;
        let mut best_id = 0u32;

        // Try progressively shorter substrings
        let max_len = (chars.len() - i).min(32); // cap at 32 chars
        for len in (1..=max_len).rev() {
            let substr: String = chars[i..i + len].iter().collect();
            if let Some(&id) = token_to_id.get(&substr) {
                best_len = len;
                best_id = id;
                break;
            }
        }

        if best_len == 0 {
            // Single char not in vocab — try byte fallback
            let c = chars[i];
            for b in c.to_string().as_bytes() {
                let byte_tok = format!("<0x{:02X}>", b);
                if let Some(&id) = token_to_id.get(&byte_tok) {
                    ids.push(id);
                } else {
                    ids.push(0); // unknown
                }
            }
            i += 1;
        } else {
            ids.push(best_id);
            i += best_len;
        }
    }

    ids
}

/// Decode a BPE token string, handling byte-level encoding.
/// GPT-2 / LLaMA-3 / BitNet byte-level BPE char → byte mapping, built once
/// lazily and cached for the life of the process.
///
/// Matches OpenAI's `bytes_to_unicode` (GPT-2 paper): printable ASCII and
/// Latin-1 bytes map to themselves (`0x21..=0x7E`, `0xA1..=0xAC`,
/// `0xAE..=0xFF`); every other byte is remapped to a character in the
/// `U+0100..` range in the order they're encountered. This is the
/// remapping that produces `Ġ` for space (`0x20` → `U+0120`), `Ċ` for
/// newline (`0x0A` → `U+010A`), etc.
// =============================================================================
// Special-token scanning (exact-match before BPE)
// =============================================================================

enum Segment<'a> {
    Literal(&'a str),
    Special(u32),
}

/// Gather the (text, id) list of tokens that look like special markers —
/// anything starting with `<|` or surrounded by `<>` delimiters. Sorted by
/// descending length so longest match wins during scanning.
///
/// Matching HuggingFace's "added_tokens" behavior without needing the
/// explicit list: if the GGUF vocab has `<|eot_id|>` as a single token
/// string, we treat it as a literal to match before BPE.
fn collect_special_tokens(token_to_id: &HashMap<String, u32>) -> Vec<(String, u32)> {
    let mut specials: Vec<(String, u32)> = token_to_id
        .iter()
        .filter(|(tok, _)| {
            let s = tok.as_str();
            // `<|...|>` covers ChatML / LLaMA-3 / Qwen-Instruct specials.
            // `<｜...｜>` (full-width pipe U+FF5C) covers DeepSeek R1 /
            // R1-Distill specials: `<｜begin▁of▁sentence｜>`,
            // `<｜User｜>`, `<｜Assistant｜>`, `<｜end▁of▁sentence｜>`.
            // Without the full-width variant the BPE encoder falls back
            // to byte-level encoding for those tags and the model, never
            // having seen the resulting byte sequence in SFT, emits
            // degenerate output ("< < < ...").
            (s.starts_with("<|") && s.ends_with("|>"))
                || (s.starts_with("<\u{FF5C}") && s.ends_with("\u{FF5C}>"))
                || s == "<s>"
                || s == "</s>"
        })
        .map(|(t, &i)| (t.clone(), i))
        .collect();
    specials.sort_by_key(|(t, _)| std::cmp::Reverse(t.len()));
    specials
}

/// Scan `text` left-to-right, emitting `Literal` spans interleaved with
/// `Special(id)` for each special-token exact match.
fn split_on_specials<'a>(text: &'a str, specials: &[(String, u32)]) -> Vec<Segment<'a>> {
    if specials.is_empty() {
        return vec![Segment::Literal(text)];
    }
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut lit_start = 0usize;
    while i < bytes.len() {
        let mut matched: Option<(usize, u32)> = None;
        for (tok, id) in specials {
            let tb = tok.as_bytes();
            if i + tb.len() <= bytes.len() && &bytes[i..i + tb.len()] == tb {
                matched = Some((tb.len(), *id));
                break;
            }
        }
        if let Some((len, id)) = matched {
            if i > lit_start {
                out.push(Segment::Literal(&text[lit_start..i]));
            }
            out.push(Segment::Special(id));
            i += len;
            lit_start = i;
        } else {
            i += 1;
        }
    }
    if lit_start < bytes.len() {
        out.push(Segment::Literal(&text[lit_start..]));
    }
    out
}

// =============================================================================
// Byte-Level Maps + Per-Token Decode
// =============================================================================

/// Inverse of [`byte_decode_map`] — byte → char forward map used by BPE
/// encode so space (`0x20`) becomes `Ġ` (U+0120) before merges run.
fn byte_encode_map() -> &'static HashMap<u8, char> {
    use std::sync::OnceLock;
    static MAP: OnceLock<HashMap<u8, char>> = OnceLock::new();
    MAP.get_or_init(|| {
        byte_decode_map()
            .iter()
            .map(|(&c, &b)| (b, c))
            .collect()
    })
}

fn byte_decode_map() -> &'static HashMap<char, u8> {
    use std::sync::OnceLock;
    static MAP: OnceLock<HashMap<char, u8>> = OnceLock::new();
    MAP.get_or_init(|| {
        // Canonical printable set: 0x21..=0x7E, 0xA1..=0xAC, 0xAE..=0xFF.
        let mut canonical: Vec<u8> = Vec::with_capacity(256);
        for b in 0x21u8..=0x7E { canonical.push(b); }
        for b in 0xA1u8..=0xAC { canonical.push(b); }
        for b in 0xAEu8..=0xFF { canonical.push(b); }

        // Forward map: byte → char. Canonical bytes map to themselves; other
        // bytes get code points 0x100, 0x101, ... in ascending order.
        let mut forward: Vec<(u8, char)> = canonical
            .iter()
            .map(|&b| (b, b as char))
            .collect();
        let mut next_cp: u32 = 0x100;
        for b in 0u8..=255 {
            if !canonical.contains(&b) {
                let c = char::from_u32(next_cp).unwrap();
                forward.push((b, c));
                next_cp += 1;
            }
        }

        // Reverse it: char → byte.
        forward.into_iter().map(|(b, c)| (c, b)).collect()
    })
}

fn decode_bpe_token(token: &str) -> String {
    // Handle byte tokens: <0xNN>
    if token.starts_with("<0x") && token.ends_with('>') && token.len() == 6 {
        if let Ok(byte) = u8::from_str_radix(&token[3..5], 16) {
            return String::from(byte as char);
        }
    }

    // Handle common BPE special tokens
    if token == "<s>" || token == "</s>" || token == "<|endoftext|>" || token == "<|im_end|>" {
        return String::new();
    }
    if token == "<|im_start|>" {
        return String::new();
    }

    // SentencePiece-style: leading ▁ (U+2581) = space
    token.replace('▁', " ")
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_level_map_covers_256_bytes_bijectively() {
        let map = byte_decode_map();
        assert_eq!(map.len(), 256, "byte decode map must cover 256 bytes");
        let mut seen = vec![false; 256];
        for &b in map.values() {
            assert!(!seen[b as usize], "byte 0x{b:02X} appears twice");
            seen[b as usize] = true;
        }
        // Key landmarks from GPT-2 / LLaMA-3 byte-level BPE:
        assert_eq!(map.get(&'Ġ'), Some(&0x20), "Ġ → space (0x20)");
        assert_eq!(map.get(&'Ċ'), Some(&0x0A), "Ċ → newline (0x0A)");
        assert_eq!(map.get(&'ĉ'), Some(&0x09), "ĉ → tab (0x09)");
        // Printable ASCII is identity.
        assert_eq!(map.get(&'A'), Some(&0x41));
        assert_eq!(map.get(&'z'), Some(&0x7A));
        // Latin-1 printable passes through.
        assert_eq!(map.get(&'¡'), Some(&0xA1));
        assert_eq!(map.get(&'ÿ'), Some(&0xFF));
    }

    #[test]
    fn gguf_bpe_decode_reverses_byte_level_tokens() {
        // Build a minimal GGUF-style BPE tokenizer with a handful of tokens.
        let tokens = vec![
            "<|endoftext|>".to_string(), // id 0
            "Ġhello".to_string(),         // id 1 → " hello"
            "Ġworld".to_string(),         // id 2 → " world"
            "!".to_string(),               // id 3
            "Ċ".to_string(),               // id 4 → "\n"
        ];
        let mut token_to_id = HashMap::new();
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }
        let tok = Tokenizer::GgufBpe {
            id_to_token: tokens,
            token_to_id,
            merges: vec![],
        };
        let decoded = tok.decode(&[1, 2, 3, 4]);
        assert_eq!(decoded, " hello world!\n");
    }

    #[test]
    fn test_char_level_roundtrip() {
        let tok = Tokenizer::char_level("hello world");
        let ids = tok.encode("hello");
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_greedy_encode() {
        let mut vocab = HashMap::new();
        vocab.insert("he".to_string(), 1);
        vocab.insert("llo".to_string(), 2);
        vocab.insert(" ".to_string(), 3);
        vocab.insert("world".to_string(), 4);

        let ids = greedy_encode("hello world", &vocab);
        assert_eq!(ids, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_decode_bpe_token() {
        assert_eq!(decode_bpe_token("hello"), "hello");
        assert_eq!(decode_bpe_token("▁hello"), " hello");
        assert_eq!(decode_bpe_token("<0x0A>"), "\n");
        assert_eq!(decode_bpe_token("<s>"), "");
    }
}
