//! `gguf_inspect` — dump GGUF metadata + tensor layout using nexus-serve's own parser.
//!
//! Used to verify a new GGUF variant (dtype, architecture, layer names) before
//! wiring the architecture dispatch. Usage:
//!
//!   cargo run --release --bin gguf_inspect -- /path/to/model.gguf

use std::collections::BTreeMap;
use std::path::PathBuf;

use nexus_serve::model::gguf::{GgmlType, GgufFile, GgufValue};

fn fmt_value(v: &GgufValue) -> String {
    match v {
        GgufValue::U8(n) => n.to_string(),
        GgufValue::I8(n) => n.to_string(),
        GgufValue::U16(n) => n.to_string(),
        GgufValue::I16(n) => n.to_string(),
        GgufValue::U32(n) => n.to_string(),
        GgufValue::I32(n) => n.to_string(),
        GgufValue::U64(n) => n.to_string(),
        GgufValue::I64(n) => n.to_string(),
        GgufValue::F32(x) => format!("{x:.6}"),
        GgufValue::F64(x) => format!("{x:.6}"),
        GgufValue::Bool(b) => b.to_string(),
        GgufValue::String(s) => {
            if s.len() <= 80 {
                format!("{s:?}")
            } else {
                format!("{:?}… ({} chars total)", &s[..80], s.len())
            }
        }
        GgufValue::Array(a) => format!("[{} items]", a.len()),
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: gguf_inspect <path-to-gguf>");
    let path = PathBuf::from(path);

    let file = GgufFile::open(&path).expect("failed to open GGUF");

    println!("=== GGUF header ===");
    println!("path:       {}", path.display());
    println!("version:    {}", file.version);
    println!("tensors:    {}", file.n_tensors);
    println!("data_off:   0x{:x}", file.data_offset);
    println!();

    println!("=== metadata ({} keys) ===", file.metadata.len());
    let sorted: BTreeMap<_, _> = file.metadata.iter().collect();
    for (k, v) in &sorted {
        println!("  {k} = {}", fmt_value(v));
    }
    println!();

    // Summarize tensor dtypes
    let mut dtype_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut dtype_bytes: BTreeMap<String, u64> = BTreeMap::new();
    for t in &file.tensors {
        let key = format!("{:?}", t.dtype);
        *dtype_counts.entry(key.clone()).or_insert(0) += 1;
        *dtype_bytes.entry(key).or_insert(0) += t.total_bytes();
    }
    println!("=== dtype summary ===");
    for (k, n) in &dtype_counts {
        let mb = dtype_bytes[k] as f64 / 1e6;
        println!("  {k:<10} × {n:>4}    {mb:>10.1} MB");
    }
    println!();

    // Show first ~30 tensors so we can pattern-match on layer names
    println!("=== first 30 tensors ===");
    for t in file.tensors.iter().take(30) {
        let dims: Vec<String> = t.dims.iter().map(|d| d.to_string()).collect();
        println!(
            "  {:<52} {:<6} [{}]  {} MB",
            t.name,
            format!("{:?}", t.dtype),
            dims.join(" × "),
            t.total_bytes() / (1024 * 1024),
        );
    }
    if file.tensors.len() > 30 {
        println!("  … ({} more tensors)", file.tensors.len() - 30);
    }

    // Flag any Unknown dtype — those are what we need to add support for.
    let unknowns: Vec<&_> = file
        .tensors
        .iter()
        .filter(|t| t.dtype == GgmlType::Unknown)
        .collect();
    if !unknowns.is_empty() {
        println!();
        println!("!!! {} tensors use an unknown dtype (not registered in GgmlType::from_u32)", unknowns.len());
    }
}
