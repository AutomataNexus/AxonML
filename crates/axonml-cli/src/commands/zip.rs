//! Zip - Bundle Models and Datasets
//!
//! Creates compressed archives of models and datasets for sharing,
//! backup, or deployment.
//!
//! @version 0.1.0
//! @author `AutomataNexus` Development Team

use std::fs;
use std::path::PathBuf;

use walkdir::WalkDir;

use super::utils::{ensure_dir, path_exists, print_header, print_info, print_kv, print_success};
use crate::cli::{ZipArgs, ZipCreateArgs, ZipExtractArgs, ZipListArgs, ZipSubcommand};
use crate::error::{CliError, CliResult};

// =============================================================================
// Execute Command
// =============================================================================

/// Execute the `zip` command
pub fn execute(args: ZipArgs) -> CliResult<()> {
    match args.action {
        ZipSubcommand::Create(create_args) => execute_create(create_args),
        ZipSubcommand::Extract(extract_args) => execute_extract(extract_args),
        ZipSubcommand::List(list_args) => execute_list(list_args),
    }
}

// =============================================================================
// Create Subcommand
// =============================================================================

fn execute_create(args: ZipCreateArgs) -> CliResult<()> {
    print_header("Create Bundle");

    // Validate inputs
    let mut source_paths: Vec<PathBuf> = Vec::new();

    if let Some(model) = &args.model {
        let path = PathBuf::from(model);
        if !path_exists(&path) {
            return Err(CliError::Model(format!("Model not found: {model}")));
        }
        source_paths.push(path);
        print_kv("Model", model);
    }

    if let Some(data) = &args.data {
        let path = PathBuf::from(data);
        if !path_exists(&path) {
            return Err(CliError::Data(format!("Dataset not found: {data}")));
        }
        source_paths.push(path);
        print_kv("Dataset", data);
    }

    if source_paths.is_empty() {
        return Err(CliError::InvalidArgument(
            "Must specify at least --model or --data".to_string(),
        ));
    }

    // Determine output path
    let output_path = PathBuf::from(&args.output);
    print_kv("Output", &args.output);

    // Check if output exists
    if output_path.exists() && !args.overwrite {
        return Err(CliError::Other(format!(
            "Output file exists: {}. Use --overwrite to replace.",
            args.output
        )));
    }

    println!();
    print_info("Creating bundle...");

    // Create the bundle (tar format for simplicity)
    let bundle = create_bundle(&source_paths, args.include_config)?;

    // Write to file
    fs::write(&output_path, &bundle)?;

    let size = bundle.len();
    print_success(&format!(
        "Bundle created: {} ({} bytes)",
        args.output,
        format_size(size)
    ));

    // Print contents
    println!();
    print_header("Bundle Contents");
    for path in &source_paths {
        if path.is_dir() {
            let count = WalkDir::new(path)
                .into_iter()
                .filter_map(std::result::Result::ok)
                .filter(|e| e.file_type().is_file())
                .count();
            println!("  {} ({} files)", path.display(), count);
        } else {
            println!("  {}", path.display());
        }
    }

    Ok(())
}

fn create_bundle(paths: &[PathBuf], include_config: bool) -> CliResult<Vec<u8>> {
    let mut bundle = Vec::new();
    let mut manifest = String::new();

    manifest.push_str("# Axonml Bundle Manifest\n");
    manifest.push_str("version: 1.0\n");
    manifest.push_str(&format!("created: {}\n", chrono::Utc::now().to_rfc3339()));
    manifest.push_str("files:\n");

    for base_path in paths {
        if base_path.is_file() {
            // Add single file
            let name = base_path
                .file_name()
                .map_or_else(|| "file".to_string(), |n| n.to_string_lossy().to_string());

            let data = fs::read(base_path)?;
            add_file_to_bundle(&mut bundle, &name, &data)?;
            manifest.push_str(&format!("  - {name}\n"));
        } else if base_path.is_dir() {
            // Add directory contents
            for entry in WalkDir::new(base_path)
                .into_iter()
                .filter_map(std::result::Result::ok)
            {
                if entry.file_type().is_file() {
                    let relative = entry
                        .path()
                        .strip_prefix(base_path.parent().unwrap_or(base_path))
                        .unwrap_or(entry.path());
                    let name = relative.to_string_lossy().to_string();

                    let data = fs::read(entry.path())?;
                    add_file_to_bundle(&mut bundle, &name, &data)?;
                    manifest.push_str(&format!("  - {name}\n"));
                }
            }
        }
    }

    // Add config files if requested
    if include_config {
        let config_paths = ["axonml.toml", "config.toml", "data_config.toml"];
        for config in config_paths {
            let config_path = PathBuf::from(config);
            if config_path.exists() {
                let data = fs::read(&config_path)?;
                add_file_to_bundle(&mut bundle, config, &data)?;
                manifest.push_str(&format!("  - {config}\n"));
            }
        }
    }

    // Add manifest at the beginning
    let mut final_bundle = Vec::new();
    add_file_to_bundle(&mut final_bundle, "MANIFEST.txt", manifest.as_bytes())?;
    final_bundle.extend(bundle);

    Ok(final_bundle)
}

fn add_file_to_bundle(bundle: &mut Vec<u8>, name: &str, data: &[u8]) -> CliResult<()> {
    // Simple format: [name_len:4][name:name_len][data_len:8][data:data_len]
    let name_bytes = name.as_bytes();
    let name_len = name_bytes.len() as u32;
    let data_len = data.len() as u64;

    bundle.extend_from_slice(&name_len.to_le_bytes());
    bundle.extend_from_slice(name_bytes);
    bundle.extend_from_slice(&data_len.to_le_bytes());
    bundle.extend_from_slice(data);

    Ok(())
}

// =============================================================================
// Extract Subcommand
// =============================================================================

fn execute_extract(args: ZipExtractArgs) -> CliResult<()> {
    print_header("Extract Bundle");

    let bundle_path = PathBuf::from(&args.input);
    if !path_exists(&bundle_path) {
        return Err(CliError::Other(format!("Bundle not found: {}", args.input)));
    }

    print_kv("Bundle", &args.input);
    print_kv("Output", &args.output);

    // Ensure output directory
    ensure_dir(&args.output)?;
    let output_dir = PathBuf::from(&args.output);

    println!();
    print_info("Extracting files...");

    // Read and extract bundle
    let bundle = fs::read(&bundle_path)?;
    let files = extract_bundle(&bundle)?;

    let mut count = 0;
    for (name, data) in files {
        let file_path = output_dir.join(&name);

        // Create parent directories
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Write file
        fs::write(&file_path, data)?;
        count += 1;

        if args.verbose {
            println!("  Extracted: {name}");
        }
    }

    print_success(&format!("Extracted {} files to {}", count, args.output));

    Ok(())
}

fn extract_bundle(bundle: &[u8]) -> CliResult<Vec<(String, Vec<u8>)>> {
    let mut files = Vec::new();
    let mut offset = 0;

    while offset < bundle.len() {
        // Read name length
        if offset + 4 > bundle.len() {
            break;
        }
        let name_len = u32::from_le_bytes([
            bundle[offset],
            bundle[offset + 1],
            bundle[offset + 2],
            bundle[offset + 3],
        ]) as usize;
        offset += 4;

        // Read name
        if offset + name_len > bundle.len() {
            break;
        }
        let name = String::from_utf8_lossy(&bundle[offset..offset + name_len]).to_string();
        offset += name_len;

        // Read data length
        if offset + 8 > bundle.len() {
            break;
        }
        let data_len = u64::from_le_bytes([
            bundle[offset],
            bundle[offset + 1],
            bundle[offset + 2],
            bundle[offset + 3],
            bundle[offset + 4],
            bundle[offset + 5],
            bundle[offset + 6],
            bundle[offset + 7],
        ]) as usize;
        offset += 8;

        // Read data
        if offset + data_len > bundle.len() {
            break;
        }
        let data = bundle[offset..offset + data_len].to_vec();
        offset += data_len;

        files.push((name, data));
    }

    Ok(files)
}

// =============================================================================
// List Subcommand
// =============================================================================

fn execute_list(args: ZipListArgs) -> CliResult<()> {
    print_header("Bundle Contents");

    let bundle_path = PathBuf::from(&args.input);
    if !path_exists(&bundle_path) {
        return Err(CliError::Other(format!("Bundle not found: {}", args.input)));
    }

    print_kv("Bundle", &args.input);

    let file_size = fs::metadata(&bundle_path)?.len();
    print_kv("Size", &format_size(file_size as usize));
    println!();

    // Read and list bundle
    let bundle = fs::read(&bundle_path)?;
    let files = extract_bundle(&bundle)?;

    println!("Files:");
    let mut total_size = 0usize;
    for (name, data) in &files {
        if args.detailed {
            println!("  {} ({} bytes)", name, data.len());
        } else {
            println!("  {name}");
        }
        total_size += data.len();
    }

    println!();
    print_kv("Total files", &files.len().to_string());
    print_kv("Uncompressed size", &format_size(total_size));

    Ok(())
}

// =============================================================================
// Helper Functions
// =============================================================================

fn format_size(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} bytes")
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bundle_roundtrip() {
        let mut bundle = Vec::new();
        add_file_to_bundle(&mut bundle, "test.txt", b"Hello, World!").unwrap();
        add_file_to_bundle(&mut bundle, "data/file.bin", &[1, 2, 3, 4, 5]).unwrap();

        let files = extract_bundle(&bundle).unwrap();
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].0, "test.txt");
        assert_eq!(files[0].1, b"Hello, World!");
        assert_eq!(files[1].0, "data/file.bin");
        assert_eq!(files[1].1, &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(500), "500 bytes");
        assert_eq!(format_size(1024), "1.00 KB");
        assert_eq!(format_size(1024 * 1024), "1.00 MB");
    }
}
