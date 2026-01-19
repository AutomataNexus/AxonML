//! Format Detection and Management
//!
//! Handles different serialization formats for model storage.

use std::path::Path;

// =============================================================================
// Format Enum
// =============================================================================

/// Supported serialization formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// Axonml native binary format (.axonml, .bin)
    Axonml,
    /// JSON format for debugging (.json)
    Json,
    /// `SafeTensors` format (.safetensors)
    SafeTensors,
}

impl Format {
    /// Get the file extension for this format.
    #[must_use] pub fn extension(&self) -> &'static str {
        match self {
            Format::Axonml => "axonml",
            Format::Json => "json",
            Format::SafeTensors => "safetensors",
        }
    }

    /// Get a human-readable name for this format.
    #[must_use] pub fn name(&self) -> &'static str {
        match self {
            Format::Axonml => "Axonml Native",
            Format::Json => "JSON",
            Format::SafeTensors => "SafeTensors",
        }
    }

    /// Check if this format is binary.
    #[must_use] pub fn is_binary(&self) -> bool {
        match self {
            Format::Axonml => true,
            Format::Json => false,
            Format::SafeTensors => true,
        }
    }

    /// Check if this format supports streaming.
    #[must_use] pub fn supports_streaming(&self) -> bool {
        match self {
            Format::Axonml => true,
            Format::Json => false,
            Format::SafeTensors => true,
        }
    }

    /// Get all supported formats.
    #[must_use] pub fn all() -> &'static [Format] {
        &[Format::Axonml, Format::Json, Format::SafeTensors]
    }
}

impl std::fmt::Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// =============================================================================
// Format Detection
// =============================================================================

/// Detect the format from a file path based on extension.
pub fn detect_format<P: AsRef<Path>>(path: P) -> Format {
    let path = path.as_ref();

    match path.extension().and_then(|e| e.to_str()) {
        Some("axonml") => Format::Axonml,
        Some("bin") => Format::Axonml,
        Some("json") => Format::Json,
        Some("safetensors") => Format::SafeTensors,
        Some("st") => Format::SafeTensors,
        _ => Format::Axonml, // default
    }
}

/// Detect format from file contents (magic bytes).
#[must_use] pub fn detect_format_from_bytes(bytes: &[u8]) -> Option<Format> {
    if bytes.len() < 8 {
        return None;
    }

    // Check for JSON (starts with '{' or '[')
    if bytes[0] == b'{' || bytes[0] == b'[' {
        return Some(Format::Json);
    }

    // SafeTensors has a specific header format
    // First 8 bytes are the header size as u64 little-endian
    // Then the header is JSON
    if bytes.len() >= 16 {
        let header_size = u64::from_le_bytes(bytes[0..8].try_into().ok()?);
        if header_size < 10_000_000 && bytes.get(8) == Some(&b'{') {
            return Some(Format::SafeTensors);
        }
    }

    // Default to Axonml binary format
    Some(Format::Axonml)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_format_from_extension() {
        assert_eq!(detect_format("model.axonml"), Format::Axonml);
        assert_eq!(detect_format("model.bin"), Format::Axonml);
        assert_eq!(detect_format("model.json"), Format::Json);
        assert_eq!(detect_format("model.safetensors"), Format::SafeTensors);
        assert_eq!(detect_format("model.st"), Format::SafeTensors);
        assert_eq!(detect_format("model.unknown"), Format::Axonml);
    }

    #[test]
    fn test_format_properties() {
        assert!(Format::Axonml.is_binary());
        assert!(!Format::Json.is_binary());
        assert!(Format::SafeTensors.is_binary());

        assert_eq!(Format::Axonml.extension(), "axonml");
        assert_eq!(Format::Json.extension(), "json");
    }

    #[test]
    fn test_detect_format_from_bytes() {
        // JSON
        assert_eq!(
            detect_format_from_bytes(b"{\"key\": \"value\"}"),
            Some(Format::Json)
        );

        // Binary (default to Axonml)
        assert_eq!(
            detect_format_from_bytes(&[0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]),
            Some(Format::Axonml)
        );
    }

    #[test]
    fn test_format_display() {
        assert_eq!(format!("{}", Format::Axonml), "Axonml Native");
        assert_eq!(format!("{}", Format::Json), "JSON");
    }
}
