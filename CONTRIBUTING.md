# Contributing to Axonml

Thank you for your interest in contributing to Axonml! This document provides guidelines and information for contributors.

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributors of all experience levels.

## Getting Started

### Prerequisites

- Rust 1.70 or later
- Cargo (comes with Rust)
- Git

### Setting Up the Development Environment

```bash
# Clone the repository
git clone https://github.com/automatanexus/axonml.git
cd axonml

# Build all crates
cargo build --workspace

# Run tests
cargo test --workspace

# Run clippy (linting)
cargo clippy --workspace

# Format code
cargo fmt --all
```

## How to Contribute

### Reporting Bugs

1. Check existing issues to avoid duplicates
2. Use the bug report template
3. Include:
   - Rust version (`rustc --version`)
   - Operating system
   - Minimal reproduction code
   - Expected vs actual behavior

### Suggesting Features

1. Check existing issues and discussions
2. Use the feature request template
3. Describe:
   - Use case and motivation
   - Proposed API (if applicable)
   - Alternatives considered

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`cargo test --workspace`)
6. Run clippy (`cargo clippy --workspace`)
7. Format code (`cargo fmt --all`)
8. Commit with clear messages
9. Push to your fork
10. Open a Pull Request

## Code Guidelines

### Rust Style

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `cargo fmt` for formatting
- Use `cargo clippy` for linting
- Document public APIs with doc comments

### File Structure

Each source file should follow this structure:

```rust
//! Module Name - Brief Description
//!
//! Detailed description of the module.
//!
//! @version X.Y.Z
//! @author AutomataNexus Development Team

// =============================================================================
// Imports
// =============================================================================

use ...;

// =============================================================================
// Constants
// =============================================================================

// =============================================================================
// Types and Traits
// =============================================================================

// =============================================================================
// Implementations
// =============================================================================

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example() {
        // ...
    }
}
```

### Testing

- Write unit tests for new functionality
- Place tests in a `tests` submodule or in `tests/`
- Use descriptive test names: `test_tensor_addition_broadcasts_correctly`
- Test edge cases and error conditions

### Documentation

- Document all public items
- Include examples in doc comments
- Update README if adding significant features
- Add entries to CHANGELOG.md

## Crate Guidelines

### Adding Dependencies

- Minimize external dependencies
- Prefer well-maintained, widely-used crates
- Feature-gate optional dependencies
- Document why each dependency is needed

### Performance

- Benchmark performance-critical code
- Avoid unnecessary allocations
- Use iterators over loops where appropriate
- Consider SIMD for hot paths

### Safety

- Avoid `unsafe` unless absolutely necessary
- Document safety invariants for any `unsafe` code
- Prefer safe abstractions

## Pull Request Process

1. **Title**: Use a clear, descriptive title
2. **Description**: Explain what and why (not just how)
3. **Tests**: Include tests for changes
4. **Documentation**: Update docs as needed
5. **Changelog**: Add entry for user-facing changes

### PR Checklist

- [ ] Code compiles without warnings
- [ ] All tests pass
- [ ] Clippy passes without warnings
- [ ] Code is formatted with `cargo fmt`
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if applicable)

## Release Process

Releases follow semantic versioning (SemVer):

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes, backwards compatible

## Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Email**: dev@automatanexus.dev

## Recognition

Contributors are recognized in:
- CHANGELOG.md for their contributions
- GitHub contributors page

Thank you for contributing to Axonml!

---

*AutomataNexus Development Team*
