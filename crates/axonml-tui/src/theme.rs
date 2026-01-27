//! Theme - NexusForge Color Theme for Axonml TUI
//!
//! Brand colors matching the NexusForge web UI.
//! - Teal for primary/user elements
//! - Cream for AI/assistant elements
//! - Terracotta accents
//!
//! @version 0.1.0
//! @author AutomataNexus Development Team

use ratatui::style::{Color, Modifier, Style};

// =============================================================================
// Brand Colors (from NexusForge theme.ts)
// =============================================================================

/// Primary teal color
pub const TEAL: Color = Color::Rgb(20, 184, 166); // #14b8a6

/// Light teal for headers/logo
pub const TEAL_LIGHT: Color = Color::Rgb(94, 234, 212); // #5eead4

/// Terracotta accent color
pub const TERRACOTTA: Color = Color::Rgb(196, 164, 132); // #c4a484

/// Cream color for light text
pub const CREAM: Color = Color::Rgb(250, 249, 246); // #faf9f6

/// Dark slate for backgrounds
pub const DARK_SLATE: Color = Color::Rgb(30, 41, 59); // #1e293b

/// Warm background
pub const WARM_BG: Color = Color::Rgb(245, 235, 224); // #f5ebe0

// =============================================================================
// Text Colors
// =============================================================================

/// Primary text color
pub const TEXT_PRIMARY: Color = Color::Rgb(17, 24, 39); // #111827

/// Secondary text color
pub const TEXT_SECONDARY: Color = Color::Rgb(107, 114, 128); // #6b7280

/// Muted text color
pub const TEXT_MUTED: Color = Color::Rgb(156, 163, 175); // #9ca3af

// =============================================================================
// Status Colors
// =============================================================================

/// Success color (green)
pub const SUCCESS: Color = Color::Rgb(16, 185, 129); // #10b981

/// Warning color (amber)
pub const WARNING: Color = Color::Rgb(245, 158, 11); // #f59e0b

/// Error color (red)
pub const ERROR: Color = Color::Rgb(239, 68, 68); // #ef4444

/// Info color (blue)
pub const INFO: Color = Color::Rgb(100, 181, 246); // #64b5f6

// =============================================================================
// Code Colors
// =============================================================================

/// Code background
pub const CODE_BG: Color = DARK_SLATE;

/// Code text
pub const CODE_TEXT: Color = Color::Rgb(226, 232, 240); // #e2e8f0

/// Code keyword
pub const CODE_KEYWORD: Color = TEAL;

// =============================================================================
// Style Presets
// =============================================================================

/// Axonml TUI Theme with pre-configured styles
pub struct AxonmlTheme;

impl AxonmlTheme {
    // -------------------------------------------------------------------------
    // Base Styles
    // -------------------------------------------------------------------------

    /// Default style
    pub fn default() -> Style {
        Style::default().fg(CREAM).bg(DARK_SLATE)
    }

    /// Header style (teal light, bold)
    pub fn header() -> Style {
        Style::default().fg(TEAL_LIGHT).add_modifier(Modifier::BOLD)
    }

    /// Title style (teal, bold)
    pub fn title() -> Style {
        Style::default().fg(TEAL).add_modifier(Modifier::BOLD)
    }

    /// Accent style (terracotta)
    pub fn accent() -> Style {
        Style::default().fg(TERRACOTTA)
    }

    /// Muted style
    pub fn muted() -> Style {
        Style::default().fg(TEXT_MUTED)
    }

    // -------------------------------------------------------------------------
    // Status Styles
    // -------------------------------------------------------------------------

    /// Success style
    pub fn success() -> Style {
        Style::default().fg(SUCCESS)
    }

    /// Warning style
    pub fn warning() -> Style {
        Style::default().fg(WARNING)
    }

    /// Error style
    pub fn error() -> Style {
        Style::default().fg(ERROR)
    }

    /// Info style
    pub fn info() -> Style {
        Style::default().fg(INFO)
    }

    // -------------------------------------------------------------------------
    // Component Styles
    // -------------------------------------------------------------------------

    /// Border style (normal)
    pub fn border() -> Style {
        Style::default().fg(TEXT_MUTED)
    }

    /// Border style (focused)
    pub fn border_focused() -> Style {
        Style::default().fg(TEAL)
    }

    /// Border style (active)
    pub fn border_active() -> Style {
        Style::default().fg(TEAL_LIGHT)
    }

    /// Selected item style
    pub fn selected() -> Style {
        Style::default()
            .fg(DARK_SLATE)
            .bg(TEAL)
            .add_modifier(Modifier::BOLD)
    }

    /// Highlighted style
    pub fn highlight() -> Style {
        Style::default().fg(CREAM).bg(TERRACOTTA)
    }

    // -------------------------------------------------------------------------
    // Data Visualization Styles
    // -------------------------------------------------------------------------

    /// Graph line style (primary)
    pub fn graph_primary() -> Style {
        Style::default().fg(TEAL)
    }

    /// Graph line style (secondary)
    pub fn graph_secondary() -> Style {
        Style::default().fg(TERRACOTTA)
    }

    /// Graph line style (tertiary)
    pub fn graph_tertiary() -> Style {
        Style::default().fg(INFO)
    }

    /// Graph axis style
    pub fn graph_axis() -> Style {
        Style::default().fg(TEXT_MUTED)
    }

    /// Graph label style
    pub fn graph_label() -> Style {
        Style::default().fg(CREAM)
    }

    // -------------------------------------------------------------------------
    // Model Architecture Styles
    // -------------------------------------------------------------------------

    /// Layer type style
    pub fn layer_type() -> Style {
        Style::default().fg(TEAL_LIGHT).add_modifier(Modifier::BOLD)
    }

    /// Layer params style
    pub fn layer_params() -> Style {
        Style::default().fg(TERRACOTTA)
    }

    /// Layer shape style
    pub fn layer_shape() -> Style {
        Style::default().fg(INFO)
    }

    // -------------------------------------------------------------------------
    // Training Styles
    // -------------------------------------------------------------------------

    /// Epoch style
    pub fn epoch() -> Style {
        Style::default().fg(TEAL).add_modifier(Modifier::BOLD)
    }

    /// Loss style (good - decreasing)
    pub fn loss_good() -> Style {
        Style::default().fg(SUCCESS)
    }

    /// Loss style (neutral)
    pub fn loss_neutral() -> Style {
        Style::default().fg(WARNING)
    }

    /// Loss style (bad - increasing)
    pub fn loss_bad() -> Style {
        Style::default().fg(ERROR)
    }

    /// Metric label style
    pub fn metric_label() -> Style {
        Style::default().fg(TEXT_MUTED)
    }

    /// Metric value style
    pub fn metric_value() -> Style {
        Style::default().fg(CREAM).add_modifier(Modifier::BOLD)
    }

    // -------------------------------------------------------------------------
    // Tab Styles
    // -------------------------------------------------------------------------

    /// Tab inactive style
    pub fn tab_inactive() -> Style {
        Style::default().fg(TEXT_MUTED)
    }

    /// Tab active style
    pub fn tab_active() -> Style {
        Style::default().fg(TEAL).add_modifier(Modifier::BOLD)
    }

    // -------------------------------------------------------------------------
    // Help/Key Styles
    // -------------------------------------------------------------------------

    /// Key binding style
    pub fn key() -> Style {
        Style::default().fg(TEAL_LIGHT).add_modifier(Modifier::BOLD)
    }

    /// Key description style
    pub fn key_desc() -> Style {
        Style::default().fg(CREAM)
    }
}

// =============================================================================
// ASCII Art
// =============================================================================

/// Axonml ASCII logo
pub const FERRITE_LOGO: &str = r#"
███████╗███████╗██████╗ ██████╗ ██╗████████╗███████╗
██╔════╝██╔════╝██╔══██╗██╔══██╗██║╚══██╔══╝██╔════╝
█████╗  █████╗  ██████╔╝██████╔╝██║   ██║   █████╗
██╔══╝  ██╔══╝  ██╔══██╗██╔══██╗██║   ██║   ██╔══╝
██║     ███████╗██║  ██║██║  ██║██║   ██║   ███████╗
╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝   ╚═╝   ╚══════╝
"#;

/// Small Axonml logo
pub const FERRITE_LOGO_SMALL: &str = "⚙ Axonml";

/// Circuit underline decoration
pub const CIRCUIT_UNDERLINE: &str = "ᔖ~~~~~~~~~~~~~~~~~~~~~~~◉~~~~~~~~~~~~~~~~~~~~~~~ᔙ";

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_colors_are_valid() {
        // Verify RGB colors are in valid range
        assert!(matches!(TEAL, Color::Rgb(20, 184, 166)));
        assert!(matches!(SUCCESS, Color::Rgb(16, 185, 129)));
    }

    #[test]
    fn test_styles_are_created() {
        let _ = AxonmlTheme::default();
        let _ = AxonmlTheme::header();
        let _ = AxonmlTheme::success();
    }
}
