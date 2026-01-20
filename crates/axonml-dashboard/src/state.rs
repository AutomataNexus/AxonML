//! Global State Management for AxonML Dashboard
//!
//! Provides reactive state signals for the application.

use leptos::*;
use gloo_storage::{LocalStorage, Storage};

use crate::types::*;

/// Key for storing access token in localStorage
const ACCESS_TOKEN_KEY: &str = "access_token";
/// Key for storing refresh token in localStorage
const REFRESH_TOKEN_KEY: &str = "refresh_token";
/// Key for storing user data in localStorage
const USER_KEY: &str = "user";

/// Toast notification type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToastType {
    Success,
    Error,
    Warning,
    Info,
}

impl ToastType {
    pub fn class(&self) -> &'static str {
        match self {
            Self::Success => "toast-success",
            Self::Error => "toast-error",
            Self::Warning => "toast-warning",
            Self::Info => "toast-info",
        }
    }

    pub fn icon(&self) -> &'static str {
        match self {
            Self::Success => "check-circle",
            Self::Error => "x-circle",
            Self::Warning => "alert-triangle",
            Self::Info => "info",
        }
    }
}

/// Toast notification
#[derive(Debug, Clone)]
pub struct Toast {
    pub id: u64,
    pub toast_type: ToastType,
    pub title: String,
    pub message: String,
    pub duration_ms: u64,
}

/// Global application state
#[derive(Clone)]
pub struct AppState {
    /// Current authenticated user
    pub user: RwSignal<Option<User>>,
    /// Whether the app is loading initial state
    pub loading: RwSignal<bool>,
    /// Whether the sidebar is collapsed
    pub sidebar_collapsed: RwSignal<bool>,
    /// Current theme (light/dark)
    pub dark_mode: RwSignal<bool>,
    /// Active toast notifications
    pub toasts: RwSignal<Vec<Toast>>,
    /// Toast ID counter
    toast_counter: RwSignal<u64>,
}

impl AppState {
    /// Create new app state
    pub fn new() -> Self {
        // Try to restore user from localStorage
        let stored_user: Option<User> = LocalStorage::get(USER_KEY).ok();

        Self {
            user: RwSignal::new(stored_user),
            loading: RwSignal::new(true),
            sidebar_collapsed: RwSignal::new(false),
            dark_mode: RwSignal::new(false),
            toasts: RwSignal::new(Vec::new()),
            toast_counter: RwSignal::new(0),
        }
    }

    /// Check if user is authenticated
    pub fn is_authenticated(&self) -> bool {
        self.user.get().is_some()
    }

    /// Get the current user if authenticated
    pub fn current_user(&self) -> Option<User> {
        self.user.get()
    }

    /// Check if current user has admin role
    pub fn is_admin(&self) -> bool {
        self.user.get().map(|u| u.role == UserRole::Admin).unwrap_or(false)
    }

    /// Store authentication tokens and user
    pub fn set_auth(&self, access_token: &str, refresh_token: &str, user: User) {
        let _ = LocalStorage::set(ACCESS_TOKEN_KEY, access_token);
        let _ = LocalStorage::set(REFRESH_TOKEN_KEY, refresh_token);
        let _ = LocalStorage::set(USER_KEY, &user);
        self.user.set(Some(user));
    }

    /// Clear authentication state
    pub fn clear_auth(&self) {
        LocalStorage::delete(ACCESS_TOKEN_KEY);
        LocalStorage::delete(REFRESH_TOKEN_KEY);
        LocalStorage::delete(USER_KEY);
        self.user.set(None);
    }

    /// Get stored access token
    pub fn get_access_token(&self) -> Option<String> {
        LocalStorage::get(ACCESS_TOKEN_KEY).ok()
    }

    /// Get stored refresh token
    pub fn get_refresh_token(&self) -> Option<String> {
        LocalStorage::get(REFRESH_TOKEN_KEY).ok()
    }

    /// Update stored access token
    pub fn update_access_token(&self, token: &str) {
        let _ = LocalStorage::set(ACCESS_TOKEN_KEY, token);
    }

    /// Toggle sidebar collapsed state
    pub fn toggle_sidebar(&self) {
        self.sidebar_collapsed.update(|v| *v = !*v);
    }

    /// Toggle dark mode
    pub fn toggle_dark_mode(&self) {
        self.dark_mode.update(|v| *v = !*v);
    }

    /// Show a toast notification
    pub fn show_toast(&self, toast_type: ToastType, title: impl Into<String>, message: impl Into<String>) {
        self.toast_counter.update(|c| *c += 1);
        let id = self.toast_counter.get();

        let toast = Toast {
            id,
            toast_type,
            title: title.into(),
            message: message.into(),
            duration_ms: 5000,
        };

        self.toasts.update(|toasts| toasts.push(toast));

        // Auto-remove toast after duration
        let toasts = self.toasts;
        set_timeout(
            move || {
                toasts.update(|t| t.retain(|toast| toast.id != id));
            },
            std::time::Duration::from_millis(5000),
        );
    }

    /// Show success toast
    pub fn toast_success(&self, title: impl Into<String>, message: impl Into<String>) {
        self.show_toast(ToastType::Success, title, message);
    }

    /// Show error toast
    pub fn toast_error(&self, title: impl Into<String>, message: impl Into<String>) {
        self.show_toast(ToastType::Error, title, message);
    }

    /// Show warning toast
    pub fn toast_warning(&self, title: impl Into<String>, message: impl Into<String>) {
        self.show_toast(ToastType::Warning, title, message);
    }

    /// Show info toast
    pub fn toast_info(&self, title: impl Into<String>, message: impl Into<String>) {
        self.show_toast(ToastType::Info, title, message);
    }

    /// Remove a specific toast
    pub fn remove_toast(&self, id: u64) {
        self.toasts.update(|t| t.retain(|toast| toast.id != id));
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

/// Provide app state context
pub fn provide_app_state() -> AppState {
    let state = AppState::new();
    provide_context(state.clone());
    state
}

/// Use app state from context
pub fn use_app_state() -> AppState {
    expect_context::<AppState>()
}

// ============================================================================
// Training State
// ============================================================================

/// State for training runs page
#[derive(Clone)]
pub struct TrainingState {
    /// All training runs
    pub runs: RwSignal<Vec<TrainingRun>>,
    /// Currently selected run
    pub selected_run: RwSignal<Option<TrainingRun>>,
    /// Real-time metrics for selected run
    pub live_metrics: RwSignal<Vec<TrainingMetrics>>,
    /// Logs for selected run
    pub logs: RwSignal<Vec<LogEntry>>,
    /// Loading state
    pub loading: RwSignal<bool>,
    /// Error message
    pub error: RwSignal<Option<String>>,
}

impl TrainingState {
    pub fn new() -> Self {
        Self {
            runs: RwSignal::new(Vec::new()),
            selected_run: RwSignal::new(None),
            live_metrics: RwSignal::new(Vec::new()),
            logs: RwSignal::new(Vec::new()),
            loading: RwSignal::new(false),
            error: RwSignal::new(None),
        }
    }
}

impl Default for TrainingState {
    fn default() -> Self {
        Self::new()
    }
}

/// Provide training state context
pub fn provide_training_state() -> TrainingState {
    let state = TrainingState::new();
    provide_context(state.clone());
    state
}

/// Use training state from context
pub fn use_training_state() -> TrainingState {
    expect_context::<TrainingState>()
}

// ============================================================================
// Models State
// ============================================================================

/// State for models page
#[derive(Clone)]
pub struct ModelsState {
    /// All models
    pub models: RwSignal<Vec<Model>>,
    /// Currently selected model
    pub selected_model: RwSignal<Option<Model>>,
    /// Versions of selected model
    pub versions: RwSignal<Vec<ModelVersion>>,
    /// Loading state
    pub loading: RwSignal<bool>,
    /// Error message
    pub error: RwSignal<Option<String>>,
}

impl ModelsState {
    pub fn new() -> Self {
        Self {
            models: RwSignal::new(Vec::new()),
            selected_model: RwSignal::new(None),
            versions: RwSignal::new(Vec::new()),
            loading: RwSignal::new(false),
            error: RwSignal::new(None),
        }
    }
}

impl Default for ModelsState {
    fn default() -> Self {
        Self::new()
    }
}

/// Provide models state context
pub fn provide_models_state() -> ModelsState {
    let state = ModelsState::new();
    provide_context(state.clone());
    state
}

/// Use models state from context
pub fn use_models_state() -> ModelsState {
    expect_context::<ModelsState>()
}

// ============================================================================
// Inference State
// ============================================================================

/// State for inference page
#[derive(Clone)]
pub struct InferenceState {
    /// All endpoints
    pub endpoints: RwSignal<Vec<InferenceEndpoint>>,
    /// Currently selected endpoint
    pub selected_endpoint: RwSignal<Option<InferenceEndpoint>>,
    /// Metrics for selected endpoint
    pub metrics: RwSignal<Vec<InferenceMetrics>>,
    /// Loading state
    pub loading: RwSignal<bool>,
    /// Error message
    pub error: RwSignal<Option<String>>,
}

impl InferenceState {
    pub fn new() -> Self {
        Self {
            endpoints: RwSignal::new(Vec::new()),
            selected_endpoint: RwSignal::new(None),
            metrics: RwSignal::new(Vec::new()),
            loading: RwSignal::new(false),
            error: RwSignal::new(None),
        }
    }
}

impl Default for InferenceState {
    fn default() -> Self {
        Self::new()
    }
}

/// Provide inference state context
pub fn provide_inference_state() -> InferenceState {
    let state = InferenceState::new();
    provide_context(state.clone());
    state
}

/// Use inference state from context
pub fn use_inference_state() -> InferenceState {
    expect_context::<InferenceState>()
}
