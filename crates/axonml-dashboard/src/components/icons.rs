//! SVG Icon Components
//!
//! Provides inline SVG icons for use throughout the dashboard.

use leptos::*;

/// Icon size variants
#[derive(Debug, Clone, Copy, Default)]
pub enum IconSize {
    Xs,
    Sm,
    #[default]
    Md,
    Lg,
    Xl,
}

impl IconSize {
    pub fn class(&self) -> &'static str {
        match self {
            Self::Xs => "icon-xs",
            Self::Sm => "icon-sm",
            Self::Md => "icon-md",
            Self::Lg => "icon-lg",
            Self::Xl => "icon-xl",
        }
    }

    pub fn size(&self) -> u32 {
        match self {
            Self::Xs => 12,
            Self::Sm => 16,
            Self::Md => 20,
            Self::Lg => 24,
            Self::Xl => 32,
        }
    }
}

/// Base icon component
#[component]
fn IconBase(
    #[prop(into)] path: String,
    #[prop(default = IconSize::Md)] size: IconSize,
    #[prop(optional, into)] class: String,
    #[prop(default = "none")] fill: &'static str,
    #[prop(default = "currentColor")] stroke: &'static str,
    #[prop(default = 2.0)] stroke_width: f64,
) -> impl IntoView {
    let s = size.size();
    view! {
        <svg
            xmlns="http://www.w3.org/2000/svg"
            width=s
            height=s
            viewBox="0 0 24 24"
            fill=fill
            stroke=stroke
            stroke-width=stroke_width
            stroke-linecap="round"
            stroke-linejoin="round"
            class=format!("icon {} {}", size.class(), class)
            inner_html=path
        />
    }
}

// Navigation Icons

#[component]
pub fn IconHome(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconDashboard(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<rect width="7" height="9" x="3" y="3" rx="1"/><rect width="7" height="5" x="14" y="3" rx="1"/><rect width="7" height="9" x="14" y="12" rx="1"/><rect width="7" height="5" x="3" y="16" rx="1"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconActivity(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconBox(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconServer(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<rect width="20" height="8" x="2" y="2" rx="2" ry="2"/><rect width="20" height="8" x="2" y="14" rx="2" ry="2"/><line x1="6" x2="6.01" y1="6" y2="6"/><line x1="6" x2="6.01" y1="18" y2="18"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconSettings(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconUser(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconUsers(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconLogout(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" x2="9" y1="12" y2="12"/>"#.to_string() size=size class=class /> }
}

// Action Icons

#[component]
pub fn IconPlus(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<line x1="12" x2="12" y1="5" y2="19"/><line x1="5" x2="19" y1="12" y2="12"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconMinus(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<line x1="5" x2="19" y1="12" y2="12"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconX(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<line x1="18" x2="6" y1="6" y2="18"/><line x1="6" x2="18" y1="6" y2="18"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconCheck(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<polyline points="20 6 9 17 4 12"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconEdit(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/><path d="m15 5 4 4"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconTrash(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconDownload(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconUpload(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconSearch(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<circle cx="11" cy="11" r="8"/><line x1="21" x2="16.65" y1="21" y2="16.65"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconRefresh(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M3 21v-5h5"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconCopy(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>"#.to_string() size=size class=class /> }
}

// Status Icons

#[component]
pub fn IconCheckCircle(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconXCircle(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<circle cx="12" cy="12" r="10"/><line x1="15" x2="9" y1="9" y2="15"/><line x1="9" x2="15" y1="9" y2="15"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconAlertCircle(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconAlertTriangle(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" x2="12" y1="9" y2="13"/><line x1="12" x2="12.01" y1="17" y2="17"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconInfo(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="16" y2="12"/><line x1="12" x2="12.01" y1="8" y2="8"/>"#.to_string() size=size class=class /> }
}

// Direction Icons

#[component]
pub fn IconChevronLeft(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<polyline points="15 18 9 12 15 6"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconChevronRight(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<polyline points="9 18 15 12 9 6"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconChevronUp(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<polyline points="18 15 12 9 6 15"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconChevronDown(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<polyline points="6 9 12 15 18 9"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconArrowLeft(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<line x1="19" x2="5" y1="12" y2="12"/><polyline points="12 19 5 12 12 5"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconArrowRight(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<line x1="5" x2="19" y1="12" y2="12"/><polyline points="12 5 19 12 12 19"/>"#.to_string() size=size class=class /> }
}

// Misc Icons

#[component]
pub fn IconMenu(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<line x1="4" x2="20" y1="12" y2="12"/><line x1="4" x2="20" y1="6" y2="6"/><line x1="4" x2="20" y1="18" y2="18"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconEye(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconEyeOff(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<path d="M9.88 9.88a3 3 0 1 0 4.24 4.24"/><path d="M10.73 5.08A10.43 10.43 0 0 1 12 5c7 0 10 7 10 7a13.16 13.16 0 0 1-1.67 2.68"/><path d="M6.61 6.61A13.526 13.526 0 0 0 2 12s3 7 10 7a9.74 9.74 0 0 0 5.39-1.61"/><line x1="2" x2="22" y1="2" y2="22"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconClock(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconCalendar(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<rect width="18" height="18" x="3" y="4" rx="2" ry="2"/><line x1="16" x2="16" y1="2" y2="6"/><line x1="8" x2="8" y1="2" y2="6"/><line x1="3" x2="21" y1="10" y2="10"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconPlay(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<polygon points="5 3 19 12 5 21 5 3"/>"#.to_string() size=size class=class fill="currentColor" stroke="none" /> }
}

#[component]
pub fn IconPause(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<rect width="4" height="16" x="6" y="4"/><rect width="4" height="16" x="14" y="4"/>"#.to_string() size=size class=class fill="currentColor" stroke="none" /> }
}

#[component]
pub fn IconStop(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<rect width="14" height="14" x="5" y="5" rx="2"/>"#.to_string() size=size class=class fill="currentColor" stroke="none" /> }
}

#[component]
pub fn IconKey(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<path d="m21 2-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0 3 3L22 7l-3-3m-3.5 3.5L19 4"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconShield(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconLock(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconUnlock(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 9.9-1"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconExternalLink(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" x2="21" y1="14" y2="3"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconGithub(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"/><path d="M9 18c-4.51 2-5-2-7-2"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconMoreVertical(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<circle cx="12" cy="12" r="1"/><circle cx="12" cy="5" r="1"/><circle cx="12" cy="19" r="1"/>"#.to_string() size=size class=class fill="currentColor" /> }
}

#[component]
pub fn IconMoreHorizontal(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<circle cx="12" cy="12" r="1"/><circle cx="5" cy="12" r="1"/><circle cx="19" cy="12" r="1"/>"#.to_string() size=size class=class fill="currentColor" /> }
}

#[component]
pub fn IconTerminal(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<polyline points="4 17 10 11 4 5"/><line x1="12" x2="20" y1="19" y2="19"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconCode(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconCpu(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconDatabase(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconBrain(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-1.54"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-1.54"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconLayers(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconZap(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconTrendingUp(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconTrendingDown(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconBarChart(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<line x1="12" x2="12" y1="20" y2="10"/><line x1="18" x2="18" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="16"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconLineChart(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconQrCode(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<rect width="5" height="5" x="3" y="3" rx="1"/><rect width="5" height="5" x="16" y="3" rx="1"/><rect width="5" height="5" x="3" y="16" rx="1"/><path d="M21 16h-3a2 2 0 0 0-2 2v3"/><path d="M21 21v.01"/><path d="M12 7v3a2 2 0 0 1-2 2H7"/><path d="M3 12h.01"/><path d="M12 3h.01"/><path d="M12 16v.01"/><path d="M16 12h1"/><path d="M21 12v.01"/><path d="M12 21v-1"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconSmartphone(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<rect width="14" height="20" x="5" y="2" rx="2" ry="2"/><line x1="12" x2="12.01" y1="18" y2="18"/>"#.to_string() size=size class=class /> }
}

#[component]
pub fn IconFingerprint(#[prop(default = IconSize::Md)] size: IconSize, #[prop(optional, into)] class: String) -> impl IntoView {
    view! { <IconBase path=r#"<path d="M2 12C2 6.5 6.5 2 12 2a10 10 0 0 1 8 4"/><path d="M5 19.5C5.5 18 6 15 6 12c0-.7.12-1.37.34-2"/><path d="M17.29 21.02c.12-.6.43-2.3.5-3.02"/><path d="M12 10a2 2 0 0 0-2 2c0 1.02-.1 2.51-.26 4"/><path d="M8.65 22c.21-.66.45-1.32.57-2"/><path d="M14 13.12c0 2.38 0 6.38-1 8.88"/><path d="M2 16h.01"/><path d="M21.8 16c.2-2 .131-5.354 0-6"/><path d="M9 6.8a6 6 0 0 1 9 5.2c0 .47 0 1.17-.02 2"/>"#.to_string() size=size class=class /> }
}
