//! Data Table Components

use std::rc::Rc;
use leptos::*;
use crate::components::icons::*;

/// Sort direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection {
    Ascending,
    Descending,
}

impl SortDirection {
    pub fn toggle(&self) -> Self {
        match self {
            Self::Ascending => Self::Descending,
            Self::Descending => Self::Ascending,
        }
    }
}

/// Table column definition
#[derive(Clone)]
pub struct TableColumn<T: Clone + 'static> {
    pub key: String,
    pub label: String,
    pub sortable: bool,
    pub render: Rc<dyn Fn(&T) -> View + 'static>,
}

impl<T: Clone + 'static> TableColumn<T> {
    pub fn new(
        key: impl Into<String>,
        label: impl Into<String>,
        render: impl Fn(&T) -> View + 'static,
    ) -> Self {
        Self {
            key: key.into(),
            label: label.into(),
            sortable: false,
            render: Rc::new(render),
        }
    }

    pub fn sortable(mut self) -> Self {
        self.sortable = true;
        self
    }
}

/// Data table component
#[component]
pub fn DataTable<T: Clone + 'static>(
    #[prop(into)] columns: Vec<TableColumn<T>>,
    #[prop(into)] data: MaybeSignal<Vec<T>>,
    #[prop(optional, into)] class: String,
    #[prop(optional)] on_row_click: Option<Callback<T>>,
    #[prop(optional)] empty_message: Option<String>,
) -> impl IntoView {
    let (sort_column, set_sort_column) = create_signal::<Option<String>>(None);
    let (sort_direction, set_sort_direction) = create_signal(SortDirection::Ascending);

    let handle_sort = move |key: String, sortable: bool| {
        if !sortable {
            return;
        }
        if sort_column.get().as_ref() == Some(&key) {
            set_sort_direction.update(|d| *d = d.toggle());
        } else {
            set_sort_column.set(Some(key));
            set_sort_direction.set(SortDirection::Ascending);
        }
    };

    let columns_clone = columns.clone();

    view! {
        <div class=format!("table-container {}", class)>
            <table class="data-table">
                <thead>
                    <tr>
                        {columns.iter().map(|col| {
                            let key = col.key.clone();
                            let key_for_class = key.clone();
                            let key_for_click = key.clone();
                            let key_for_sort = key.clone();
                            let sortable = col.sortable;
                            let label = col.label.clone();

                            view! {
                                <th
                                    class=move || format!(
                                        "table-header {} {}",
                                        if sortable { "sortable" } else { "" },
                                        if sort_column.get().as_ref() == Some(&key_for_class) { "sorted" } else { "" }
                                    )
                                    on:click=move |_| handle_sort(key_for_click.clone(), sortable)
                                >
                                    <div class="header-content">
                                        <span>{label.clone()}</span>
                                        <Show when=move || sortable>
                                            <span class="sort-icon">
                                                {
                                                    let key = key_for_sort.clone();
                                                    move || {
                                                        if sort_column.get().as_ref() == Some(&key) {
                                                            match sort_direction.get() {
                                                                SortDirection::Ascending => view! { <IconChevronUp size=IconSize::Sm /> },
                                                                SortDirection::Descending => view! { <IconChevronDown size=IconSize::Sm /> },
                                                            }
                                                        } else {
                                                            view! { <IconChevronUp size=IconSize::Sm class="text-muted".to_string() /> }
                                                        }
                                                    }
                                                }
                                            </span>
                                        </Show>
                                    </div>
                                </th>
                            }
                        }).collect_view()}
                    </tr>
                </thead>
                <tbody>
                    {move || {
                        let rows = data.get();
                        if rows.is_empty() {
                            let msg = empty_message.clone().unwrap_or_else(|| "No data available".to_string());
                            let col_count = columns_clone.len();
                            return view! {
                                <tr>
                                    <td colspan=col_count class="table-empty">
                                        <div class="empty-state">
                                            <IconBox size=IconSize::Xl class="text-muted".to_string() />
                                            <p>{msg}</p>
                                        </div>
                                    </td>
                                </tr>
                            }.into_view();
                        }

                        rows.iter().map(|row| {
                            let row_clone = row.clone();
                            let row_for_click = row.clone();
                            let clickable = on_row_click.is_some();
                            let on_click = on_row_click.clone();

                            view! {
                                <tr
                                    class=move || if clickable { "clickable" } else { "" }
                                    on:click=move |_| {
                                        if let Some(cb) = on_click.as_ref() {
                                            cb.call(row_for_click.clone());
                                        }
                                    }
                                >
                                    {columns_clone.iter().map(|col| {
                                        let cell_view = (col.render)(&row_clone);
                                        view! { <td>{cell_view}</td> }
                                    }).collect_view()}
                                </tr>
                            }
                        }).collect_view()
                    }}
                </tbody>
            </table>
        </div>
    }
}

/// Simple key-value table
#[component]
pub fn KeyValueTable(
    #[prop(into)] data: MaybeSignal<Vec<(String, String)>>,
    #[prop(optional, into)] class: String,
) -> impl IntoView {
    view! {
        <table class=format!("kv-table {}", class)>
            <tbody>
                {move || {
                    data.get().into_iter().map(|(key, value)| {
                        view! {
                            <tr>
                                <td class="kv-key">{key}</td>
                                <td class="kv-value">{value}</td>
                            </tr>
                        }
                    }).collect_view()
                }}
            </tbody>
        </table>
    }
}

/// Status badge component
#[component]
pub fn StatusBadge(
    #[prop(into)] status: String,
    #[prop(optional, into)] class: String,
) -> impl IntoView {
    let badge_class = match status.to_lowercase().as_str() {
        "running" | "active" | "starting" => "badge-success",
        "completed" | "success" => "badge-success",
        "failed" | "error" => "badge-error",
        "stopped" | "paused" => "badge-warning",
        "pending" | "queued" => "badge-info",
        _ => "badge-default",
    };

    view! {
        <span class=format!("badge {} {}", badge_class, class)>
            {status}
        </span>
    }
}

/// Pagination component
#[component]
pub fn Pagination(
    #[prop(into)] current_page: RwSignal<u32>,
    #[prop(into)] total_pages: MaybeSignal<u32>,
    #[prop(default = 5)] visible_pages: u32,
    #[prop(optional, into)] class: String,
) -> impl IntoView {
    let go_to_page = move |page: u32| {
        let total = total_pages.get();
        if page >= 1 && page <= total {
            current_page.set(page);
        }
    };

    let page_numbers = move || {
        let current = current_page.get();
        let total = total_pages.get();
        let half = visible_pages / 2;

        let start = if current <= half {
            1
        } else if current > total - half {
            total.saturating_sub(visible_pages - 1).max(1)
        } else {
            current - half
        };

        let end = (start + visible_pages - 1).min(total);

        (start..=end).collect::<Vec<_>>()
    };

    view! {
        <div class=format!("pagination {}", class)>
            <button
                class="btn btn-ghost pagination-btn"
                disabled=move || current_page.get() <= 1
                on:click=move |_| go_to_page(current_page.get() - 1)
            >
                <IconChevronLeft size=IconSize::Sm />
            </button>

            <Show when=move || { page_numbers().first().copied().unwrap_or(1) > 1 }>
                <button
                    class="btn btn-ghost pagination-btn"
                    on:click=move |_| go_to_page(1)
                >
                    "1"
                </button>
                <Show when=move || { page_numbers().first().copied().unwrap_or(1) > 2 }>
                    <span class="pagination-ellipsis">"..."</span>
                </Show>
            </Show>

            {move || {
                page_numbers().into_iter().map(|page| {
                    let is_current = move || current_page.get() == page;
                    view! {
                        <button
                            class=move || format!(
                                "btn pagination-btn {}",
                                if is_current() { "btn-primary" } else { "btn-ghost" }
                            )
                            on:click=move |_| go_to_page(page)
                        >
                            {page}
                        </button>
                    }
                }).collect_view()
            }}

            <Show when=move || { page_numbers().last().copied().unwrap_or(1) < total_pages.get() }>
                <Show when=move || { page_numbers().last().copied().unwrap_or(1) < total_pages.get().saturating_sub(1) }>
                    <span class="pagination-ellipsis">"..."</span>
                </Show>
                <button
                    class="btn btn-ghost pagination-btn"
                    on:click=move |_| go_to_page(total_pages.get())
                >
                    {move || total_pages.get()}
                </button>
            </Show>

            <button
                class="btn btn-ghost pagination-btn"
                disabled=move || current_page.get() >= total_pages.get()
                on:click=move |_| go_to_page(current_page.get() + 1)
            >
                <IconChevronRight size=IconSize::Sm />
            </button>
        </div>
    }
}
