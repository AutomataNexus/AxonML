//! Training notebooks API endpoints for AxonML
//!
//! Handles notebook CRUD, cell execution, checkpoints, and AI assistance.

use crate::api::AppState;
use crate::auth::{AuthError, AuthUser};
use crate::db::notebooks::{
    CellOutput, CellStatus, CellType, NewCheckpoint, NewNotebook, NotebookCell,
    NotebookCheckpoint, NotebookRepository, NotebookStatus, TrainingNotebook, UpdateNotebook,
};
use crate::training::notebook_executor::result_to_cell_output;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct ListNotebooksQuery {
    #[serde(default = "default_limit")]
    pub limit: u32,
    #[serde(default)]
    pub offset: u32,
}

fn default_limit() -> u32 {
    100
}

#[derive(Debug, Deserialize)]
pub struct CreateNotebookRequest {
    pub name: String,
    pub description: Option<String>,
    #[serde(default)]
    pub cells: Vec<NotebookCellRequest>,
    pub model_id: Option<String>,
    pub dataset_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct NotebookCellRequest {
    #[serde(default)]
    pub cell_type: String,
    pub source: String,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct NotebookResponse {
    pub id: String,
    pub user_id: String,
    pub name: String,
    pub description: Option<String>,
    pub cells: Vec<CellResponse>,
    pub metadata: NotebookMetadataResponse,
    pub model_id: Option<String>,
    pub dataset_id: Option<String>,
    pub status: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Serialize)]
pub struct CellResponse {
    pub id: String,
    pub cell_type: String,
    pub source: String,
    pub outputs: Vec<CellOutput>,
    pub status: String,
    pub execution_count: Option<u32>,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct NotebookMetadataResponse {
    pub kernel: Option<String>,
    pub language: Option<String>,
    pub framework: Option<String>,
    pub tags: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct NotebookListItem {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub status: String,
    pub cell_count: usize,
    pub model_id: Option<String>,
    pub dataset_id: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Deserialize)]
pub struct UpdateNotebookRequest {
    pub name: Option<String>,
    pub description: Option<String>,
    pub cells: Option<Vec<NotebookCellRequest>>,
    pub model_id: Option<String>,
    pub dataset_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AddCellRequest {
    #[serde(default)]
    pub cell_type: String,
    pub source: String,
    pub position: Option<usize>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateCellRequest {
    pub source: Option<String>,
    pub cell_type: Option<String>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Deserialize)]
pub struct ExecuteCellRequest {
    #[serde(default)]
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct ExecuteCellResponse {
    pub cell_id: String,
    pub outputs: Vec<CellOutput>,
    pub execution_count: u32,
    pub status: String,
    pub duration_ms: u64,
}

#[derive(Debug, Deserialize)]
pub struct AiAssistRequest {
    /// User's prompt/question
    pub prompt: String,
    /// ID of the currently selected cell (if any)
    #[serde(default)]
    pub selected_cell_id: Option<String>,
    /// Type of cell to generate (code or markdown)
    #[serde(default)]
    pub cell_type: String,
    /// Whether to include imports in generated code
    #[serde(default)]
    pub include_imports: bool,
}

#[derive(Debug, Serialize)]
pub struct AiAssistResponse {
    pub suggestion: String,
    pub explanation: Option<String>,
    pub confidence: f32,
    pub model: String,
    pub tokens_generated: u32,
}

#[derive(Debug, Deserialize)]
pub struct SaveCheckpointRequest {
    pub epoch: u32,
    pub step: u32,
    pub metrics: serde_json::Value,
    pub model_state_base64: String,
    pub optimizer_state_base64: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CheckpointResponse {
    pub id: String,
    pub notebook_id: String,
    pub epoch: u32,
    pub step: u32,
    pub metrics: serde_json::Value,
    pub model_state_path: String,
    pub optimizer_state_path: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Deserialize)]
pub struct UploadModelVersionRequest {
    pub checkpoint_id: String,
    pub model_id: String,
    pub metrics: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct UploadModelVersionResponse {
    pub version_id: String,
    pub model_id: String,
    pub version: u32,
    pub checkpoint_id: String,
}

#[derive(Debug, Deserialize)]
pub struct ImportNotebookRequest {
    pub content: String,
    pub format: String,
}

#[derive(Debug, Serialize)]
pub struct ExportNotebookResponse {
    pub content: String,
    pub format: String,
    pub filename: String,
}

// ============================================================================
// Conversion Helpers
// ============================================================================

fn notebook_to_response(notebook: TrainingNotebook) -> NotebookResponse {
    NotebookResponse {
        id: notebook.id,
        user_id: notebook.user_id,
        name: notebook.name,
        description: notebook.description,
        cells: notebook.cells.into_iter().map(cell_to_response).collect(),
        metadata: NotebookMetadataResponse {
            kernel: notebook.metadata.kernel,
            language: notebook.metadata.language,
            framework: notebook.metadata.framework,
            tags: notebook.metadata.tags,
        },
        model_id: notebook.model_id,
        dataset_id: notebook.dataset_id,
        status: format!("{:?}", notebook.status).to_lowercase(),
        created_at: notebook.created_at.to_rfc3339(),
        updated_at: notebook.updated_at.to_rfc3339(),
    }
}

fn notebook_to_list_item(notebook: TrainingNotebook) -> NotebookListItem {
    NotebookListItem {
        id: notebook.id,
        name: notebook.name,
        description: notebook.description,
        status: format!("{:?}", notebook.status).to_lowercase(),
        cell_count: notebook.cells.len(),
        model_id: notebook.model_id,
        dataset_id: notebook.dataset_id,
        created_at: notebook.created_at.to_rfc3339(),
        updated_at: notebook.updated_at.to_rfc3339(),
    }
}

fn cell_to_response(cell: NotebookCell) -> CellResponse {
    CellResponse {
        id: cell.id,
        cell_type: format!("{:?}", cell.cell_type).to_lowercase(),
        source: cell.source,
        outputs: cell.outputs,
        status: format!("{:?}", cell.status).to_lowercase(),
        execution_count: cell.execution_count,
        metadata: cell.metadata,
    }
}

fn checkpoint_to_response(checkpoint: NotebookCheckpoint) -> CheckpointResponse {
    CheckpointResponse {
        id: checkpoint.id,
        notebook_id: checkpoint.notebook_id,
        epoch: checkpoint.epoch,
        step: checkpoint.step,
        metrics: checkpoint.metrics,
        model_state_path: checkpoint.model_state_path,
        optimizer_state_path: checkpoint.optimizer_state_path,
        created_at: checkpoint.created_at.to_rfc3339(),
    }
}

fn parse_cell_type(s: &str) -> CellType {
    match s.to_lowercase().as_str() {
        "markdown" | "md" => CellType::Markdown,
        _ => CellType::Code,
    }
}

fn request_to_cell(req: NotebookCellRequest) -> NotebookCell {
    NotebookCell {
        id: uuid::Uuid::new_v4().to_string(),
        cell_type: parse_cell_type(&req.cell_type),
        source: req.source,
        outputs: vec![],
        status: CellStatus::Idle,
        execution_count: None,
        metadata: req.metadata,
    }
}

// ============================================================================
// Handlers
// ============================================================================

/// List notebooks for the current user
pub async fn list_notebooks(
    State(state): State<AppState>,
    user: AuthUser,
    Query(query): Query<ListNotebooksQuery>,
) -> Result<Json<Vec<NotebookListItem>>, AuthError> {
    let repo = NotebookRepository::new(&state.db);

    let notebooks = if user.role == "admin" {
        repo.list_all(Some(query.limit), Some(query.offset)).await
    } else {
        repo.list_by_user(&user.id, Some(query.limit), Some(query.offset)).await
    }
    .map_err(|e| AuthError::Internal(e.to_string()))?;

    let response: Vec<NotebookListItem> = notebooks
        .into_iter()
        .map(notebook_to_list_item)
        .collect();

    Ok(Json(response))
}

/// Create a new training notebook
pub async fn create_notebook(
    State(state): State<AppState>,
    user: AuthUser,
    Json(req): Json<CreateNotebookRequest>,
) -> Result<(StatusCode, Json<NotebookResponse>), AuthError> {
    let repo = NotebookRepository::new(&state.db);

    let cells: Vec<NotebookCell> = req.cells.into_iter().map(request_to_cell).collect();

    let notebook = repo
        .create(NewNotebook {
            user_id: user.id,
            name: req.name,
            description: req.description,
            cells,
            model_id: req.model_id,
            dataset_id: req.dataset_id,
        })
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok((StatusCode::CREATED, Json(notebook_to_response(notebook))))
}

/// Get a notebook by ID
pub async fn get_notebook(
    State(state): State<AppState>,
    user: AuthUser,
    Path(id): Path<String>,
) -> Result<Json<NotebookResponse>, AuthError> {
    let repo = NotebookRepository::new(&state.db);

    let notebook = repo
        .find_by_id(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Notebook not found".to_string()))?;

    // Check ownership
    if notebook.user_id != user.id && user.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    Ok(Json(notebook_to_response(notebook)))
}

/// Update a notebook
pub async fn update_notebook(
    State(state): State<AppState>,
    user: AuthUser,
    Path(id): Path<String>,
    Json(req): Json<UpdateNotebookRequest>,
) -> Result<Json<NotebookResponse>, AuthError> {
    let repo = NotebookRepository::new(&state.db);

    // Check ownership first
    let existing = repo
        .find_by_id(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Notebook not found".to_string()))?;

    if existing.user_id != user.id && user.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    let cells = req.cells.map(|c| c.into_iter().map(request_to_cell).collect());

    let notebook = repo
        .update(
            &id,
            UpdateNotebook {
                name: req.name,
                description: req.description,
                cells,
                model_id: req.model_id,
                dataset_id: req.dataset_id,
                status: None,
            },
        )
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok(Json(notebook_to_response(notebook)))
}

/// Delete a notebook
pub async fn delete_notebook(
    State(state): State<AppState>,
    user: AuthUser,
    Path(id): Path<String>,
) -> Result<StatusCode, AuthError> {
    let repo = NotebookRepository::new(&state.db);

    // Check ownership first
    let existing = repo
        .find_by_id(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Notebook not found".to_string()))?;

    if existing.user_id != user.id && user.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    repo.delete(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok(StatusCode::NO_CONTENT)
}

/// Add a cell to a notebook
pub async fn add_cell(
    State(state): State<AppState>,
    user: AuthUser,
    Path(notebook_id): Path<String>,
    Json(req): Json<AddCellRequest>,
) -> Result<(StatusCode, Json<CellResponse>), AuthError> {
    let repo = NotebookRepository::new(&state.db);

    // Check ownership
    let existing = repo
        .find_by_id(&notebook_id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Notebook not found".to_string()))?;

    if existing.user_id != user.id && user.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    let cell = NotebookCell {
        id: uuid::Uuid::new_v4().to_string(),
        cell_type: parse_cell_type(&req.cell_type),
        source: req.source,
        outputs: vec![],
        status: CellStatus::Idle,
        execution_count: None,
        metadata: req.metadata,
    };

    let cell_id = cell.id.clone();
    let notebook = repo
        .add_cell(&notebook_id, cell, req.position)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    // Find the added cell
    let added_cell = notebook
        .cells
        .into_iter()
        .find(|c| c.id == cell_id)
        .ok_or(AuthError::Internal("Cell not found after add".to_string()))?;

    Ok((StatusCode::CREATED, Json(cell_to_response(added_cell))))
}

/// Update a cell in a notebook
pub async fn update_cell(
    State(state): State<AppState>,
    user: AuthUser,
    Path((notebook_id, cell_id)): Path<(String, String)>,
    Json(req): Json<UpdateCellRequest>,
) -> Result<Json<CellResponse>, AuthError> {
    let repo = NotebookRepository::new(&state.db);

    // Get notebook and check ownership
    let notebook = repo
        .find_by_id(&notebook_id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Notebook not found".to_string()))?;

    if notebook.user_id != user.id && user.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    // Find the cell
    let mut cell = notebook
        .cells
        .into_iter()
        .find(|c| c.id == cell_id)
        .ok_or(AuthError::NotFound("Cell not found".to_string()))?;

    // Apply updates
    if let Some(source) = req.source {
        cell.source = source;
    }
    if let Some(cell_type) = req.cell_type {
        cell.cell_type = parse_cell_type(&cell_type);
    }
    if let Some(metadata) = req.metadata {
        cell.metadata = metadata;
    }

    let updated_notebook = repo
        .update_cell(&notebook_id, cell)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    let updated_cell = updated_notebook
        .cells
        .into_iter()
        .find(|c| c.id == cell_id)
        .ok_or(AuthError::Internal("Cell not found after update".to_string()))?;

    Ok(Json(cell_to_response(updated_cell)))
}

/// Delete a cell from a notebook
pub async fn delete_cell(
    State(state): State<AppState>,
    user: AuthUser,
    Path((notebook_id, cell_id)): Path<(String, String)>,
) -> Result<StatusCode, AuthError> {
    let repo = NotebookRepository::new(&state.db);

    // Check ownership
    let existing = repo
        .find_by_id(&notebook_id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Notebook not found".to_string()))?;

    if existing.user_id != user.id && user.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    repo.delete_cell(&notebook_id, &cell_id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok(StatusCode::NO_CONTENT)
}

/// Execute a code cell using the notebook executor
pub async fn execute_cell(
    State(state): State<AppState>,
    user: AuthUser,
    Path((notebook_id, cell_id)): Path<(String, String)>,
    Json(req): Json<ExecuteCellRequest>,
) -> Result<Json<ExecuteCellResponse>, AuthError> {
    let repo = NotebookRepository::new(&state.db);
    let timeout_ms = req.timeout_ms.unwrap_or(60000); // Default 60 second timeout for compilation

    // Get notebook and check ownership
    let notebook = repo
        .find_by_id(&notebook_id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Notebook not found".to_string()))?;

    if notebook.user_id != user.id && user.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    // Find the cell and its index
    let cell_index = notebook
        .cells
        .iter()
        .position(|c| c.id == cell_id)
        .ok_or(AuthError::NotFound("Cell not found".to_string()))?;

    let mut cell = notebook.cells[cell_index].clone();

    // Get previous cells for context (all code cells before this one)
    let previous_cells: Vec<NotebookCell> = notebook
        .cells
        .iter()
        .take(cell_index)
        .filter(|c| c.cell_type == CellType::Code)
        .cloned()
        .collect();

    // Mark as running
    cell.status = CellStatus::Running;
    repo.update_cell(&notebook_id, cell.clone())
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    let start = std::time::Instant::now();
    let execution_count = cell.execution_count.unwrap_or(0) + 1;

    // Execute the cell using the notebook executor
    let result = state.notebook_executor
        .execute_cell(&cell, &previous_cells, timeout_ms)
        .await;

    let duration = start.elapsed();

    // Convert result to cell output
    let output = result_to_cell_output(result, execution_count);

    // Update cell status based on output
    let status = if output.output_type == "error" {
        cell.status = CellStatus::Error;
        "error"
    } else {
        cell.status = CellStatus::Completed;
        "completed"
    };

    cell.outputs = vec![output.clone()];
    cell.execution_count = Some(execution_count);

    repo.update_cell(&notebook_id, cell)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok(Json(ExecuteCellResponse {
        cell_id,
        outputs: vec![output],
        execution_count,
        status: status.to_string(),
        duration_ms: duration.as_millis() as u64,
    }))
}

/// Get AI assistance for a cell
pub async fn ai_assist(
    State(state): State<AppState>,
    _user: AuthUser,
    Path(notebook_id): Path<String>,
    Json(req): Json<AiAssistRequest>,
) -> Result<Json<AiAssistResponse>, AuthError> {
    // Check if Ollama is available
    if !state.ollama.is_available().await {
        let suggestion = match parse_cell_type(&req.cell_type) {
            CellType::Code => format!(
                "# AI service unavailable\n# Prompt: {}\n\n# Please ensure Ollama is running with:\n# ollama serve\n# And pull a model with:\n# ollama pull qwen2.5-coder:7b",
                req.prompt
            ),
            CellType::Markdown => format!(
                "# {}\n\n*AI service unavailable. Please start Ollama to get AI assistance.*",
                req.prompt
            ),
        };
        return Ok(Json(AiAssistResponse {
            suggestion,
            explanation: Some("Ollama LLM service is not available. Please start it with 'ollama serve'.".to_string()),
            confidence: 0.0,
            model: String::new(),
            tokens_generated: 0,
        }));
    }

    // Fetch the notebook to get full context
    let notebook_repo = NotebookRepository::new(&state.db);
    let notebook = notebook_repo
        .find_by_id(&notebook_id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Notebook not found".to_string()))?;

    // Build context from notebook
    let context = build_notebook_context(&state, &notebook, req.selected_cell_id.as_deref()).await;

    // Generate with Ollama
    let result = match parse_cell_type(&req.cell_type) {
        CellType::Code => {
            state.ollama.generate_code(
                &req.prompt,
                Some(&context),
                req.include_imports,
            ).await
        }
        CellType::Markdown => {
            state.ollama.generate_markdown(&req.prompt).await
        }
    };

    match result {
        Ok(suggestion) => {
            Ok(Json(AiAssistResponse {
                suggestion: suggestion.code,
                explanation: suggestion.explanation,
                confidence: 0.85,
                model: suggestion.model,
                tokens_generated: suggestion.tokens_generated,
            }))
        }
        Err(e) => {
            tracing::error!("AI assist generation failed: {}", e);
            Ok(Json(AiAssistResponse {
                suggestion: format!("# Error generating suggestion\n# {}", e),
                explanation: Some(format!("Generation failed: {}", e)),
                confidence: 0.0,
                model: String::new(),
                tokens_generated: 0,
            }))
        }
    }
}

/// Build comprehensive context from the notebook for AI assistance
async fn build_notebook_context(
    state: &AppState,
    notebook: &crate::db::notebooks::TrainingNotebook,
    selected_cell_id: Option<&str>,
) -> String {
    let mut context = String::new();

    // Notebook metadata
    context.push_str(&format!("# Notebook: {}\n", notebook.name));
    if let Some(desc) = &notebook.description {
        context.push_str(&format!("# Description: {}\n", desc));
    }
    context.push_str("\n");

    // Model info if linked
    if let Some(model_id) = &notebook.model_id {
        if let Ok(Some(model)) = crate::db::models::ModelRepository::new(&state.db)
            .find_by_id(model_id)
            .await
        {
            context.push_str("## Linked Model\n");
            context.push_str(&format!("# Name: {}\n", model.name));
            context.push_str(&format!("# Type: {}\n", model.model_type));
            if let Some(desc) = &model.description {
                context.push_str(&format!("# Description: {}\n", desc));
            }
            context.push_str("\n");
        }
    }

    // Dataset info if linked
    if let Some(dataset_id) = &notebook.dataset_id {
        if let Ok(Some(dataset)) = crate::db::datasets::DatasetRepository::new(&state.db)
            .find_by_id(dataset_id)
            .await
        {
            context.push_str("## Linked Dataset\n");
            context.push_str(&format!("# Name: {}\n", dataset.name));
            context.push_str(&format!("# Type: {:?}\n", dataset.dataset_type));
            if let Some(desc) = &dataset.description {
                context.push_str(&format!("# Description: {}\n", desc));
            }
            context.push_str("\n");
        }
    }

    // Find selected cell position
    let selected_idx = selected_cell_id.and_then(|id| {
        notebook.cells.iter().position(|c| c.id == id)
    });

    // Include all cells as context, marking the selected one
    context.push_str("## Notebook Cells\n\n");
    for (i, cell) in notebook.cells.iter().enumerate() {
        let is_selected = selected_idx == Some(i);
        let cell_marker = if is_selected { " [SELECTED - Generate for this cell]" } else { "" };

        match cell.cell_type {
            crate::db::notebooks::CellType::Code => {
                context.push_str(&format!("### Cell {} (Code){}\n```python\n{}\n```\n\n",
                    i + 1, cell_marker, cell.source));
            }
            crate::db::notebooks::CellType::Markdown => {
                context.push_str(&format!("### Cell {} (Markdown){}\n{}\n\n",
                    i + 1, cell_marker, cell.source));
            }
        }
    }

    // Add hints based on what's in the notebook
    let has_model_def = notebook.cells.iter().any(|c|
        c.source.contains("class") && (c.source.contains("nn.Module") || c.source.contains("Model"))
    );
    let has_dataloader = notebook.cells.iter().any(|c|
        c.source.contains("DataLoader") || c.source.contains("Dataset")
    );
    let has_optimizer = notebook.cells.iter().any(|c|
        c.source.contains("optim.") || c.source.contains("optimizer")
    );
    let has_training_loop = notebook.cells.iter().any(|c|
        c.source.contains("for epoch") || c.source.contains("train(")
    );

    context.push_str("## Current State\n");
    context.push_str(&format!("# Has model definition: {}\n", has_model_def));
    context.push_str(&format!("# Has data loading: {}\n", has_dataloader));
    context.push_str(&format!("# Has optimizer: {}\n", has_optimizer));
    context.push_str(&format!("# Has training loop: {}\n", has_training_loop));

    // Suggest what might be needed next
    if !has_model_def {
        context.push_str("# Suggestion: User may need a model definition\n");
    } else if !has_dataloader {
        context.push_str("# Suggestion: User may need data loading code\n");
    } else if !has_optimizer {
        context.push_str("# Suggestion: User may need optimizer setup\n");
    } else if !has_training_loop {
        context.push_str("# Suggestion: User may need a training loop\n");
    }

    context
}

/// Save a training checkpoint
pub async fn save_checkpoint(
    State(state): State<AppState>,
    user: AuthUser,
    Path(notebook_id): Path<String>,
    Json(req): Json<SaveCheckpointRequest>,
) -> Result<(StatusCode, Json<CheckpointResponse>), AuthError> {
    let repo = NotebookRepository::new(&state.db);

    // Check notebook ownership
    let notebook = repo
        .find_by_id(&notebook_id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Notebook not found".to_string()))?;

    if notebook.user_id != user.id && user.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    // Decode and save model state
    let model_state = base64::Engine::decode(
        &base64::engine::general_purpose::STANDARD,
        &req.model_state_base64,
    )
    .map_err(|e| AuthError::Internal(format!("Invalid base64 for model state: {}", e)))?;

    let checkpoint_id = uuid::Uuid::new_v4().to_string();
    let checkpoints_dir = state.config.checkpoints_dir().join(&notebook_id);
    std::fs::create_dir_all(&checkpoints_dir)
        .map_err(|e| AuthError::Internal(format!("Failed to create checkpoints dir: {}", e)))?;

    let model_state_path = checkpoints_dir.join(format!("{}_model.bin", checkpoint_id));
    std::fs::write(&model_state_path, &model_state)
        .map_err(|e| AuthError::Internal(format!("Failed to write model state: {}", e)))?;

    // Optionally save optimizer state
    let optimizer_state_path = if let Some(opt_state_b64) = req.optimizer_state_base64 {
        let opt_state = base64::Engine::decode(
            &base64::engine::general_purpose::STANDARD,
            &opt_state_b64,
        )
        .map_err(|e| AuthError::Internal(format!("Invalid base64 for optimizer state: {}", e)))?;

        let opt_path = checkpoints_dir.join(format!("{}_optimizer.bin", checkpoint_id));
        std::fs::write(&opt_path, &opt_state)
            .map_err(|e| AuthError::Internal(format!("Failed to write optimizer state: {}", e)))?;

        Some(opt_path.to_string_lossy().to_string())
    } else {
        None
    };

    // Save checkpoint to database
    let checkpoint = repo
        .create_checkpoint(NewCheckpoint {
            notebook_id: notebook_id.clone(),
            epoch: req.epoch,
            step: req.step,
            metrics: req.metrics,
            model_state_path: model_state_path.to_string_lossy().to_string(),
            optimizer_state_path,
        })
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok((StatusCode::CREATED, Json(checkpoint_to_response(checkpoint))))
}

/// List checkpoints for a notebook
pub async fn list_checkpoints(
    State(state): State<AppState>,
    user: AuthUser,
    Path(notebook_id): Path<String>,
) -> Result<Json<Vec<CheckpointResponse>>, AuthError> {
    let repo = NotebookRepository::new(&state.db);

    // Check notebook ownership
    let notebook = repo
        .find_by_id(&notebook_id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Notebook not found".to_string()))?;

    if notebook.user_id != user.id && user.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    let checkpoints = repo
        .list_checkpoints(&notebook_id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    let response: Vec<CheckpointResponse> = checkpoints
        .into_iter()
        .map(checkpoint_to_response)
        .collect();

    Ok(Json(response))
}

/// Get best checkpoint by metric
pub async fn get_best_checkpoint(
    State(state): State<AppState>,
    user: AuthUser,
    Path(notebook_id): Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<Option<CheckpointResponse>>, AuthError> {
    let repo = NotebookRepository::new(&state.db);

    // Check notebook ownership
    let notebook = repo
        .find_by_id(&notebook_id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Notebook not found".to_string()))?;

    if notebook.user_id != user.id && user.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    let metric_key = params.get("metric").map(|s| s.as_str()).unwrap_or("loss");
    let minimize = params.get("minimize").map(|s| s == "true").unwrap_or(true);

    let checkpoint = repo
        .get_best_checkpoint(&notebook_id, metric_key, minimize)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok(Json(checkpoint.map(checkpoint_to_response)))
}

/// Upload a checkpoint as a model version
pub async fn upload_model_version(
    State(state): State<AppState>,
    user: AuthUser,
    Path(notebook_id): Path<String>,
    Json(req): Json<UploadModelVersionRequest>,
) -> Result<(StatusCode, Json<UploadModelVersionResponse>), AuthError> {
    let notebook_repo = NotebookRepository::new(&state.db);

    // Check notebook ownership
    let notebook = notebook_repo
        .find_by_id(&notebook_id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Notebook not found".to_string()))?;

    if notebook.user_id != user.id && user.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    // Get the checkpoint
    let checkpoint = notebook_repo
        .get_checkpoint(&req.checkpoint_id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Checkpoint not found".to_string()))?;

    // Create model version from checkpoint
    use crate::db::models::ModelRepository;
    let model_repo = ModelRepository::new(&state.db);

    // Get model to verify it exists
    let _model = model_repo
        .find_by_id(&req.model_id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Model not found".to_string()))?;

    // Get existing versions to determine next version number
    let versions = model_repo
        .list_versions(&req.model_id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;
    let version_number = versions.iter().map(|v| v.version).max().unwrap_or(0) + 1;
    let versions_dir = state.config.models_dir().join(&req.model_id);
    std::fs::create_dir_all(&versions_dir)
        .map_err(|e| AuthError::Internal(format!("Failed to create versions dir: {}", e)))?;

    let version_path = versions_dir.join(format!("v{}.bin", version_number));
    std::fs::copy(&checkpoint.model_state_path, &version_path)
        .map_err(|e| AuthError::Internal(format!("Failed to copy model file: {}", e)))?;

    let file_size = std::fs::metadata(&version_path)
        .map_err(|e| AuthError::Internal(format!("Failed to get file size: {}", e)))?
        .len();

    // Create version record
    let version = model_repo
        .create_version(crate::db::models::NewModelVersion {
            model_id: req.model_id.clone(),
            file_path: version_path.to_string_lossy().to_string(),
            file_size,
            metrics: req.metrics.or(Some(checkpoint.metrics)),
            training_run_id: None,
        })
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok((
        StatusCode::CREATED,
        Json(UploadModelVersionResponse {
            version_id: version.id,
            model_id: req.model_id,
            version: version.version,
            checkpoint_id: req.checkpoint_id,
        }),
    ))
}

/// Import a notebook from ipynb or other format
pub async fn import_notebook(
    State(state): State<AppState>,
    user: AuthUser,
    Json(req): Json<ImportNotebookRequest>,
) -> Result<(StatusCode, Json<NotebookResponse>), AuthError> {
    let repo = NotebookRepository::new(&state.db);

    let (name, cells) = match req.format.to_lowercase().as_str() {
        "ipynb" | "jupyter" => {
            // Parse Jupyter notebook format
            let ipynb: serde_json::Value = serde_json::from_str(&req.content)
                .map_err(|e| AuthError::Internal(format!("Invalid ipynb format: {}", e)))?;

            let name = ipynb
                .get("metadata")
                .and_then(|m| m.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("Imported Notebook")
                .to_string();

            let cells: Vec<NotebookCell> = ipynb
                .get("cells")
                .and_then(|c| c.as_array())
                .map(|cells| {
                    cells
                        .iter()
                        .map(|cell| {
                            let cell_type = cell
                                .get("cell_type")
                                .and_then(|t| t.as_str())
                                .unwrap_or("code");
                            let source = cell
                                .get("source")
                                .map(|s| {
                                    if let Some(arr) = s.as_array() {
                                        arr.iter()
                                            .filter_map(|v| v.as_str())
                                            .collect::<Vec<_>>()
                                            .join("")
                                    } else {
                                        s.as_str().unwrap_or("").to_string()
                                    }
                                })
                                .unwrap_or_default();

                            NotebookCell {
                                id: uuid::Uuid::new_v4().to_string(),
                                cell_type: parse_cell_type(cell_type),
                                source,
                                outputs: vec![],
                                status: CellStatus::Idle,
                                execution_count: None,
                                metadata: HashMap::new(),
                            }
                        })
                        .collect()
                })
                .unwrap_or_default();

            (name, cells)
        }
        _ => {
            return Err(AuthError::Internal(format!(
                "Unsupported format: {}",
                req.format
            )));
        }
    };

    let notebook = repo
        .create(NewNotebook {
            user_id: user.id,
            name,
            description: Some("Imported notebook".to_string()),
            cells,
            model_id: None,
            dataset_id: None,
        })
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok((StatusCode::CREATED, Json(notebook_to_response(notebook))))
}

/// Export a notebook to ipynb or other format
pub async fn export_notebook(
    State(state): State<AppState>,
    user: AuthUser,
    Path(id): Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Json<ExportNotebookResponse>, AuthError> {
    let repo = NotebookRepository::new(&state.db);

    let notebook = repo
        .find_by_id(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Notebook not found".to_string()))?;

    if notebook.user_id != user.id && user.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    let format = params.get("format").map(|s| s.as_str()).unwrap_or("ipynb");

    let (content, filename) = match format {
        "ipynb" | "jupyter" => {
            let cells: Vec<serde_json::Value> = notebook
                .cells
                .iter()
                .map(|cell| {
                    serde_json::json!({
                        "cell_type": format!("{:?}", cell.cell_type).to_lowercase(),
                        "source": cell.source.split('\n').map(|l| format!("{}\n", l)).collect::<Vec<_>>(),
                        "metadata": cell.metadata,
                        "outputs": cell.outputs,
                        "execution_count": cell.execution_count,
                    })
                })
                .collect();

            let ipynb = serde_json::json!({
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {
                    "name": notebook.name,
                    "kernelspec": {
                        "name": "axonml",
                        "display_name": "AxonML",
                        "language": "rust"
                    },
                    "language_info": {
                        "name": "rust",
                        "version": "1.70.0"
                    }
                },
                "cells": cells
            });

            let content = serde_json::to_string_pretty(&ipynb)
                .map_err(|e| AuthError::Internal(e.to_string()))?;

            let filename = format!(
                "{}.ipynb",
                notebook.name.to_lowercase().replace(' ', "_")
            );

            (content, filename)
        }
        _ => {
            return Err(AuthError::Internal(format!(
                "Unsupported format: {}",
                format
            )));
        }
    };

    Ok(Json(ExportNotebookResponse {
        content,
        format: format.to_string(),
        filename,
    }))
}

/// Start notebook execution (run all cells)
pub async fn start_notebook(
    State(state): State<AppState>,
    user: AuthUser,
    Path(id): Path<String>,
) -> Result<Json<NotebookResponse>, AuthError> {
    let repo = NotebookRepository::new(&state.db);

    let notebook = repo
        .find_by_id(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Notebook not found".to_string()))?;

    if notebook.user_id != user.id && user.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    // Update status to running
    let notebook = repo
        .update_status(&id, NotebookStatus::Running)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    // Spawn background task to execute all cells
    let notebook_id = id.clone();
    let db = state.db.clone();
    let executor = state.notebook_executor.clone();

    tokio::spawn(async move {
        let repo = NotebookRepository::new(&db);

        // Get fresh notebook data
        let notebook = match repo.find_by_id(&notebook_id).await {
            Ok(Some(nb)) => nb,
            _ => {
                tracing::error!(notebook_id = %notebook_id, "Failed to load notebook for execution");
                return;
            }
        };

        let cells = notebook.cells.clone();
        let mut executed_cells = Vec::new();
        let mut all_success = true;

        // Execute each code cell in sequence
        for (idx, cell) in cells.iter().enumerate() {
            if cell.cell_type != CellType::Code {
                executed_cells.push(cell.clone());
                continue;
            }

            // Get previous cells for context
            let previous_cells: Vec<NotebookCell> = cells[..idx]
                .iter()
                .filter(|c| c.cell_type == CellType::Code)
                .cloned()
                .collect();

            // Update cell status to running
            let mut running_cell = cell.clone();
            running_cell.status = CellStatus::Running;
            let _ = repo.update_cell(&notebook_id, running_cell.clone()).await;

            // Execute the cell
            let result = executor
                .execute_cell(cell, &previous_cells, 60000) // 60 second timeout
                .await;

            // Update cell with result
            let execution_count = cell.execution_count.unwrap_or(0) + 1;
            let output = result_to_cell_output(result, execution_count);

            let mut completed_cell = cell.clone();
            completed_cell.execution_count = Some(execution_count);
            completed_cell.outputs = vec![output.clone()];
            completed_cell.status = if output.output_type == "error" {
                all_success = false;
                CellStatus::Error
            } else {
                CellStatus::Completed
            };

            let _ = repo.update_cell(&notebook_id, completed_cell.clone()).await;
            executed_cells.push(completed_cell);

            // Stop on error
            if !all_success {
                break;
            }
        }

        // Update notebook status
        let final_status = if all_success {
            NotebookStatus::Completed
        } else {
            NotebookStatus::Failed
        };

        let _ = repo.update_status(&notebook_id, final_status).await;
        tracing::info!(notebook_id = %notebook_id, status = ?final_status, "Notebook execution completed");
    });

    Ok(Json(notebook_to_response(notebook)))
}

/// Stop notebook execution
pub async fn stop_notebook(
    State(state): State<AppState>,
    user: AuthUser,
    Path(id): Path<String>,
) -> Result<Json<NotebookResponse>, AuthError> {
    let repo = NotebookRepository::new(&state.db);

    let notebook = repo
        .find_by_id(&id)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?
        .ok_or(AuthError::NotFound("Notebook not found".to_string()))?;

    if notebook.user_id != user.id && user.role != "admin" {
        return Err(AuthError::Unauthorized);
    }

    // Update status to stopped
    let notebook = repo
        .update_status(&id, NotebookStatus::Stopped)
        .await
        .map_err(|e| AuthError::Internal(e.to_string()))?;

    Ok(Json(notebook_to_response(notebook)))
}
