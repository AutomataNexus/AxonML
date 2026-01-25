//! Ollama Client for Local LLM Inference
//!
//! Provides async client for communicating with Ollama API.

use reqwest::Client;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Default Ollama endpoint
pub const DEFAULT_OLLAMA_URL: &str = "http://127.0.0.1:11434";

/// Default model for code generation
pub const DEFAULT_CODE_MODEL: &str = "qwen2.5-coder:7b";

#[derive(Error, Debug)]
pub enum OllamaError {
    #[error("HTTP request failed: {0}")]
    Request(#[from] reqwest::Error),

    #[error("Ollama service unavailable")]
    ServiceUnavailable,

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Generation failed: {0}")]
    GenerationFailed(String),
}

/// Ollama generate request
#[derive(Debug, Serialize)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<i64>>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<GenerateOptions>,
}

/// Generation options
#[derive(Debug, Serialize, Default)]
pub struct GenerateOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
}

/// Ollama generate response
#[derive(Debug, Deserialize)]
pub struct GenerateResponse {
    pub model: String,
    pub response: String,
    pub done: bool,
    #[serde(default)]
    pub context: Option<Vec<i64>>,
    #[serde(default)]
    pub total_duration: Option<u64>,
    #[serde(default)]
    pub load_duration: Option<u64>,
    #[serde(default)]
    pub prompt_eval_count: Option<u32>,
    #[serde(default)]
    pub eval_count: Option<u32>,
    #[serde(default)]
    pub eval_duration: Option<u64>,
}

/// Ollama client for LLM inference
#[derive(Clone)]
pub struct OllamaClient {
    client: Client,
    base_url: String,
    model: String,
}

impl OllamaClient {
    /// Create a new Ollama client with default settings
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: DEFAULT_OLLAMA_URL.to_string(),
            model: DEFAULT_CODE_MODEL.to_string(),
        }
    }

    /// Create with custom URL and model
    pub fn with_config(base_url: &str, model: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.to_string(),
            model: model.to_string(),
        }
    }

    /// Check if Ollama service is available
    pub async fn is_available(&self) -> bool {
        let url = format!("{}/api/tags", self.base_url);
        self.client.get(&url).send().await.is_ok()
    }

    /// Generate code based on a prompt
    pub async fn generate_code(
        &self,
        prompt: &str,
        context: Option<&str>,
        include_imports: bool,
    ) -> Result<CodeSuggestion, OllamaError> {
        let system_prompt = build_code_system_prompt(include_imports);
        let full_prompt = if let Some(ctx) = context {
            format!("Context:\n```\n{}\n```\n\nTask: {}", ctx, prompt)
        } else {
            prompt.to_string()
        };

        let request = GenerateRequest {
            model: self.model.clone(),
            prompt: full_prompt,
            system: Some(system_prompt),
            template: None,
            context: None,
            stream: false,
            options: Some(GenerateOptions {
                temperature: Some(0.7),
                top_p: Some(0.9),
                num_predict: Some(1024),
                stop: Some(vec!["```".to_string()]),
                ..Default::default()
            }),
        };

        let response = self.generate(request).await?;

        // Extract code from response
        let (code, explanation) = extract_code_and_explanation(&response.response);

        Ok(CodeSuggestion {
            code,
            explanation,
            model: response.model,
            tokens_generated: response.eval_count.unwrap_or(0),
        })
    }

    /// Generate markdown content
    pub async fn generate_markdown(
        &self,
        prompt: &str,
    ) -> Result<CodeSuggestion, OllamaError> {
        let system_prompt = r#"You are a technical documentation writer for machine learning projects.
Generate clear, well-structured markdown documentation.
Use proper headings, lists, and code blocks where appropriate.
Be concise but comprehensive."#.to_string();

        let request = GenerateRequest {
            model: self.model.clone(),
            prompt: prompt.to_string(),
            system: Some(system_prompt),
            template: None,
            context: None,
            stream: false,
            options: Some(GenerateOptions {
                temperature: Some(0.7),
                num_predict: Some(512),
                ..Default::default()
            }),
        };

        let response = self.generate(request).await?;

        Ok(CodeSuggestion {
            code: response.response.trim().to_string(),
            explanation: None,
            model: response.model,
            tokens_generated: response.eval_count.unwrap_or(0),
        })
    }

    /// Raw generate call to Ollama
    pub async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse, OllamaError> {
        let url = format!("{}/api/generate", self.base_url);

        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(OllamaError::ModelNotFound(request.model));
        }

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(OllamaError::GenerationFailed(format!(
                "Status {}: {}", status, body
            )));
        }

        let result: GenerateResponse = response.json().await?;
        Ok(result)
    }
}

impl Default for OllamaClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of code generation
#[derive(Debug)]
pub struct CodeSuggestion {
    pub code: String,
    pub explanation: Option<String>,
    pub model: String,
    pub tokens_generated: u32,
}

/// Build system prompt for code generation
fn build_code_system_prompt(include_imports: bool) -> String {
    let import_instruction = if include_imports {
        "Include necessary imports at the top of the code."
    } else {
        "Do not include imports, assume they are already available."
    };

    format!(r#"You are an expert machine learning engineer and Python/Rust programmer.
You are helping a user write code for the AxonML machine learning framework.

AxonML is similar to PyTorch with these key modules:
- axonml.nn: Neural network layers (Linear, Conv2d, BatchNorm, ReLU, etc.)
- axonml.optim: Optimizers (SGD, Adam, AdamW)
- axonml.data: DataLoader, Dataset
- axonml.autograd: Automatic differentiation

Guidelines:
- Write clean, well-documented code
- {}
- Use type hints where appropriate
- Follow ML best practices
- Keep code concise but readable

Respond with ONLY the code, no explanations unless asked.
If you need to explain, put explanations in code comments."#, import_instruction)
}

/// Extract code and explanation from LLM response
fn extract_code_and_explanation(response: &str) -> (String, Option<String>) {
    let response = response.trim();

    // Check if response contains code blocks
    if response.contains("```") {
        let mut code_parts = Vec::new();
        let mut explanation_parts = Vec::new();
        let mut in_code_block = false;
        let mut current_block = String::new();

        for line in response.lines() {
            if line.starts_with("```") {
                if in_code_block {
                    // End of code block
                    code_parts.push(current_block.trim().to_string());
                    current_block.clear();
                }
                in_code_block = !in_code_block;
            } else if in_code_block {
                current_block.push_str(line);
                current_block.push('\n');
            } else {
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    explanation_parts.push(trimmed.to_string());
                }
            }
        }

        let code = code_parts.join("\n\n");
        let explanation = if explanation_parts.is_empty() {
            None
        } else {
            Some(explanation_parts.join(" "))
        };

        (code, explanation)
    } else {
        // No code blocks, treat entire response as code
        (response.to_string(), None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_code_simple() {
        let response = "def hello():\n    print('Hello')";
        let (code, explanation) = extract_code_and_explanation(response);
        assert_eq!(code, response);
        assert!(explanation.is_none());
    }

    #[test]
    fn test_extract_code_with_blocks() {
        let response = "Here's the code:\n```python\ndef hello():\n    print('Hello')\n```\nThis prints hello.";
        let (code, explanation) = extract_code_and_explanation(response);
        assert!(code.contains("def hello()"));
        assert!(explanation.is_some());
        assert!(explanation.unwrap().contains("prints hello"));
    }
}
