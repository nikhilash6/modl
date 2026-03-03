//! HuggingFace Hub API client for searching and pulling models directly.
//!
//! Supports `modl pull hf:owner/repo` and `modl search --hf <query>`.

use anyhow::{Context, Result};
use serde::Deserialize;

const HF_API_BASE: &str = "https://huggingface.co/api";

// ── API response types ───────────────────────────────────────────────────

/// Model info from HuggingFace API.
/// Search endpoint returns a subset of fields; individual model endpoint returns all.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct HfModel {
    /// Full model ID like "stabilityai/stable-diffusion-xl-base-1.0"
    pub id: String,

    #[serde(default)]
    pub author: Option<String>,

    #[serde(default)]
    pub tags: Vec<String>,

    #[serde(default)]
    pub downloads: Option<u64>,

    #[serde(default)]
    pub likes: Option<u64>,

    #[serde(default, rename = "lastModified")]
    pub last_modified: Option<String>,

    #[serde(default)]
    pub pipeline_tag: Option<String>,

    /// File listing — only present when fetching individual model info.
    #[serde(default)]
    pub siblings: Vec<HfSibling>,
}

#[derive(Debug, Deserialize)]
pub struct HfSibling {
    #[serde(rename = "rfilename")]
    pub filename: String,
    #[serde(default)]
    pub size: Option<u64>,
}

/// A resolved file ready for download from HuggingFace.
#[derive(Debug)]
#[allow(dead_code)]
pub struct HfResolvedFile {
    pub repo_id: String,
    pub filename: String,
    pub url: String,
    pub size: u64,
}

// ── Public API ──────────────────────────────────────────────────────────

/// Search HuggingFace Hub for models matching a query.
/// Filters to model types relevant to the AI image generation ecosystem.
pub async fn search(query: &str, token: Option<&str>, limit: usize) -> Result<Vec<HfModel>> {
    let client = build_client(token)?;

    // Search with relevant pipeline tags for image generation
    let url = format!(
        "{}/models?search={}&sort=downloads&direction=-1&limit={}&filter=diffusers",
        HF_API_BASE,
        urlencoding::encode(query),
        limit,
    );

    let resp = client
        .get(&url)
        .send()
        .await
        .context("Failed to search HuggingFace")?;

    if !resp.status().is_success() {
        anyhow::bail!("HuggingFace search failed: HTTP {}", resp.status());
    }

    let models: Vec<HfModel> = resp.json().await.context("Failed to parse HF response")?;
    Ok(models)
}

/// Fetch full model info including file listing.
pub async fn get_model(repo_id: &str, token: Option<&str>) -> Result<HfModel> {
    let client = build_client(token)?;

    let url = format!("{}/models/{}", HF_API_BASE, repo_id);

    let resp = client
        .get(&url)
        .send()
        .await
        .context("Failed to fetch model info from HuggingFace")?;

    if resp.status().as_u16() == 404 {
        anyhow::bail!("Model '{}' not found on HuggingFace", repo_id);
    }
    if !resp.status().is_success() {
        anyhow::bail!("HuggingFace API error: HTTP {}", resp.status());
    }

    let model: HfModel = resp.json().await.context("Failed to parse HF model info")?;
    Ok(model)
}

/// Resolve the best downloadable file from a HuggingFace repo.
///
/// Prefers (in order): `.safetensors`, `.ckpt`, `.bin`, `.pth`, `.gguf`.
/// Returns the file name, download URL, and size.
pub async fn resolve_download(
    repo_id: &str,
    variant: Option<&str>,
    token: Option<&str>,
) -> Result<HfResolvedFile> {
    let model = get_model(repo_id, token).await?;

    let model_files = find_model_files(&model.siblings);

    if model_files.is_empty() {
        anyhow::bail!(
            "No downloadable model files found in hf:{}\n\
             This repo may use diffusers multi-file format (not yet supported for direct pull).\n\
             Check: https://huggingface.co/{}/tree/main",
            repo_id,
            repo_id,
        );
    }

    // If a variant is specified, try to match it in the filename
    let selected = if let Some(v) = variant {
        let v_lower = v.to_lowercase();
        model_files
            .iter()
            .find(|f| f.filename.to_lowercase().contains(&v_lower))
            .or(model_files.first())
    } else {
        model_files.first()
    };

    let file = selected.unwrap();

    // Fetch actual file size via HEAD request if not available from listing
    let size = if let Some(s) = file.size {
        s
    } else {
        let client = build_client(token)?;
        let url = download_url(repo_id, &file.filename);
        let resp = client.head(&url).send().await.ok();
        resp.and_then(|r| r.content_length()).unwrap_or(0)
    };

    Ok(HfResolvedFile {
        repo_id: repo_id.to_string(),
        filename: file.filename.clone(),
        url: download_url(repo_id, &file.filename),
        size,
    })
}

/// Build the download URL for a file in a HuggingFace repo.
pub fn download_url(repo_id: &str, filename: &str) -> String {
    format!(
        "https://huggingface.co/{}/resolve/main/{}",
        repo_id, filename
    )
}

/// Guess the `AssetType` from HuggingFace metadata.
pub fn guess_asset_type(model: &HfModel, filename: &str) -> String {
    let tags_lower: Vec<String> = model.tags.iter().map(|t| t.to_lowercase()).collect();
    let pipeline = model.pipeline_tag.as_deref().unwrap_or("").to_lowercase();
    let name_lower = model.id.to_lowercase();
    let file_lower = filename.to_lowercase();

    // Check for LoRA indicators
    if tags_lower.iter().any(|t| t.contains("lora"))
        || name_lower.contains("lora")
        || file_lower.contains("lora")
    {
        return "lora".to_string();
    }

    // VAE
    if tags_lower.iter().any(|t| t == "vae")
        || name_lower.contains("vae")
        || file_lower.contains("vae")
    {
        return "vae".to_string();
    }

    // ControlNet
    if tags_lower.iter().any(|t| t.contains("controlnet")) || name_lower.contains("controlnet") {
        return "controlnet".to_string();
    }

    // Text encoder
    if name_lower.contains("text_encoder")
        || name_lower.contains("text-encoder")
        || name_lower.contains("clip")
        || name_lower.contains("t5-xxl")
    {
        return "text_encoder".to_string();
    }

    // Upscaler
    if name_lower.contains("upscal")
        || name_lower.contains("esrgan")
        || name_lower.contains("realesrgan")
    {
        return "upscaler".to_string();
    }

    // IP-Adapter
    if name_lower.contains("ip-adapter") || name_lower.contains("ipadapter") {
        return "ipadapter".to_string();
    }

    // Default to checkpoint for text-to-image
    if pipeline.contains("text-to-image")
        || pipeline.contains("image-to-image")
        || tags_lower.iter().any(|t| t.contains("diffusers"))
    {
        return "checkpoint".to_string();
    }

    "checkpoint".to_string()
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Filter file listing to model-relevant files, sorted by preference.
fn find_model_files(siblings: &[HfSibling]) -> Vec<&HfSibling> {
    const MODEL_EXTENSIONS: &[&str] = &[".safetensors", ".ckpt", ".bin", ".pth", ".pt", ".gguf"];

    // Exclude common non-model files
    const EXCLUDE_PATTERNS: &[&str] = &[
        "optimizer",
        "training_args",
        "scheduler",
        "tokenizer",
        "config.json",
        "model_index",
        "README",
        ".gitattributes",
    ];

    let mut files: Vec<&HfSibling> = siblings
        .iter()
        .filter(|s| {
            let name = &s.filename;
            // Must have a model extension
            MODEL_EXTENSIONS.iter().any(|ext| name.ends_with(ext))
                // Must not be in a subdirectory (multi-file diffusers format)
                && !name.contains('/')
                // Must not match exclude patterns
                && !EXCLUDE_PATTERNS
                    .iter()
                    .any(|p| name.to_lowercase().contains(&p.to_lowercase()))
        })
        .collect();

    // Sort by preference: .safetensors first, then by size descending (primary model = biggest)
    files.sort_by(|a, b| {
        let a_st = a.filename.ends_with(".safetensors");
        let b_st = b.filename.ends_with(".safetensors");
        match (a_st, b_st) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => {
                let a_size = a.size.unwrap_or(0);
                let b_size = b.size.unwrap_or(0);
                b_size.cmp(&a_size)
            }
        }
    });

    files
}

fn build_client(token: Option<&str>) -> Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(15))
        .user_agent("modl-cli");

    if let Some(t) = token {
        builder = builder.default_headers({
            let mut headers = reqwest::header::HeaderMap::new();
            headers.insert(
                reqwest::header::AUTHORIZATION,
                reqwest::header::HeaderValue::from_str(&format!("Bearer {}", t)).unwrap(),
            );
            headers
        });
    }

    builder.build().context("Failed to build HTTP client")
}
