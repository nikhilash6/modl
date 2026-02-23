use anyhow::{Context, Result};
use console::style;
use indicatif::{ProgressBar, ProgressStyle};

use crate::core::registry::RegistryIndex;

pub async fn run() -> Result<()> {
    println!("{} Fetching latest registry index...", style("→").cyan());

    let primary_url = RegistryIndex::remote_url();
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner} {msg}")
            .unwrap(),
    );
    pb.set_message("Downloading index.json");
    pb.enable_steady_tick(std::time::Duration::from_millis(80));

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .unwrap_or_default();

    // Try primary URL, fall back to GitHub raw
    let body = match fetch_index(&client, &primary_url).await {
        Ok(body) => body,
        Err(e) => {
            let fallback = RegistryIndex::fallback_url();
            pb.suspend(|| {
                eprintln!(
                    "  {} Primary registry unavailable ({}), trying fallback...",
                    style("⚠").yellow(),
                    e
                );
            });
            fetch_index(&client, fallback)
                .await
                .context("Failed to fetch registry from both primary and fallback URLs")?
        }
    };

    let index: RegistryIndex =
        serde_json::from_str(&body).context("Failed to parse registry index")?;

    // Save to local cache
    let path = RegistryIndex::local_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, &body).context("Failed to write index to disk")?;

    pb.finish_and_clear();

    println!(
        "{} Registry updated — {} models available",
        style("✓").green(),
        style(index.items.len()).bold()
    );

    Ok(())
}

/// Fetch the index.json from a URL, returning the body as a string.
async fn fetch_index(client: &reqwest::Client, url: &str) -> Result<String> {
    let response = client
        .get(url)
        .send()
        .await
        .with_context(|| format!("Failed to connect to {}", url))?;

    if !response.status().is_success() {
        anyhow::bail!("HTTP {} from {}", response.status(), url);
    }

    response
        .text()
        .await
        .with_context(|| format!("Failed to read response from {}", url))
}
