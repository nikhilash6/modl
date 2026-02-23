use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::PathBuf;

use super::manifest::Manifest;

/// The registry index — compiled from all manifests
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct RegistryIndex {
    #[serde(default)]
    pub version: u32,
    pub items: Vec<Manifest>,
}

impl RegistryIndex {
    /// Load the local copy of the registry index
    pub fn load() -> Result<Self> {
        let path = Self::local_path();
        if !path.exists() {
            anyhow::bail!("Registry index not found. Run `mods update` to fetch it.");
        }
        let contents = std::fs::read_to_string(&path).context("Failed to read registry index")?;
        let index: RegistryIndex =
            serde_json::from_str(&contents).context("Failed to parse registry index")?;
        Ok(index)
    }

    /// Save index to local cache
    #[allow(dead_code)]
    pub fn save(&self) -> Result<()> {
        let path = Self::local_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json)?;
        Ok(())
    }

    pub fn local_path() -> PathBuf {
        dirs::home_dir()
            .expect("Could not determine home directory")
            .join(".mods")
            .join("index.json")
    }

    /// Look up a manifest by ID
    pub fn find(&self, id: &str) -> Option<&Manifest> {
        self.items.iter().find(|m| m.id == id)
    }

    /// Search items by query string (matches id, name, tags, description)
    pub fn search(&self, query: &str) -> Vec<&Manifest> {
        let q = query.to_lowercase();
        self.items
            .iter()
            .filter(|m| {
                m.id.to_lowercase().contains(&q)
                    || m.name.to_lowercase().contains(&q)
                    || m.tags.iter().any(|t| t.to_lowercase().contains(&q))
                    || m.description
                        .as_ref()
                        .is_some_and(|d| d.to_lowercase().contains(&q))
            })
            .collect()
    }

    /// Build a lookup map by ID for faster access
    #[allow(dead_code)]
    pub fn as_map(&self) -> HashMap<&str, &Manifest> {
        self.items.iter().map(|m| (m.id.as_str(), m)).collect()
    }

    /// The URL to fetch the latest index from.
    ///
    /// Uses registry.mods.sh (Cloudflare-cached) as primary,
    /// with raw GitHub as fallback. Override via MODS_REGISTRY_URL env var.
    pub fn remote_url() -> String {
        std::env::var("MODS_REGISTRY_URL").unwrap_or_else(|_| {
            "https://registry.mods.sh/index.json".to_string()
        })
    }

    /// Fallback URL if the primary registry is unreachable
    pub fn fallback_url() -> &'static str {
        "https://raw.githubusercontent.com/modshq-org/mods-registry/main/index.json"
    }
}
