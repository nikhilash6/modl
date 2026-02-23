pub mod layouts;

use std::path::{Path, PathBuf};

use crate::core::config::ToolType;
use crate::core::manifest::AssetType;

/// Get the subfolder path for a given asset type within a tool's directory
pub fn asset_folder(tool_type: &ToolType, asset_type: &AssetType) -> PathBuf {
    match tool_type {
        ToolType::Comfyui => layouts::comfyui(asset_type),
        ToolType::A1111 => layouts::a1111(asset_type),
        ToolType::Invokeai => layouts::invokeai(asset_type),
        ToolType::Custom => PathBuf::from("models"),
    }
}

/// Build the full symlink path for a model file within a tool installation
pub fn symlink_path(
    tool_root: &Path,
    tool_type: &ToolType,
    asset_type: &AssetType,
    file_name: &str,
) -> PathBuf {
    tool_root
        .join(asset_folder(tool_type, asset_type))
        .join(file_name)
}

/// Auto-detect installed tools by checking common locations.
///
/// On Windows, also scans drive roots (C:\, D:\) for ComfyUI Portable
/// installations, which is the most common setup.
pub fn detect_tools() -> Vec<(ToolType, PathBuf)> {
    let mut found = Vec::new();
    let home = match dirs::home_dir() {
        Some(h) => h,
        None => return found,
    };

    // Common ComfyUI locations (relative to home)
    let comfyui_dirs = vec![
        "ComfyUI",
        "comfyui",
        ".comfyui",
        "ai-lab/ComfyUI",
    ];

    for dir in &comfyui_dirs {
        let path = home.join(dir);
        if path.join("models").exists() {
            found.push((ToolType::Comfyui, path));
            break;
        }
    }

    // Windows: scan drive roots for ComfyUI Portable
    #[cfg(windows)]
    {
        if !found.iter().any(|(t, _)| matches!(t, ToolType::Comfyui)) {
            for drive in &["C:", "D:", "E:", "F:"] {
                let portable_names = [
                    "ComfyUI_windows_portable",
                    "ComfyUI-portable",
                    "ComfyUI",
                ];
                for name in &portable_names {
                    let path = PathBuf::from(format!("{}\\{}", drive, name));
                    // Portable has ComfyUI subfolder inside the portable dir
                    let inner = path.join("ComfyUI");
                    if inner.join("models").exists() {
                        found.push((ToolType::Comfyui, inner));
                        break;
                    }
                    // Or the root itself may have models/
                    if path.join("models").exists() {
                        found.push((ToolType::Comfyui, path));
                        break;
                    }
                }
                if found.iter().any(|(t, _)| matches!(t, ToolType::Comfyui)) {
                    break;
                }
            }
        }
    }

    // Common A1111 locations
    for dir in &["stable-diffusion-webui", "sd-webui", "automatic1111"] {
        let path = home.join(dir);
        if path.join("models").exists() {
            found.push((ToolType::A1111, path));
            break;
        }
    }

    found
}

/// Check if Windows Developer Mode is enabled (for symlinks without admin).
/// Returns None on non-Windows platforms.
#[allow(dead_code)]
pub fn check_windows_dev_mode() -> Option<bool> {
    #[cfg(windows)]
    {
        // Check via registry: HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock
        // DeveloperModeEnabled = 1 means enabled
        use std::process::Command;
        let output = Command::new("reg")
            .args([
                "query",
                r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock",
                "/v",
                "AllowDevelopmentWithoutDevLicense",
            ])
            .output();
        match output {
            Ok(o) => {
                let stdout = String::from_utf8_lossy(&o.stdout);
                Some(stdout.contains("0x1"))
            }
            Err(_) => Some(false),
        }
    }

    #[cfg(not(windows))]
    {
        None
    }
}
