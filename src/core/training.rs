use anyhow::{Context, Result, bail};
use serde::Serialize;
use std::env;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::core::runtime;

#[derive(Debug, Clone)]
pub struct TrainOptions {
    pub dataset: PathBuf,
    pub base: String,
    pub name: String,
    pub trigger_word: String,
    pub steps: u32,
    pub dry_run: bool,
}

#[derive(Debug, Clone)]
pub struct TrainResult {
    pub config_path: PathBuf,
    pub python_path: PathBuf,
}

#[derive(Debug, Serialize)]
struct TrainConfig {
    schema_version: String,
    name: String,
    base_model: String,
    dataset_path: String,
    trigger_word: String,
    steps: u32,
    created_at: String,
}

pub async fn run(options: TrainOptions) -> Result<TrainResult> {
    if !options.dataset.exists() {
        bail!(
            "Dataset path does not exist: {}",
            options.dataset.to_string_lossy()
        );
    }

    if !options.dataset.is_dir() {
        bail!(
            "Dataset path must be a directory: {}",
            options.dataset.to_string_lossy()
        );
    }

    let runtime_root = if options.dry_run {
        let install = runtime::install(Some("trainer-cu124"), None)?;
        install.runtime_root
    } else {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join(".mods")
            .join("runtime")
    };

    let jobs_dir = runtime_root.join("jobs");
    fs::create_dir_all(&jobs_dir)
        .with_context(|| format!("Failed to create {}", jobs_dir.display()))?;

    let timestamp = chrono::Utc::now().format("%Y%m%d-%H%M%S");
    let config_path = jobs_dir.join(format!("train-{}-{}.yaml", options.name, timestamp));

    let config = TrainConfig {
        schema_version: "v1".to_string(),
        name: options.name.clone(),
        base_model: options.base.clone(),
        dataset_path: options.dataset.to_string_lossy().to_string(),
        trigger_word: options.trigger_word.clone(),
        steps: options.steps,
        created_at: chrono::Utc::now().to_rfc3339(),
    };

    let yaml = serde_yaml::to_string(&config).context("Failed to serialize training config")?;
    fs::write(&config_path, yaml)
        .with_context(|| format!("Failed to write {}", config_path.display()))?;

    if options.dry_run {
        return Ok(TrainResult {
            config_path,
            python_path: runtime_root
                .join("envs")
                .join("trainer-cu124")
                .join("bin")
                .join("python"),
        });
    }

    let setup = runtime::setup_training(false).await?;
    if !setup.ready {
        bail!(
            "Training runtime setup finished but ai-toolkit command could not be auto-detected. Run `mods train-setup --reinstall` or set MODS_AITOOLKIT_TRAIN_CMD."
        );
    }

    run_python_training_proxy(setup.python_path.as_path(), config_path.as_path())?;

    Ok(TrainResult {
        config_path,
        python_path: setup.python_path,
    })
}

fn run_python_training_proxy(python_path: &Path, config_path: &Path) -> Result<()> {
    let worker_python_root = resolve_worker_python_root()?;
    let py_path = match env::var("PYTHONPATH") {
        Ok(current) if !current.trim().is_empty() => {
            format!("{}:{}", worker_python_root.display(), current)
        }
        _ => worker_python_root.to_string_lossy().to_string(),
    };

    let mut command = Command::new(python_path);
    command
        .arg("-m")
        .arg("mods_worker.main")
        .arg("train")
        .arg("--config")
        .arg(config_path)
        .env("PYTHONPATH", py_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());

    if let Ok(Some(template)) = runtime::train_command_template() {
        command.env("MODS_AITOOLKIT_TRAIN_CMD", template);
    }

    let mut child = command.spawn().with_context(|| {
        format!(
            "Failed to start training proxy using {}",
            python_path.to_string_lossy()
        )
    })?;

    let stdout = child
        .stdout
        .take()
        .context("Failed to capture training proxy stdout")?;
    let reader = BufReader::new(stdout);

    for line in reader.lines() {
        let line = line.context("Failed reading training proxy output")?;
        if line.trim().is_empty() {
            continue;
        }

        match serde_json::from_str::<serde_json::Value>(&line) {
            Ok(event) => {
                let kind = event
                    .get("type")
                    .and_then(|v| v.as_str())
                    .or_else(|| {
                        event
                            .get("event")
                            .and_then(|e| e.get("type"))
                            .and_then(|v| v.as_str())
                    })
                    .unwrap_or("event");
                println!("[train:{}] {}", kind, line);
            }
            Err(_) => {
                println!("[train:raw] {}", line);
            }
        }
    }

    let status = child
        .wait()
        .context("Failed waiting on training proxy process")?;

    if !status.success() {
        bail!("Training proxy failed with status: {}", status);
    }

    Ok(())
}

fn resolve_worker_python_root() -> Result<PathBuf> {
    if let Ok(custom) = env::var("MODS_WORKER_PYTHON_ROOT") {
        let path = PathBuf::from(custom);
        if path.exists() {
            return Ok(path);
        }
        bail!(
            "MODS_WORKER_PYTHON_ROOT points to missing path: {}",
            path.display()
        );
    }

    let default_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python");
    if default_path.exists() {
        Ok(default_path)
    } else {
        bail!(
            "Worker python package not found at {}. Set MODS_WORKER_PYTHON_ROOT to a valid path.",
            default_path.display()
        )
    }
}
