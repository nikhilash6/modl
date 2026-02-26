use anyhow::Result;
use console::style;
use std::path::PathBuf;

use crate::core::training::{self, TrainOptions};

pub async fn run(
    dataset: &str,
    base: &str,
    name: &str,
    trigger: &str,
    steps: u32,
    dry_run: bool,
) -> Result<()> {
    let options = TrainOptions {
        dataset: PathBuf::from(dataset),
        base: base.to_string(),
        name: name.to_string(),
        trigger_word: trigger.to_string(),
        steps,
        dry_run,
    };

    println!(
        "{} Starting training bridge ({})",
        style("→").cyan(),
        style("runtime: trainer-cu124").dim()
    );

    let result = training::run(options).await?;

    if dry_run {
        println!("{} Dry run complete", style("✓").green().bold());
        println!("  Config: {}", result.config_path.display());
        println!("  Python: {}", result.python_path.display());
        println!();
        println!(
            "{} Remove --dry-run to execute the Python training proxy.",
            style("i").dim()
        );
    } else {
        println!("{} Training bridge run complete", style("✓").green().bold());
        println!("  Config: {}", result.config_path.display());
        println!("  Python: {}", result.python_path.display());
    }

    Ok(())
}
