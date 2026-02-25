use anyhow::Result;
use console::style;

use crate::core::registry::RegistryIndex;

pub async fn run() -> Result<()> {
    println!("{} Fetching latest registry index...", style("→").cyan());

    let index = RegistryIndex::fetch_and_save().await?;

    println!(
        "{} Registry updated — {} models available",
        style("✓").green(),
        style(index.items.len()).bold()
    );

    Ok(())
}
