#!/bin/bash
set -e

echo "[modl-gpu] Starting GPU agent..."

# torch + diffusers are pre-installed in the image. No bootstrap needed.

# Pre-pull models if MODL_MODELS is set (comma-separated)
if [ -n "$MODL_MODELS" ]; then
    IFS=',' read -ra MODELS <<< "$MODL_MODELS"
    for model in "${MODELS[@]}"; do
        model=$(echo "$model" | xargs)
        if [ -n "$model" ]; then
            echo "[modl-gpu] Pulling model: $model"
            modl pull "$model" || echo "[modl-gpu] Warning: failed to pull $model"
        fi
    done
fi

# Start the agent — polls orchestrator for jobs, runs them locally
exec modl gpu agent \
    --session-token "${MODL_SESSION_TOKEN}" \
    --api-base "${MODL_API_BASE:-https://hub.modl.run}"
