#!/bin/bash
# Overnight LoRA training — maxi (dog) character across 5 models
# RTX 4090 24GB — sequential runs, each ~20-60 min
# Started: $(date)
# Don't set -e — if one model fails (e.g. OOM), continue to next
export TERM=xterm

MODL="/home/pedro/.local/bin/modl"
DATASET="maxi"
TRIGGER="OHWX"
CLASS="dog"
TYPE="character"
LOG="/home/pedro/overnight-train.log"

echo "=== Overnight Maxi LoRA Training ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# 1. SDXL — fastest, safest baseline (~7GB VRAM, ~15 min)
echo "[1/5] SDXL — $(date)" | tee -a "$LOG"
$MODL train \
  --dataset "$DATASET" \
  --base sdxl-base-1.0 \
  --name maxi-sdxl \
  --trigger "$TRIGGER" \
  --class-word "$CLASS" \
  --lora-type "$TYPE" \
  --steps 1500 \
  --rank 16 \
  --lr 1e-4 \
  --preset standard \
  2>&1 | tee -a "$LOG"
echo "[1/5] SDXL done — $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# 2. Flux 2 Klein 4B — small, fast (~14GB VRAM, ~20 min)
echo "[2/5] Klein 4B — $(date)" | tee -a "$LOG"
$MODL train \
  --dataset "$DATASET" \
  --base flux2-klein-4b \
  --name maxi-klein-4b \
  --trigger "$TRIGGER" \
  --class-word "$CLASS" \
  --lora-type "$TYPE" \
  --steps 1500 \
  --rank 16 \
  --lr 4e-4 \
  --preset standard \
  2>&1 | tee -a "$LOG"
echo "[2/5] Klein 4B done — $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# 3. Flux 2 Klein 9B — tight on 24GB (~24GB VRAM, ~30 min)
echo "[3/5] Klein 9B — $(date)" | tee -a "$LOG"
$MODL train \
  --dataset "$DATASET" \
  --base flux2-klein-9b \
  --name maxi-klein-9b \
  --trigger "$TRIGGER" \
  --class-word "$CLASS" \
  --lora-type "$TYPE" \
  --steps 1500 \
  --rank 16 \
  --lr 4e-4 \
  --preset standard \
  2>&1 | tee -a "$LOG"
echo "[3/5] Klein 9B done — $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# 4. Flux Schnell — distilled 4-step model (~20GB FP8, ~30 min)
echo "[4/5] Flux Schnell — $(date)" | tee -a "$LOG"
$MODL train \
  --dataset "$DATASET" \
  --base flux-schnell \
  --name maxi-schnell \
  --trigger "$TRIGGER" \
  --class-word "$CLASS" \
  --lora-type "$TYPE" \
  --steps 1500 \
  --rank 16 \
  --lr 4e-4 \
  --preset standard \
  2>&1 | tee -a "$LOG"
echo "[4/5] Flux Schnell done — $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# 5. Z-Image (base, not turbo) — ~20GB quantized (~40 min)
echo "[5/5] Z-Image — $(date)" | tee -a "$LOG"
$MODL train \
  --dataset "$DATASET" \
  --base z-image \
  --name maxi-zimage \
  --trigger "$TRIGGER" \
  --class-word "$CLASS" \
  --lora-type "$TYPE" \
  --steps 1500 \
  --rank 16 \
  --lr 1e-4 \
  --preset standard \
  2>&1 | tee -a "$LOG"
echo "[5/5] Z-Image done — $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

echo "=== All done! ===" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
echo "Results: modl train ls" | tee -a "$LOG"
