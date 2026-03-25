#!/bin/bash
# Round 2: Klein 4B, Klein 9B, Z-Image resume
export TERM=xterm

MODL="/home/pedro/.local/bin/modl"
DATASET="maxi"
TRIGGER="OHWX"
CLASS="dog"
TYPE="character"
LOG="/home/pedro/overnight-train-r2.log"

echo "=== Round 2 — $(date) ===" | tee "$LOG"

# 1. Klein 4B — 1500 steps
echo "[1/3] Klein 4B — $(date)" | tee -a "$LOG"
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
echo "[1/3] Klein 4B done — $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# 2. Klein 9B — 1500 steps
echo "[2/3] Klein 9B — $(date)" | tee -a "$LOG"
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
echo "[2/3] Klein 9B done — $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# 3. Z-Image resume — 1500 more steps (total 3000)
echo "[3/3] Z-Image resume to 3000 — $(date)" | tee -a "$LOG"
CHECKPOINT=$(ls -t ~/.modl/training_output/maxi-zimage/maxi-zimage/maxi-zimage.safetensors 2>/dev/null | head -1)
$MODL train \
  --dataset "$DATASET" \
  --base z-image \
  --name maxi-zimage \
  --trigger "$TRIGGER" \
  --class-word "$CLASS" \
  --lora-type "$TYPE" \
  --steps 3000 \
  --rank 16 \
  --lr 1e-4 \
  --preset standard \
  --resume "$CHECKPOINT" \
  2>&1 | tee -a "$LOG"
echo "[3/3] Z-Image resume done — $(date)" | tee -a "$LOG"

echo "=== Round 2 complete! $(date) ===" | tee -a "$LOG"
