#!/bin/bash
# Round 3: Klein 4B, Klein 9B, Z-Image (all fresh, fixes applied)
export TERM=xterm

MODL="/home/pedro/.local/bin/modl"
DATASET="maxi"
TRIGGER="OHWX"
CLASS="dog"
TYPE="character"
LOG="/home/pedro/overnight-train-r3.log"

echo "=== Round 3 — $(date) ===" | tee "$LOG"

# 1. Klein 4B — 2000 steps (4B is LR-sensitive, code clamps to 5e-5)
echo "[1/3] Klein 4B — $(date)" | tee -a "$LOG"
$MODL train \
  --dataset "$DATASET" \
  --base flux2-klein-4b \
  --name maxi-klein-4b \
  --trigger "$TRIGGER" \
  --class-word "$CLASS" \
  --lora-type "$TYPE" \
  --steps 2000 \
  --rank 16 \
  --preset standard \
  2>&1 | tee -a "$LOG"
echo "[1/3] Klein 4B done — $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# 2. Klein 9B — 2000 steps
echo "[2/3] Klein 9B — $(date)" | tee -a "$LOG"
$MODL train \
  --dataset "$DATASET" \
  --base flux2-klein-9b \
  --name maxi-klein-9b \
  --trigger "$TRIGGER" \
  --class-word "$CLASS" \
  --lora-type "$TYPE" \
  --steps 2000 \
  --rank 16 \
  --preset standard \
  2>&1 | tee -a "$LOG"
echo "[2/3] Klein 9B done — $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# 3. Z-Image — 3000 steps fresh
echo "[3/3] Z-Image — $(date)" | tee -a "$LOG"
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
  2>&1 | tee -a "$LOG"
echo "[3/3] Z-Image done — $(date)" | tee -a "$LOG"

echo "=== Round 3 complete! $(date) ===" | tee -a "$LOG"
