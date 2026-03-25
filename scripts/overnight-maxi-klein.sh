#!/bin/bash
# Round 3: Klein 4B and Klein 9B (after config_builder fix)
export TERM=xterm

MODL="/home/pedro/.local/bin/modl"
DATASET="maxi"
TRIGGER="OHWX"
CLASS="dog"
TYPE="character"
LOG="/home/pedro/overnight-train-klein.log"

echo "=== Klein training — $(date) ===" | tee "$LOG"

# 1. Klein 4B — 2000 steps (Klein needs more steps per the code comments)
echo "[1/2] Klein 4B — $(date)" | tee -a "$LOG"
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
echo "[1/2] Klein 4B done — $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# 2. Klein 9B — 2000 steps
echo "[2/2] Klein 9B — $(date)" | tee -a "$LOG"
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
echo "[2/2] Klein 9B done — $(date)" | tee -a "$LOG"

echo "=== Klein complete! $(date) ===" | tee -a "$LOG"
