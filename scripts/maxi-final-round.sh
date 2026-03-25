#!/bin/bash
# Final round: Klein 4B r32 + Z-Image, both with --class-word pomeranian
export TERM=xterm

MODL="/home/pedro/.local/bin/modl"
LOG="/home/pedro/maxi-final-round.log"

echo "=== Final round — $(date) ===" | tee "$LOG"

# 1. Klein 4B rank 32 with pomeranian
echo "[1/2] Klein 4B r32 pomeranian — $(date)" | tee -a "$LOG"
$MODL train \
  --dataset maxi \
  --base flux2-klein-4b \
  --name maxi-klein-4b-r32-pom \
  --trigger OHWX \
  --class-word pomeranian \
  --lora-type character \
  --steps 2000 \
  --rank 32 \
  --preset standard \
  2>&1 | tee -a "$LOG"
echo "[1/2] Klein 4B r32 pomeranian done — $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# 2. Z-Image with prodigy (auto from branch defaults) + pomeranian
echo "[2/2] Z-Image prodigy pomeranian — $(date)" | tee -a "$LOG"
$MODL train \
  --dataset maxi \
  --base z-image \
  --name maxi-zimage-v2 \
  --trigger OHWX \
  --class-word pomeranian \
  --lora-type character \
  --steps 3000 \
  --rank 16 \
  --preset standard \
  2>&1 | tee -a "$LOG"
echo "[2/2] Z-Image done — $(date)" | tee -a "$LOG"

echo "=== Final round complete! $(date) ===" | tee -a "$LOG"
