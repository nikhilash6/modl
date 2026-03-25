#!/bin/bash
# Klein 4B rank 32 — after Klein 9B finishes
export TERM=xterm
/home/pedro/.local/bin/modl train \
  --dataset maxi \
  --base flux2-klein-4b \
  --name maxi-klein-4b-r32 \
  --trigger OHWX \
  --class-word dog \
  --lora-type character \
  --steps 2000 \
  --rank 32 \
  --preset standard \
  2>&1 | tee /home/pedro/klein4b-r32-train.log
