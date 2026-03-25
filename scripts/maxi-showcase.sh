#!/bin/bash
# Generate 8 showcase images with the best Maxi LoRA
# Run after final training round — picks the best LoRA automatically

MODL="/home/pedro/.local/bin/modl"
OUTDIR="$HOME/maxi-showcase"
mkdir -p "$OUTDIR"

# Pick the best LoRA (pomeranian r32 if available, otherwise r32, otherwise r16)
if [ -f "$HOME/.modl/loras/maxi-klein-4b-r32-pom.safetensors" ]; then
  LORA="maxi-klein-4b-r32-pom"
elif [ -f "$HOME/.modl/loras/maxi-klein-4b-r32.safetensors" ]; then
  LORA="maxi-klein-4b-r32"
else
  LORA="maxi-klein-4b"
fi

echo "Using LoRA: $LORA"
echo "Output: $OUTDIR"

PROMPTS=(
  "OHWX pomeranian floating in outer space wearing a tiny astronaut helmet, stars and nebula in the background, cinematic lighting"
  "OHWX pomeranian sitting on a throne made of dog treats, wearing a tiny golden crown, dramatic royal portrait"
  "OHWX pomeranian surfing a giant wave at sunset, action shot, water splashing"
  "OHWX pomeranian as a renaissance painting, oil on canvas, ornate golden frame, museum lighting"
  "OHWX pomeranian in a cozy coffee shop, sitting on a velvet chair, warm morning light through the window"
  "OHWX pomeranian running through a field of sunflowers, golden hour, motion blur on the flowers"
  "OHWX pomeranian wearing a detective hat and magnifying glass, noir style, foggy street at night"
  "OHWX pomeranian in a japanese garden with cherry blossoms falling, serene, soft pink light"
)

for i in "${!PROMPTS[@]}"; do
  echo "[$((i+1))/8] Generating..."
  $MODL generate "${PROMPTS[$i]}" \
    --lora "$LORA" \
    --output "$OUTDIR/maxi_showcase_$((i+1)).jpg" \
    2>&1 | tail -1
done

echo "=== Done! 8 images in $OUTDIR ==="
