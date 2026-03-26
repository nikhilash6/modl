#!/usr/bin/env python3
"""Build annotated training progression grids for the character LoRA guide."""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

TRAINING_OUTPUT = Path.home() / ".modl" / "training_output"
OUT = Path.home() / "projects" / "modl-org" / "modl-site" / "public" / "guides" / "character-lora"

# Try to load a nice font, fall back to default
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    FONT_SM = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    FONT_LABEL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
except:
    FONT = ImageFont.load_default()
    FONT_SM = FONT
    FONT_LABEL = FONT

CELL = 380  # each image cell size
PAD = 4
LABEL_H = 36  # height for column/row labels
BG = (15, 15, 20)
LABEL_COLOR = (200, 200, 210)
ACCENT = (139, 92, 246)


def find_sample(run: str, step: int, prompt: int) -> Path | None:
    """Find a sample image by run name, step, and prompt index."""
    samples_dir = TRAINING_OUTPUT / run / run / "samples"
    if not samples_dir.exists():
        return None
    step_str = f"{step:09d}"
    for f in samples_dir.iterdir():
        if f.name.endswith(f"__{step_str}_{prompt}.jpg"):
            return f
    return None


def load_and_resize(path: Path | None, size: int = CELL) -> Image.Image:
    """Load image and resize to square, or return placeholder."""
    if path and path.exists():
        img = Image.open(path).convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        return img
    # Placeholder
    img = Image.new("RGB", (size, size), (30, 30, 40))
    draw = ImageDraw.Draw(img)
    draw.text((size // 2 - 20, size // 2 - 8), "N/A", fill=(80, 80, 80), font=FONT_SM)
    return img


def build_grid(
    rows: list[tuple[str, list]],  # [(row_label, [(run, step, prompt), ...]), ...]
    col_labels: list[str],
    title: str,
    out_name: str,
):
    """Build an annotated grid with row/column labels."""
    n_cols = len(col_labels)
    n_rows = len(rows)

    W = LABEL_H + n_cols * (CELL + PAD) + PAD
    H = LABEL_H + 40 + n_rows * (CELL + PAD) + PAD  # 40 for title

    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    # Title
    draw.text((LABEL_H + PAD, 8), title, fill=ACCENT, font=FONT_LABEL)

    # Column labels
    for i, label in enumerate(col_labels):
        x = LABEL_H + PAD + i * (CELL + PAD) + CELL // 2
        draw.text((x, 40), label, fill=LABEL_COLOR, font=FONT, anchor="mt")

    # Rows
    for r, (row_label, cells) in enumerate(rows):
        y = LABEL_H + 40 + PAD + r * (CELL + PAD)

        # Row label (rotated)
        txt_img = Image.new("RGB", (CELL, LABEL_H), BG)
        txt_draw = ImageDraw.Draw(txt_img)
        txt_draw.text((CELL // 2, LABEL_H // 2), row_label, fill=LABEL_COLOR, font=FONT_SM, anchor="mm")
        txt_img = txt_img.rotate(90, expand=True)
        canvas.paste(txt_img, (0, y))

        # Cells
        for c, (run, step, prompt) in enumerate(cells):
            x = LABEL_H + PAD + c * (CELL + PAD)
            path = find_sample(run, step, prompt)
            img = load_and_resize(path)
            canvas.paste(img, (x, y))

    canvas.save(OUT / out_name, quality=92)
    print(f"  {out_name}: {W}x{H}")


print("Building guide grids...\n")

# -----------------------------------------------------------------------
# GRID 1: Klein 4B r32 pomeranian — identity emerging
# -----------------------------------------------------------------------
build_grid(
    rows=[
        ("Identity", [
            ("maxi-klein-4b-r32-pom", 0, 0),
            ("maxi-klein-4b-r32-pom", 600, 0),
            ("maxi-klein-4b-r32-pom", 1200, 0),
            ("maxi-klein-4b-r32-pom", 2000, 0),
        ]),
        ("Close-up", [
            ("maxi-klein-4b-r32-pom", 0, 3),
            ("maxi-klein-4b-r32-pom", 600, 3),
            ("maxi-klein-4b-r32-pom", 1200, 3),
            ("maxi-klein-4b-r32-pom", 2000, 3),
        ]),
        ("Beach", [
            ("maxi-klein-4b-r32-pom", 0, 5),
            ("maxi-klein-4b-r32-pom", 600, 5),
            ("maxi-klein-4b-r32-pom", 1200, 5),
            ("maxi-klein-4b-r32-pom", 2000, 5),
        ]),
    ],
    col_labels=["Step 0", "Step 600", "Step 1200", "Step 2000"],
    title="Klein 4B — Identity Emerging (rank 32, pomeranian)",
    out_name="grid-klein4b-progression.jpg",
)

# -----------------------------------------------------------------------
# GRID 2: Z-Image adamw vs prodigy
# -----------------------------------------------------------------------
build_grid(
    rows=[
        ("adamw", [
            ("maxi-zimage", 0, 0),
            ("maxi-zimage", 600, 0),
            ("maxi-zimage", 1200, 0),
            ("maxi-zimage", 1800, 0),
        ]),
        ("prodigy", [
            ("maxi-zimage-v2", 0, 0),
            ("maxi-zimage-v2", 600, 0),
            ("maxi-zimage-v2", 1200, 0),
            ("maxi-zimage-v2", 1800, 0),
        ]),
    ],
    col_labels=["Step 0", "Step 600", "Step 1200", "Step 1800"],
    title="Z-Image Base — adamw8bit vs prodigy (same dataset)",
    out_name="grid-adamw-vs-prodigy.jpg",
)

# -----------------------------------------------------------------------
# GRID 3: Model comparison — 5 models, 3 prompts at final step
# -----------------------------------------------------------------------
models = [
    ("SDXL", "maxi-sdxl", 1500),
    ("Klein 4B", "maxi-klein-4b-r32-pom", 2000),
    ("Klein 9B", "maxi-klein-9b", 2000),
    ("Schnell", "maxi-schnell", 1500),
    ("Z-Image", "maxi-zimage-v2", 3000),
]

build_grid(
    rows=[
        ("Identity", [(run, step, 0) for _, run, step in models]),
        ("Close-up", [(run, step, 3) for _, run, step in models]),
        ("Beach", [(run, step, 5) for _, run, step in models]),
    ],
    col_labels=[name for name, _, _ in models],
    title="Model Comparison — Same Dataset, Best Checkpoint",
    out_name="grid-model-comparison.jpg",
)

# -----------------------------------------------------------------------
# GRID 4: Class word — dog vs pomeranian
# -----------------------------------------------------------------------
build_grid(
    rows=[
        ("r16 + dog", [
            ("maxi-klein-4b", 0, 0),
            ("maxi-klein-4b", 600, 0),
            ("maxi-klein-4b", 1200, 0),
            ("maxi-klein-4b", 2000, 0),
        ]),
        ("r32 + pom", [
            ("maxi-klein-4b-r32-pom", 0, 0),
            ("maxi-klein-4b-r32-pom", 600, 0),
            ("maxi-klein-4b-r32-pom", 1200, 0),
            ("maxi-klein-4b-r32-pom", 2000, 0),
        ]),
    ],
    col_labels=["Step 0", "Step 600", "Step 1200", "Step 2000"],
    title="Class Word Impact — 'dog' vs 'pomeranian' (Klein 4B)",
    out_name="grid-classword.jpg",
)

# -----------------------------------------------------------------------
# GRID 5: Bleed check — Z-Image v2 at step 0 vs 3000
# -----------------------------------------------------------------------
build_grid(
    rows=[
        ("Step 0", [
            ("maxi-zimage-v2", 0, 0),
            ("maxi-zimage-v2", 0, 9),
            ("maxi-zimage-v2", 0, 10),
        ]),
        ("Step 3000", [
            ("maxi-zimage-v2", 3000, 0),
            ("maxi-zimage-v2", 3000, 9),
            ("maxi-zimage-v2", 3000, 10),
        ]),
    ],
    col_labels=["OHWX pomeranian", "Woman (no trigger)", "Retriever (no trigger)"],
    title="Bleed Check — Does the LoRA Contaminate Other Subjects?",
    out_name="grid-bleed-check.jpg",
)

print("\nAll grids built!")
