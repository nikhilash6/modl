# mods: The Opinionated Image Generation Toolkit

1. CLI-first (like rails/cargo/git)
2. Opinionated defaults with escape hatches
3. Covers the *full lifecycle*: models → datasets → training → inference → outputs
4. Local-first, cloud-burst capable
5. Actually maintained for modern models (Flux, Z-Image, Qwen, etc.)

## The MVP: What mods Actually Is

### Philosophy: CLI is the truth, UI is a window

```
mods                          # shows status: models, running jobs, recent generations
mods models                   # list all local models (base + LoRAs)
mods models pull flux-dev     # download Flux.1-dev (like ollama pull)
mods models pull civitai:123  # pull from CivitAI by ID
mods models ls                # list with size, type, base model

mods train                    # interactive: picks dataset, model, writes config, starts
mods train --dataset ./products --base flux-dev --name product-lora-v1
mods train --config custom.yml  # escape hatch: full ai-toolkit YAML

mods generate "a photo of OHWX on marble countertop"  # uses last trained LoRA
mods generate "a photo of OHWX" --lora product-lora-v1 --seed 42
mods generate "a cat" --base flux-schnell              # no LoRA, base model

mods datasets                  # list datasets
mods datasets create ./photos  # creates dataset dir with proper structure
mods datasets caption ./photos # auto-caption with Florence/BLIP

mods serve                     # starts local web UI on :3000
mods deploy                    # deploys to Modal (BYOK)
```

### The Three Layers

```
Layer 1: mods CLI (Rust)
├── Model manager (download, organize, track)
├── Config generator (simple flags → ai-toolkit YAML)
├── Process orchestrator (spawn training, inference)
└── Output organizer (SQLite DAM)

Layer 2: mods Python runtime (managed by CLI)
├── ai-toolkit (training, managed as dependency)
├── diffusers (inference)
└── Auto-captioning (Florence-2 / BLIP)

Layer 3: mods web UI (optional, `mods serve`)
├── Next.js + shadcn/ui
├── Reads from same SQLite + filesystem
└── Generation playground
```

### Why Rust CLI + Python runtime?

- Rust CLI: fast startup, single binary distribution, model file management is I/O-bound (perfect for Rust), cross-platform
- Python runtime: ai-toolkit and diffusers are Python. No point fighting this.
- The CLI orchestrates Python processes. Like how `cargo` doesn't compile Rust itself — it calls `rustc`.

---

## MVP Task List for Claude Code

### Sprint 1: Model Manager (the `mods` you already started)

```
[ ] mods models pull <model>
    - Registry of known models (flux-dev, flux-schnell, sdxl, z-image-turbo)
    - Downloads from HF with progress bar
    - Stores in ~/.mods/models/ with metadata
    - Handles auth (HF token in ~/.mods/config.toml)

[ ] mods models ls
    - Shows: name, type (base/lora), size, base_model, local path
    - Groups: base models vs LoRAs

[ ] mods models link <path>
    - Symlink existing models (don't force re-download)
    - Scan ComfyUI/A1111 model directories

[ ] ~/.mods/config.toml
    - models_dir, outputs_dir, datasets_dir
    - hf_token
    - default_base_model
    - gpu (auto-detect or manual)
```

### Sprint 2: Training (`mods train`)

```
[ ] mods datasets create <name> --from <dir>
    - Copies/symlinks images to ~/.mods/datasets/<name>/
    - Validates file formats (jpg/png only, no webp)
    - Pairs images with .txt files
    - Reports: "12 images, 10 captioned, 2 missing captions"

[ ] mods datasets caption <name>
    - Runs Florence-2 or BLIP on uncaptioned images
    - Writes .txt files alongside images
    - Shows captions for review in terminal

[ ] mods train (interactive mode)
    Prompt-driven:
    > Base model? [flux-dev] ↵
    > Dataset? products ↵
    > Trigger word? OHWX ↵
    > Name this LoRA? product-v1 ↵
    > Quick/Standard/Advanced? [Standard] ↵
    
    Presets:
    - Quick: 1000 steps, rank 8, ~20 min on 4090
    - Standard: 2000 steps, rank 16, ~45 min on 4090
    - Advanced: opens $EDITOR with full YAML

[ ] mods train --dataset products --base flux-dev --name product-v1
    - Non-interactive mode with sensible defaults
    - Generates ai-toolkit YAML config
    - Spawns training process
    - Shows: live loss, step count, ETA, sample images
    - Saves LoRA to ~/.mods/models/loras/product-v1.safetensors
    - Records training metadata in SQLite

[ ] Config generation: the opinionated part
    Given base_model + dataset_size + trigger_word, auto-pick:
    - steps (scale with dataset: 150-200 per image, min 1000, max 4000)
    - learning_rate (1e-4 for most, 5e-5 for small datasets)
    - rank (8 for <20 images, 16 for 20-100, 32 for 100+)
    - resolution (match base model default)
    - optimizer (adamw8bit always)
    - sample prompts (auto-generate from trigger + captions)
    - quantize (true if VRAM < 40GB)
```

### Sprint 3: Inference (`mods generate`)

```
[ ] mods generate "<prompt>"
    - Auto-detects: if prompt contains a known trigger word, loads that LoRA
    - Otherwise uses default base model
    - Sensible defaults: 1024x1024, 28 steps, guidance 3.5
    - Saves to ~/.mods/outputs/<date>/<timestamp>.png
    - Prints: seed, inference time, file path

[ ] mods generate "<prompt>" --lora <name> --seed 42 --size 16:9
    - Size presets: 1:1 (1024x1024), 16:9 (1344x768), 9:16, 4:3
    - Seed for reproducibility
    - --count 4 for batch

[ ] mods generate --batch prompts.txt
    - One prompt per line
    - Parallel generation if VRAM allows

[ ] Output management
    - SQLite tracks: prompt, seed, lora, params, file path, timestamp
    - mods outputs (list recent)
    - mods outputs search "marble countertop"
    - mods outputs open <id> (opens in system viewer)
```

### Sprint 4: Web UI (`mods serve`)

```
[ ] mods serve → starts Next.js on localhost:3000
    - Reads from same ~/.mods/ directory
    - SQLite for metadata
    - Filesystem for images/models

[ ] Pages (4 total, that's it):

    1. Dashboard (/)
       - Recent generations (image grid)
       - Active training jobs
       - Model count, storage used

    2. Generate (/generate)
       - THE main page, like A1111's txt2img but cleaner
       - Prompt textarea (big, center)
       - LoRA dropdown (grouped by base model)
       - Quick params: size preset, guidance, steps
       - "Advanced" disclosure → all params
       - Generate button → image(s) appear below
       - Click image → full size + metadata + "regenerate" + "download"
       - History sidebar (last 50 generations)

    3. Models (/models)
       - Grid of base models + LoRAs
       - Each card: preview image, name, size, base model
       - Click → detail: training config, sample images
       - "Train new" button → training form
       - "Pull model" → search/download

    4. Train (/train)
       - Dataset selector (or upload)
       - Base model selector
       - The three presets: Quick / Standard / Advanced
       - Live training view: loss graph, sample images, progress
       - History of past training runs

[ ] UI stack:
    - Next.js 14, App Router
    - shadcn/ui (Button, Card, Select, Slider, Dialog, Tabs)
    - Tailwind
    - Dark mode only (it's an image tool, dark is correct)
    - No auth (it's local)
    - Communicates with mods CLI via local HTTP API or direct SQLite reads
```

### Sprint 5: Modal Cloud Burst (`mods deploy`)

```
[ ] mods deploy setup
    - Prompts for Modal token
    - Stores in ~/.mods/config.toml
    - Creates Modal secrets (HF token)

[ ] mods train --cloud
    - Same command, adds --cloud flag
    - Generates config, uploads dataset to Modal volume
    - Spawns Modal training function
    - Streams logs back to terminal
    - Downloads completed LoRA to local

[ ] mods deploy serve
    - Deploys inference endpoint to Modal
    - Returns URL: https://your-app--inference.modal.run
    - mods generate "prompt" --remote → uses Modal endpoint

[ ] mods deploy sync
    - Pushes local LoRAs to B2/Modal volume
    - Pulls remote LoRAs to local
    - Bidirectional sync of model registry
```

## What mods is NOT

- **Not a node editor.** No graphs. If you want ComfyUI, use ComfyUI.
- **Not a marketplace.** CivitAI exists. mods can *pull from* CivitAI.
- **Not a hosted service.** You run it. On your machine or your Modal account.
- **Not infinitely configurable.** Three training presets. If Quick/Standard doesn't work, Advanced gives you the full YAML. That's it.
- **Not a VC play.** It's a tool. Like Ollama. Use local or pay cloud usage.


## Product / Business Path (if you want one)

### Option A: Open Source Tool (recommended start)

- MIT license (already planned)
- Build community, get feedback
- Revenue: none initially, and that's fine
- Like: Kamal, Rails, Ollama


## Why This Works (For You Specifically)

1. **You're already building mods** — this is just the vision crystallized
2. **You're the ideal user** — 4090, trains LoRAs, builds products on top of them
3. **It feeds ReframeHQ** — train product LoRAs with mods, deploy inference via Modal, serve from your Shopify app
4. **It's a portfolio piece** — Rust CLI + Python ML + Next.js UI + Modal cloud. Shows the full stack.
5. **It solves YOUR confusion** — right now you switch between ComfyUI, ai-toolkit CLI, manual HF downloads, and scattered outputs. mods unifies all of that.
6. **The timing is right** — A1111 is dying, ComfyUI is hostile, Forge is frozen, and new models (Flux, Z-Image, Qwen) are arriving monthly with no good tool to manage them.


## First Weekend: What to Actually Build

Forget everything else. Build this:

```bash
# Friday evening
mods models pull flux-schnell    # downloads to ~/.mods/models/
mods models ls                   # lists what you have

# Saturday
mods datasets create products --from ~/photos/airbnb-products/
mods train --dataset products --base flux-schnell --name airbnb-v1
# → trains on your 4090, 20 mins, saves LoRA

# Sunday
mods generate "a photo of OHWX in a modern apartment, morning light"
# → generates image, saves to ~/.mods/outputs/
mods serve
# → opens web UI, shows your models and generations
```

That's the MVP. Three commands to go from photos to trained LoRA to generated images. Everything else is iteration.