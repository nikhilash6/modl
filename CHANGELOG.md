# Changelog

All notable changes to modl are documented here.

## [0.1.7] - 2026-03-15

### Added
- `modl preprocess` command — 7 methods: canny, depth, pose, softedge, scribble, lineart, normal
- `modl generate --controlnet` — structural control with ControlNet weights
  - Z-Image Turbo: v1, v2.1 lite (3 layers), v2.1 full (15 layers)
  - Z-Image (non-turbo): shares turbo ControlNet
  - Flux Dev: Union Pro 2.0
  - Flux Schnell: shares Flux Dev ControlNet
  - SDXL: Union (standard + promax variants)
  - Qwen-Image: Union (quality reduced with GGUF quantization)
- `modl generate --style-ref` — IP-Adapter (SDXL, Flux) and native multi-ref (Klein)
- `modl edit` for Flux 2 Klein 4B/9B — structural editing without ControlNet weights
- ZImageControlWrapper — full v2.1 ControlNet (6GB, 15 layers) fits on 24GB GPUs via model_cpu_offload
- Bundled configs for z-image, flux, qwen-image ControlNets (offline loading)
- ControlNet metadata (type, strength, end, image) embedded in PNG output
- Auto fp8 when ControlNet is active (frees VRAM for ControlNet weights)
- CPU offload detection for large base models (qwen-image, flux-dev)

### Changed
- Default ControlNet strength for Z-Image: 0.75 → 0.6
- Split gen_adapter.py (1517 LOC) into pipeline_loader.py (656), controlnet.py (398), gen_adapter.py (500)

### Removed
- flux2-dev from ControlNet support (needs VideoX-Fun pipeline, not yet in diffusers)

## [0.1.6] - 2026-03-13

### Added
- Database split: core/db.rs (924 LOC) into 7 domain submodules
- Centralized paths module (core::paths)
- Training service layer (core::training)

## [0.1.5] - 2026-03-09

### Added
- Pipeline loading strategy: hf_directory, full_checkpoint, transformer_only, gguf
- Flux 2 Klein 4B/9B support (distilled, 4 steps)
- fp8 layerwise casting for inference
- Bundled model configs (no HF downloads at runtime)

## [0.1.4] - 2026-03-03

### Added
- Web UI (React 19 + TypeScript + Tailwind)
- Axum HTTP server with SSE streaming
- Generate, outputs, datasets, training tabs

## [0.1.3] - 2026-02-25

### Added
- LoRA training via ai-toolkit
- Training presets and preflight checks
- Content-addressed model storage

## [0.1.2] - 2026-02-22

### Added
- Image primitives: score, detect, segment, face-restore, upscale, remove-bg
- Vision-language primitives: ground, describe, vl-tag

## [0.1.1] - 2026-02-19

### Added
- Model dependency resolution (auto-install VAE, text encoders)
- Variant selection (fp16/fp8/GGUF based on GPU VRAM)
- HuggingFace + CivitAI auth

## [0.1.0] - 2026-02-17

### Added
- Initial release
- modl pull, generate, ls, rm
- Flux Dev, Flux Schnell, SDXL support
- Content-addressed store with symlinks
