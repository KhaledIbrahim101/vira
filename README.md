# Vira MVP (Phase 2): Anime Text-to-Video Pipeline

Production-style local MVP for prompt-to-video generation with shot-parallel workers, continuity plumbing, configurable post-processing, and S3 signed URL support.

## Architecture
- **API** (`services/api`): job submission/status/result/resume, rate limits, metrics.
- **Director++** (`services/director`): deterministic shot planning + stable style + generated character profile.
- **Queue**: Celery + Redis queues (`director`, `gpu`, `postprocess`).
- **GPU workers** (`services/worker_gpu`): render shots with retry/backoff and continuity ref-frame extraction.
- **Postprocess** (`services/postprocess`): plugin-like chain (upscale -> interpolation -> optional denoise).
- **DB**: Postgres + SQLAlchemy + Alembic.
- **Storage**: local or S3 with signed URL support.

## Phase 2 features

### A) Shot continuity plumbing (I2V-like)
- `ShotPlan` now includes:
  - `continuity_mode` (`none` or `last_frame`)
  - `input_ref_image_path` (optional)
- For shot `N>0` with `continuity_mode=last_frame`, worker:
  1. reads shot `N-1` output,
  2. extracts last frame PNG via ffmpeg,
  3. calls `ModelRunner.generate_video_from_image(...)`.
- `DummyRunner` implements this using a ref-image zoom/pan animation (no text overlay).

### B) Director++ prompting
- Rich prompt template per shot with sections:
  - `STYLE`, `CHARACTER`, `ACTION`, `CAMERA`, `SCENE`, `LIGHTING`, `MOOD`
- Stable style block across shots.
- Deterministic character profile generated once per job and stored in DB (`jobs.character_profile`).
- Shot count rules:
  - 25–30s: 3–6 shots
  - else: 2–4 shots

### C) Postprocess plugin chain
Configurable per job via `POST /jobs` -> `postprocess`:
- `upscale_enabled` (placeholder via ffmpeg scale)
- `interpolation_enabled` (placeholder via ffmpeg `minterpolate`, replaceable by RIFE)
- `denoise_enabled` (optional ffmpeg `hqdn3d`)
- `target_resolution`, `target_fps`

### D) Scalability/reliability
- API rate limiting + max concurrent active jobs.
- Shot retries with exponential backoff.
- Postprocess retries.
- Resume endpoint: `POST /jobs/{job_id}/resume` to continue from existing artifacts.

### E) Storage and URLs
- Local backend (default): API streams file for `/jobs/{id}/result`.
- S3 backend: `/jobs/{id}/result` returns `{ "signed_url": "..." }`.

## API
- `POST /jobs`
  - body: `{ prompt, duration_sec(10-30), aspect_ratio, postprocess }`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/result`
- `POST /jobs/{job_id}/resume`
- `GET /health`
- `GET /metrics`

## Environment variables
- `DATABASE_URL`
- `REDIS_URL`
- `STORAGE_BACKEND` (`local` or `s3`)
- `STORAGE_ROOT`
- `OUTPUT_BUCKET`
- `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL`
- `SIGNED_URL_TTL_SEC`
- `MAX_CONCURRENT_JOBS`
- `SUBMISSION_RATE_LIMIT_PER_MINUTE`

## Local run
```bash
cd infra
docker compose up --build
```

Scale GPU workers:
```bash
docker compose up --scale worker_gpu=4
```

## NVIDIA runtime option
For real model workers:
```yaml
worker_gpu:
  runtime: nvidia
  environment:
    NVIDIA_VISIBLE_DEVICES: all
    NVIDIA_DRIVER_CAPABILITIES: compute,utility,video
```

## Testing
Unit tests:
```bash
pytest tests/test_planner_phase2.py tests/test_schema_validation.py
```

Integration contract tests (requires full stack up):
```bash
pytest tests/test_integration_contract.py
```

## Model adapters
- `services/worker_gpu/runner.py` contains:
  - `ModelRunner` interface
  - `DummyRunner` implementation
  - `WanRunner` skeleton (TODO hooks for real Wan2.x integration; no weights downloaded)



## Wan2.x model backend (feature-flagged)

Set backend selector:
- `MODEL_BACKEND=dummy` (default)
- `MODEL_BACKEND=wan`

WAN env vars:
- `WAN_MODEL_PATH` (default `/models/wan2`)
- `WAN_DEVICE` (default `cuda`)
- `WAN_DTYPE` (`float16|bfloat16|float32`, default `float16`)
- `WAN_VRAM_MODE` (`safe|balanced|max`, default `safe`)

Behavior:
- Worker loads Wan pipelines once per process (singleton runner in worker process).
- Supports both T2V and I2V (`generate_video_from_image`).
- On CUDA OOM, Wan runner retries once with reduced resolution/frames, then fails with a clear error.

### GPU image
A CUDA-based worker image is provided at `services/worker_gpu/Dockerfile.gpu`.

Example build/run:
```bash
docker build -f services/worker_gpu/Dockerfile.gpu -t vira-worker-gpu .
docker run --gpus all   -e MODEL_BACKEND=wan   -e WAN_MODEL_PATH=/models/wan2   -v /path/to/wan2-weights:/models/wan2:ro   vira-worker-gpu celery -A common.celery_app.celery_app worker -Q gpu --concurrency=1 --loglevel=info
```

> Do not commit model weights. Mount them via volume (read-only recommended).

### Benchmark
Use benchmark utility for shot-time and throughput:
```bash
PYTHONPATH=/workspace/Vira/libs/common:/workspace/Vira MODEL_BACKEND=dummy python scripts/benchmark_runner.py --shots 4 --duration 3
# WAN mode (requires GPU deps + mounted model)
PYTHONPATH=/workspace/Vira/libs/common:/workspace/Vira MODEL_BACKEND=wan WAN_MODEL_PATH=/models/wan2 python scripts/benchmark_runner.py --shots 2 --duration 3
```
