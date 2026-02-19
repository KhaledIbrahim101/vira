# Developer Runbook: Vira (Anime Text-to-Video)

This runbook explains how to boot the repository and generate videos locally.

## 1) What this repo does

Pipeline:
1. API accepts a prompt (`POST /jobs`).
2. Director plans multi-shot sequence.
3. GPU worker renders shots (dummy backend by default, WAN optional).
4. Postprocess stitches shots, upscales, interpolates FPS.
5. Result is returned by `/jobs/{id}/result`.

## 2) Prerequisites

- Docker + Docker Compose
- (Optional, WAN backend) NVIDIA GPU + NVIDIA container runtime
- `ffmpeg` is bundled in containers

## 3) Start the full stack (dummy backend)

```bash
cd infra
docker compose up --build
```

Services:
- API: `http://localhost:8000`
- Redis + Postgres + workers are started by compose.

## 4) Generate your first video

### Option A: curl

```bash
curl -X POST http://localhost:8000/jobs \
  -H 'content-type: application/json' \
  -d '{
    "prompt":"cinematic anime swordswoman under moonlight",
    "duration_sec":12,
    "aspect_ratio":"16:9",
    "postprocess": {
      "upscale_enabled": true,
      "interpolation_enabled": true,
      "denoise_enabled": false,
      "target_resolution":"1920x1080",
      "target_fps":30
    }
  }'
```

Copy the returned `job_id`, then:

```bash
curl http://localhost:8000/jobs/<job_id>
curl -L http://localhost:8000/jobs/<job_id>/result --output result.mp4
```

### Option B: helper script

From repo root:

```bash
python scripts/submit_and_poll.py "anime mecha battle at sunrise"
```

## 5) Scale shot rendering parallelism

```bash
cd infra
docker compose up --scale worker_gpu=4
```

Each worker handles one shot task at a time.

## 6) WAN backend (real model path)

Default backend is dummy. To use WAN:

- `MODEL_BACKEND=wan`
- `WAN_MODEL_PATH=/models/wan2`
- `WAN_DEVICE=cuda`
- `WAN_DTYPE=float16`
- `WAN_VRAM_MODE=safe`

### Important
- Do **not** commit model weights.
- Mount weights via volume (read-only recommended), e.g. `/path/to/wan2-weights:/models/wan2:ro`.

### GPU worker image

Build CUDA worker image:

```bash
docker build -f services/worker_gpu/Dockerfile.gpu -t vira-worker-gpu .
```

Run GPU worker manually:

```bash
docker run --gpus all \
  -e MODEL_BACKEND=wan \
  -e WAN_MODEL_PATH=/models/wan2 \
  -e WAN_DEVICE=cuda \
  -e WAN_DTYPE=float16 \
  -e WAN_VRAM_MODE=safe \
  -v /path/to/wan2-weights:/models/wan2:ro \
  vira-worker-gpu celery -A common.celery_app.celery_app worker -Q gpu --concurrency=1 --loglevel=info
```

## 7) Benchmark generation speed

```bash
PYTHONPATH=/workspace/Vira/libs/common:/workspace/Vira MODEL_BACKEND=dummy python scripts/benchmark_runner.py --shots 4 --duration 3
```

WAN benchmark (requires GPU deps and mounted weights):

```bash
PYTHONPATH=/workspace/Vira/libs/common:/workspace/Vira MODEL_BACKEND=wan WAN_MODEL_PATH=/models/wan2 WAN_DEVICE=cuda python scripts/benchmark_runner.py --shots 2 --duration 3
```

## 8) Health and diagnostics

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

Resume unfinished job:

```bash
curl -X POST http://localhost:8000/jobs/<job_id>/resume
```

## 9) Common issues

- `429 Max concurrent jobs reached`: increase `MAX_CONCURRENT_JOBS`.
- `429 Rate limit exceeded`: increase `SUBMISSION_RATE_LIMIT_PER_MINUTE`.
- WAN OOM: use `WAN_VRAM_MODE=safe`, lower duration/resolution, or more VRAM.
- No Docker/NVIDIA runtime: use dummy backend first to validate full flow.
