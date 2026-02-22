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
- **No GPU required** for local development: the default stack uses a dummy backend. Real model (WAN) inference runs on the cloud; a local NVIDIA GPU is only needed if you choose to run the WAN backend on your own machine (not recommended for consumer GPUs, e.g. under 8GB VRAM).
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

Multi-line (Unix/WSL):

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

Single-line (copy-paste friendly on Windows/MINGW):

```bash
curl -X POST http://localhost:8000/jobs -H "content-type: application/json" -d "{\"prompt\":\"cinematic anime swordswoman under moonlight\",\"duration_sec\":10,\"aspect_ratio\":\"16:9\",\"postprocess\":{\"upscale_enabled\":true,\"interpolation_enabled\":true,\"denoise_enabled\":false,\"target_resolution\":\"1920x1080\",\"target_fps\":30}}"
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

### Where to get a GPU machine

You need a machine with an NVIDIA GPU to run the WAN worker. If your laptop/PC doesn’t have a suitable GPU (e.g. &lt;8GB VRAM), use one of these:

| Option | What it is | Typical use |
|--------|------------|--------------|
| **Cloud GPU VMs** | Rent a VM with an NVIDIA GPU by the hour. | Run the WAN worker in the cloud; your API and Postgres/Redis can stay on your machine or in the same cloud. |
| **RunPod / Lambda Labs / Vast.ai** | GPU cloud providers aimed at ML workloads; often cheaper and simpler than big clouds. | Same as above: create a GPU instance, install Docker + NVIDIA runtime, run the worker container. |
| **AWS (EC2 GPU instances)** | e.g. `g4dn` or `g5` instances with NVIDIA T4/A10G. | Good if you already use AWS; you can put API + Postgres/Redis + worker there or only the worker. |
| **Google Cloud / Azure** | GPU VMs (e.g. NVIDIA T4, A100). | Same idea: create a GPU VM, run the worker, point it at your Redis/Postgres. |
| **Your own desktop/server** | A PC or server with an NVIDIA GPU (e.g. RTX 3080, 4090). | Run the worker on the same LAN as your API, or on the same machine. |

**Typical flow:** Create a GPU instance (e.g. on RunPod or AWS) → SSH in → install Docker and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) → build/pull the worker image, download model weights → run the worker with `DATABASE_URL` and `REDIS_URL` pointing at your **existing** API’s Postgres and Redis (so the worker and API share the same job queue and DB).

### Deploying the WAN worker on GCP (gcloud)

All steps use **project** `vira-488000`, **region** `asia-southeast1`, **zone** `asia-southeast1-b`. Run gcloud from your PC or Cloud Shell; SSH steps run on the VMs after you connect.

---

**Step 1 – Enable Compute Engine API and set project**

```bash
gcloud services enable compute.googleapis.com --project=vira-488000
gcloud config set project vira-488000
```

Ensure the project has **billing** enabled and **GPU quota** (e.g. "NVIDIA T4 GPUs") requested in [Quotas](https://console.cloud.google.com/iam-admin/quotas?project=vira-488000). Request an increase from 0 if needed.

---

**Step 2 – Create VPC network and subnet**

```bash
gcloud compute networks create vira-network --project=vira-488000 --subnet-mode=custom
gcloud compute networks subnets create vira-subnet --project=vira-488000 --region=asia-southeast1 --network=vira-network --range=10.0.0.0/24
```

**Step 2b – Allow SSH (and API port) in the VPC**

Custom VPCs do not get the default “allow SSH” rule. Add a firewall rule so you can SSH and reach the API:

```bash
gcloud compute firewall-rules create vira-allow-ssh-api --project=vira-488000 --network=vira-network --allow=tcp:22,tcp:8000 --source-ranges=0.0.0.0/0 --description="SSH and API for Vira VMs"
```

(To restrict to your IP only, replace `0.0.0.0/0` with e.g. `YOUR_IP/32`.)

---

**Step 3 – Create API VM (no GPU)**

```bash
gcloud compute instances create vira-api --project=vira-488000 --zone=asia-southeast1-b --machine-type=e2-medium --network-interface=network=vira-network,subnet=vira-subnet --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud --boot-disk-size=30GB
```

---

**Step 4 – Create GPU VM**

```bash
gcloud compute instances create vira-gpu-worker --project=vira-488000 --zone=asia-southeast1-b --machine-type=n1-standard-4 --accelerator=type=nvidia-tesla-t4,count=1 --network-interface=network=vira-network,subnet=vira-subnet --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud --maintenance-policy=TERMINATE --boot-disk-size=100GB
```

---

**Step 5 – Get API VM internal IP**

Use this for the GPU worker's `DATABASE_URL` and `REDIS_URL`:

```bash
gcloud compute instances describe vira-api --project=vira-488000 --zone=asia-southeast1-b --format="get(networkInterfaces[0].networkIP)"
```

Save the output (e.g. `10.0.0.2`) as `API_INTERNAL_IP`.

---

**Step 6 – On the API VM: install Docker, clone repo, start stack**

SSH in:

```bash
gcloud compute ssh vira-api --project=vira-488000 --zone=asia-southeast1-b
```

On the API VM (Docker Compose plugin requires Docker’s official repo on Ubuntu):

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin git
sudo systemctl enable docker && sudo systemctl start docker
sudo usermod -aG docker $USER
```

Log out and SSH in again, then:

```bash
git clone <your-repo-url> vira && cd vira/infra
docker compose up -d postgres redis migrate api director postprocess
```

Do **not** start `worker_gpu` on this VM. Exit SSH when done.

---

**Step 7 – On the GPU VM: install NVIDIA driver, Docker, Container Toolkit**

SSH in:

```bash
gcloud compute ssh vira-gpu-worker --project=vira-488000 --zone=asia-southeast1-b
```

On the GPU VM:

```bash
sudo apt-get update && sudo apt-get install -y nvidia-driver-535
sudo reboot
```

After the VM is back (wait 1–2 minutes), SSH in again. Install Docker from Docker’s official repo (so `docker-compose-plugin` is available):

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable docker && sudo systemctl start docker
sudo usermod -aG docker $USER
```

Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (Ubuntu, on the GPU VM):

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Log out and SSH in again (so your user is in the `docker` group).

---

**Step 8 – Shared storage (choose one)**

**Option A – Filestore (NFS)**  
Create a Filestore instance in the same VPC, mount on both VMs (e.g. `/mnt/vira-storage`). On the API VM set `STORAGE_ROOT=/mnt/vira-storage` and mount that path in compose. On the GPU VM run the worker with `-v /mnt/vira-storage:/data/storage` and `STORAGE_ROOT=/data/storage`.

**Option B – GCS (S3-compatible)**  
No shared disk: API and GPU VM both use the same GCS bucket via S3-compatible (XML) API.

1. **Create a GCS bucket** (from your PC or Cloud Shell):

```bash
gsutil mb -p vira-488000 -l asia-southeast1 gs://vira-488000-vira-storage
```

2. **Create HMAC keys** for S3-compatible access (GCS Interoperability).  
   - **Console:** [Cloud Storage → Settings → Interoperability](https://console.cloud.google.com/storage/settings;tab=interoperability) → Create a key for a service account (create one under IAM if needed). Copy the **Access Key** and **Secret**.  
   - **gcloud:** Create a service account, then create HMAC key:

```bash
# Create a service account for storage (if you don't have one)
gcloud iam service-accounts create vira-storage --project=vira-488000 --display-name="Vira GCS S3"
# Grant it access to the bucket (replace BUCKET_NAME)
export BUCKET_NAME=vira-488000-vira-storage
gsutil iam ch serviceAccount:vira-storage@vira-488000.iam.gserviceaccount.com:objectAdmin gs://$BUCKET_NAME
# Create HMAC key; output contains Access ID and Secret (save them)
gcloud storage hmac create vira-storage@vira-488000.iam.gserviceaccount.com --project=vira-488000
```

Save the **Access ID** as `AWS_ACCESS_KEY_ID` and the **Secret** as `AWS_SECRET_ACCESS_KEY`.

3. **On the API VM:** set S3 env in compose so api, director, and postprocess use the bucket. Edit `infra/docker-compose.yml` (or use an override/env file) so the `api`, `director`, and `postprocess` services have:

```yaml
STORAGE_BACKEND: s3
OUTPUT_BUCKET: vira-488000-vira-storage
AWS_REGION: auto
AWS_ACCESS_KEY_ID: "<your-HMAC-access-id>"
AWS_SECRET_ACCESS_KEY: "<your-HMAC-secret>"
AWS_ENDPOINT_URL: https://storage.googleapis.com
```

Remove or leave `STORAGE_ROOT` (not used when backend is s3). Restart: `docker compose up -d api director postprocess`.

4. **On the GPU VM:** when you run the worker container (Step 9), use the same bucket and credentials instead of local storage. Run this **on the GPU VM** (replace `API_INTERNAL_IP` with the API VM internal IP from Step 5, and use your HMAC access ID and secret):

```bash
docker run --gpus all -d --restart unless-stopped \
  -e DATABASE_URL=postgresql+psycopg://postgres:postgres@API_INTERNAL_IP:5432/vira \
  -e REDIS_URL=redis://API_INTERNAL_IP:6379/0 \
  -e STORAGE_BACKEND=s3 \
  -e OUTPUT_BUCKET=vira-488000-vira-storage \
  -e AWS_REGION=auto \
  -e AWS_ACCESS_KEY_ID=YOUR_HMAC_ACCESS_ID \
  -e AWS_SECRET_ACCESS_KEY=YOUR_HMAC_SECRET \
  -e AWS_ENDPOINT_URL=https://storage.googleapis.com \
  -e MODEL_BACKEND=wan \
  -e WAN_MODEL_PATH=/models/wan2 \
  -e WAN_DEVICE=cuda \
  -e WAN_DTYPE=float16 \
  -e WAN_VRAM_MODE=safe \
  -v /home/$USER/wan2-weights:/models/wan2:ro \
  --name vira-wan-worker \
  vira-worker-gpu celery -A common.celery_app.celery_app worker -Q gpu --concurrency=1 --loglevel=info
```

Replace `API_INTERNAL_IP`, `YOUR_HMAC_ACCESS_ID`, and `YOUR_HMAC_SECRET` with your values. Do **not** add `-v /mnt/vira-storage:/data/storage` when using S3.

docker run --gpus all -d --restart unless-stopped \
  -e DATABASE_URL=postgresql+psycopg://postgres:postgres@10.0.0.2:5432/vira \
  -e REDIS_URL=redis://10.0.0.2:6379/0 \
  -e STORAGE_BACKEND=s3 \
  -e OUTPUT_BUCKET=vira-488000-vira-storage \
  -e AWS_REGION=auto \
  -e AWS_ACCESS_KEY_ID=YOUR_HMAC_ACCESS_ID \
  -e AWS_SECRET_ACCESS_KEY=YOUR_HMAC_SECRET \
  -e AWS_ENDPOINT_URL=https://storage.googleapis.com \
  -e MODEL_BACKEND=wan \
  -e WAN_MODEL_PATH=/models/wan2 \
  -e WAN_DEVICE=cuda \
  -e WAN_DTYPE=float16 \
  -e WAN_VRAM_MODE=safe \
  -v /home/$USER/wan2-weights:/models/wan2:ro \
  --name vira-wan-worker \
  vira-worker-gpu celery -A common.celery_app.celery_app worker -Q gpu --concurrency=1 --loglevel=info

---

**Step 9 – On the GPU VM: clone repo, model weights, build and run worker**

SSH into the GPU VM (if not already):

```bash
gcloud compute ssh vira-gpu-worker --project=vira-488000 --zone=asia-southeast1-b
```

Replace `API_INTERNAL_IP` with the value from Step 5. If using Filestore, use that mount path; if using GCS, set `STORAGE_BACKEND=s3` and the same bucket/credentials.

**Option 1 – 1.3B model (recommended for T4 / 16GB VRAM to avoid OOM):**

```bash
# On the GPU VM: create dir and download Wan 1.3B Diffusers
export WAN_WEIGHTS_DIR="${WAN_WEIGHTS_DIR:-/home/$USER/wan13-weights}"
mkdir -p "$WAN_WEIGHTS_DIR"
pip install -q huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Wan-AI/Wan2.1-T2V-1.3B-Diffusers', local_dir='$WAN_WEIGHTS_DIR', local_dir_use_symlinks=False)
"
# Then build and run (use the same docker run as below with -v $WAN_WEIGHTS_DIR:/models/wan2:ro)
```

**Full Step 9 (clone, 1.3B weights, build, run):**

```bash
git clone <your-repo-url> vira && cd vira

# Download 1.3B model (avoids OOM on T4/16GB)
export WAN_WEIGHTS_DIR="${WAN_WEIGHTS_DIR:-/home/$USER/wan13-weights}"
mkdir -p "$WAN_WEIGHTS_DIR"
pip install -q huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
import os
snapshot_download('Wan-AI/Wan2.1-T2V-1.3B-Diffusers', local_dir=os.environ['WAN_WEIGHTS_DIR'], local_dir_use_symlinks=False)
"

docker build -f services/worker_gpu/Dockerfile.gpu -t vira-worker-gpu .
docker run --gpus all -d --restart unless-stopped \
  -e DATABASE_URL=postgresql+psycopg://postgres:postgres@API_INTERNAL_IP:5432/vira \
  -e REDIS_URL=redis://API_INTERNAL_IP:6379/0 \
  -e STORAGE_BACKEND=local \
  -e STORAGE_ROOT=/data/storage \
  -e MODEL_BACKEND=wan \
  -e WAN_MODEL_PATH=/models/wan2 \
  -e WAN_DEVICE=cuda \
  -e WAN_DTYPE=float16 \
  -e WAN_VRAM_MODE=safe \
  -v "$WAN_WEIGHTS_DIR":/models/wan2:ro \
  -v /mnt/vira-storage:/data/storage \
  --name vira-wan-worker \
  vira-worker-gpu celery -A common.celery_app.celery_app worker -Q gpu --concurrency=1 --loglevel=info
```

If using GCS instead of a shared mount, set `-e STORAGE_BACKEND=s3` and the same bucket/credentials; omit or adjust the `-v /mnt/vira-storage:/data/storage` volume. For the GCS docker run block (with 10.0.0.2 and HMAC), use `-v "$WAN_WEIGHTS_DIR":/models/wan2:ro` after downloading 1.3B into `WAN_WEIGHTS_DIR`.

**Free disk after switching to 1.3B:** stop the worker, then remove the old 14B weights dir (e.g. `rm -rf /home/$USER/wan2-weights`) so the VM reclaims that disk space. See “Free disk: remove old Wan weights” below.

---

**Step 10 – Verify**

Get the API VM external IP:

```bash
gcloud compute instances describe vira-api --project=vira-488000 --zone=asia-southeast1-b --format="get(networkInterfaces[0].accessConfigs[0].natIP)"
```

From your PC, submit a job (replace `API_EXTERNAL_IP`):

```bash
curl -X POST http://34.177.93.19:8000/jobs -H "content-type: application/json" -d "{\"prompt\":\"cinematic anime swordswoman under moonlight\",\"duration_sec\":12,\"aspect_ratio\":\"16:9\"}"
```

The job should be picked up by the GPU worker; the final video should reflect your prompt.

curl http://34.177.93.19:8000/jobs/d19e7470-2792-4406-a6b2-4067745e08e9
curl -L http://34.177.93.19:8000/jobs/8b825cb6-d7f0-4b9b-9682-639f8f2194b7/result --output result.mp4

---

### Steps to deploy WAN backend

1. **Prerequisites**
   - A machine with an NVIDIA GPU (see “Where to get a GPU machine” above). Recommended ≥16GB VRAM; `WAN_VRAM_MODE=safe` can reduce usage.
   - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on that machine so Docker can use the GPU (`--gpus all`).

2. **Get model weights**
   - Obtain a diffusers-compatible text-to-video model (e.g. Wan 2.x or compatible Hugging Face model).
   - Save/cache it in a directory on the host, e.g. `/path/to/wan2-weights`, with the layout expected by `diffusers` `AutoPipelineForTextToVideo.from_pretrained(local_path)`.
   - **Default for small GPUs (e.g. T4 16GB):** use **Wan2.1-T2V-1.3B-Diffusers** (`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`) to avoid OOM; the 14B model requires a larger VM (e.g. 24GB+ VRAM).

3. **Build the GPU worker image**
   ```bash
   docker build -f services/worker_gpu/Dockerfile.gpu -t vira-worker-gpu .
   ```

4. **Run the GPU worker**
   - The worker must use the **same** Redis and Postgres as your API (same `REDIS_URL`, `DATABASE_URL`) so jobs flow correctly.
   - **Option A – same host as API (Docker):**
     ```bash
     docker run --gpus all \
       -e DATABASE_URL=postgresql+psycopg://postgres:postgres@<postgres-host>:5432/vira \
       -e REDIS_URL=redis://<redis-host>:6379/0 \
       -e STORAGE_BACKEND=local \
       -e STORAGE_ROOT=/data/storage \
       -e MODEL_BACKEND=wan \
       -e WAN_MODEL_PATH=/models/wan2 \
       -e WAN_DEVICE=cuda \
       -e WAN_DTYPE=float16 \
       -e WAN_VRAM_MODE=safe \
       -v /path/to/wan2-weights:/models/wan2:ro \
       -v /path/to/shared/storage:/data/storage \
       vira-worker-gpu celery -A common.celery_app.celery_app worker -Q gpu --concurrency=1 --loglevel=info
     ```
     Replace `<postgres-host>` / `<redis-host>` with your API’s Postgres/Redis (e.g. `host.docker.internal` on Mac/Windows, or the actual host IP). Use the same storage path/volume the API uses so the worker can read/write shot files.
   - **Option B – cloud GPU VM:** Install Docker + NVIDIA Container Toolkit on the VM, then run the same `docker run` as above, but set `DATABASE_URL` and `REDIS_URL` to your central Postgres and Redis (e.g. RDS and ElastiCache, or your API server’s Postgres/Redis). Ensure the VM can reach Postgres, Redis, and shared storage (e.g. NFS or S3; if S3, set `STORAGE_BACKEND=s3` and AWS env vars).

5. **Leave API/director/postprocess as-is**
   - Keep the API, director, postprocess, Postgres, and Redis running where they are. Only the GPU worker runs with `MODEL_BACKEND=wan`; the director still enqueues `gpu` tasks and the WAN worker consumes them.

6. **Verify**
   - Submit a job via `POST /jobs`. The job should move from SHOTS_RENDERING to SHOTS_DONE and then POSTPROCESSING; the downloaded video should reflect your prompt (real generated content, not the dummy color bars).

### GPU worker image (reference)

Build:

```bash
docker build -f services/worker_gpu/Dockerfile.gpu -t vira-worker-gpu .
```

Run (minimal; replace Redis/Postgres/storage as above):

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

### Free disk: remove old Wan weights

After switching to the 1.3B model (or any new model path), free disk on the GPU VM by removing the old weights directory. **Do this only after the worker is stopped and you are no longer using that path.**

On the GPU VM:

```bash
# 1. Stop and remove the worker (so nothing is using the old mount)
docker stop vira-wan-worker 2>/dev/null; docker rm vira-wan-worker 2>/dev/null

# 2. Remove the old 14B weights dir (adjust path if you used a different one)
rm -rf /home/$USER/wan2-weights

# 3. Optionally reclaim space from Docker (images, build cache)
docker system prune -f
```

Then start the worker again with the new model (e.g. 1.3B in `$WAN_WEIGHTS_DIR`).

## 9) Common issues

- `429 Max concurrent jobs reached`: increase `MAX_CONCURRENT_JOBS`.
- `429 Rate limit exceeded`: increase `SUBMISSION_RATE_LIMIT_PER_MINUTE`.
- WAN OOM: use `WAN_VRAM_MODE=safe`, lower duration/resolution, or more VRAM.
- No Docker/NVIDIA runtime: use dummy backend first to validate full flow.
- **Job stuck in POSTPROCESSING**: the postprocess worker (stitch/upscale/interpolate) is failing and retrying. Check `docker compose logs postprocess` for the error. If you see "minterpolate" or "No such filter", the code now falls back to a simpler fps filter; rebuild and restart the stack.
