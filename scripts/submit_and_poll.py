import sys
import time
import requests

prompt = sys.argv[1] if len(sys.argv) > 1 else "A lone samurai in neon rain"
payload = {
    "prompt": prompt,
    "duration_sec": 12,
    "aspect_ratio": "16:9",
    "postprocess": {
        "upscale_enabled": True,
        "interpolation_enabled": True,
        "denoise_enabled": False,
        "target_resolution": "1920x1080",
        "target_fps": 30,
    },
}
resp = requests.post("http://localhost:8000/jobs", json=payload, timeout=20)
resp.raise_for_status()
job_id = resp.json()["job_id"]
print("job_id", job_id)

while True:
    r = requests.get(f"http://localhost:8000/jobs/{job_id}", timeout=20)
    r.raise_for_status()
    data = r.json()
    print(data)
    if data["status"] in {"DONE", "FAILED"}:
        break
    time.sleep(2)

if data["status"] == "DONE":
    out = requests.get(f"http://localhost:8000/jobs/{job_id}/result", timeout=120)
    out.raise_for_status()
    if out.headers.get("content-type", "").startswith("application/json"):
        print("signed_url", out.json().get("signed_url"))
    else:
        with open(f"{job_id}.mp4", "wb") as f:
            f.write(out.content)
        print("saved", f"{job_id}.mp4")
