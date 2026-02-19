"""Integration tests for a running stack (API + workers + db + redis).
Run after `docker compose up --build`.
"""

import os
import time
import requests

BASE = os.getenv("VIRA_API", "http://localhost:8000")


def _wait(job_id: str, timeout=240):
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(f"{BASE}/jobs/{job_id}", timeout=20)
        r.raise_for_status()
        data = r.json()
        if data["status"] in {"DONE", "FAILED"}:
            return data
        time.sleep(2)
    raise TimeoutError("job timeout")


def test_job_10s_done_and_result_exists():
    r = requests.post(f"{BASE}/jobs", json={"prompt": "anime lone hero", "duration_sec": 10, "aspect_ratio": "16:9"}, timeout=30)
    r.raise_for_status()
    job_id = r.json()["job_id"]
    data = _wait(job_id)
    assert data["status"] == "DONE"


def test_job_30s_multi_shot_and_continuity_plumbing():
    r = requests.post(f"{BASE}/jobs", json={"prompt": "anime mech pursuit", "duration_sec": 30, "aspect_ratio": "16:9"}, timeout=30)
    r.raise_for_status()
    job_id = r.json()["job_id"]
    data = _wait(job_id, timeout=360)
    assert data["status"] == "DONE"
