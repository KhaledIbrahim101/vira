"""Simple benchmark for per-shot latency and throughput.

Usage:
  MODEL_BACKEND=dummy python scripts/benchmark_runner.py --shots 4 --duration 5 --resolution 960x540 --fps 24
  MODEL_BACKEND=wan WAN_MODEL_PATH=/models/wan2 WAN_DEVICE=cuda python scripts/benchmark_runner.py --shots 2 --duration 3
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from common.config import settings
from services.worker_gpu.runner import DummyRunner, WanRunner


def make_runner():
    if settings.model_backend == "wan":
        return WanRunner(
            model_path=settings.wan_model_path,
            device=settings.wan_device,
            dtype=settings.wan_dtype,
            vram_mode=settings.wan_vram_mode,
            output_root="/tmp/vira_bench",
        )
    return DummyRunner(output_root="/tmp/vira_bench")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--duration", type=int, default=4)
    parser.add_argument("--resolution", default="960x540")
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    Path("/tmp/vira_bench").mkdir(parents=True, exist_ok=True)
    runner = make_runner()

    start = time.perf_counter()
    shot_times = []
    for i in range(args.shots):
        t0 = time.perf_counter()
        out = runner.generate_video(
            shot_prompt=f"benchmark anime action shot {i}",
            negative_prompt="text, watermark, logo",
            duration=args.duration,
            resolution=args.resolution,
            fps=args.fps,
            seed=1000 + i,
        )
        dt = time.perf_counter() - t0
        shot_times.append(dt)
        print(f"shot={i} time_sec={dt:.2f} output={out}")

    total = time.perf_counter() - start
    shots_per_min = (args.shots / total) * 60.0 if total > 0 else 0.0
    print("---")
    print(f"backend={settings.model_backend} device={settings.wan_device if settings.model_backend=='wan' else 'cpu'}")
    print(f"shots={args.shots} total_sec={total:.2f} avg_shot_sec={sum(shot_times)/len(shot_times):.2f} throughput_shots_per_min={shots_per_min:.2f}")


if __name__ == "__main__":
    os.environ.setdefault("MODEL_BACKEND", os.getenv("MODEL_BACKEND", "dummy"))
    main()
