import logging
import subprocess
from pathlib import Path
from common.celery_app import celery_app
from common.config import settings
from common.db import SessionLocal
from common.models import Job, JobStatus, Shot, ShotStatus
from common.storage import StorageClient
from services.worker_gpu.runner import DummyRunner, ModelRunner, WanRunner

logger = logging.getLogger(__name__)
storage = StorageClient()

# Single runner per worker process: model is loaded once (on first shot) and reused for all shots in this process.
_runner: ModelRunner | None = None


def get_runner() -> ModelRunner:
    global _runner
    if _runner is not None:
        return _runner

    if settings.model_backend == "wan":
        logger.info("Creating Wan runner (model will load on first shot and be cached for all shots)")
        _runner = WanRunner(
            model_path=settings.wan_model_path,
            device=settings.wan_device,
            dtype=settings.wan_dtype,
            vram_mode=settings.wan_vram_mode,
        )
    else:
        _runner = DummyRunner()
    return _runner


def _extract_last_frame(video_path: str, out_png: str) -> str:
    subprocess.run(["ffmpeg", "-y", "-sseof", "-0.1", "-i", video_path, "-vframes", "1", out_png], check=True)
    return out_png


@celery_app.task(bind=True, name="services.worker_gpu.tasks.render_shot", max_retries=3)
def render_shot(self, shot_id: str):
    runner = get_runner()
    with SessionLocal() as db:
        shot = db.get(Shot, shot_id)
        if not shot:
            return
        if shot.output_path and shot.status == ShotStatus.DONE:
            return
        shot.status = ShotStatus.RENDERING
        db.commit()

    try:
        with SessionLocal() as db:
            shot = db.get(Shot, shot_id)
            if shot.continuity_mode == "last_frame" and shot.idx > 0:
                prev = db.query(Shot).filter(Shot.job_id == shot.job_id, Shot.idx == shot.idx - 1).first()
                if not prev or prev.status != ShotStatus.DONE or not prev.output_path:
                    raise self.retry(countdown=min(60, 2 ** max(1, shot.retries + 1)))

                prev_local = prev.output_path
                if storage.s3_enabled:
                    prev_local = storage.fetch_to_local(prev.output_path, f"/tmp/vira_refs/{shot.job_id}/prev_{shot.idx-1}.mp4")
                ref_dir = Path(f"/tmp/vira_refs/{shot.job_id}")
                ref_dir.mkdir(parents=True, exist_ok=True)
                ref_img = str(ref_dir / f"shot_{shot.idx}_ref.png")
                _extract_last_frame(prev_local, ref_img)
                shot.input_ref_image_path = ref_img
                db.commit()
                output = runner.generate_video_from_image(
                    ref_image=ref_img,
                    shot_prompt=shot.prompt,
                    negative_prompt=shot.negative_prompt,
                    duration=shot.duration_sec,
                    resolution=shot.resolution,
                    fps=shot.fps_internal,
                    seed=shot.seed,
                )
            else:
                output = runner.generate_video(
                    shot_prompt=shot.prompt,
                    negative_prompt=shot.negative_prompt,
                    duration=shot.duration_sec,
                    resolution=shot.resolution,
                    fps=shot.fps_internal,
                    seed=shot.seed,
                )

            object_path = f"jobs/{shot.job_id}/shots/{shot.idx}.mp4"
            stored_path = storage.upload_file(output, object_path)

            shot.status = ShotStatus.DONE
            shot.output_path = stored_path
            shot.error_message = None
            db.commit()

            job = db.get(Job, shot.job_id)
            if all(s.status == ShotStatus.DONE for s in job.shots):
                job.status = JobStatus.SHOTS_DONE
                db.commit()
                celery_app.send_task("services.postprocess.tasks.postprocess_job", kwargs={"job_id": str(job.id)})
    except Exception as exc:
        with SessionLocal() as db:
            shot = db.get(Shot, shot_id)
            if not shot:
                return
            shot.status = ShotStatus.FAILED
            shot.retries += 1
            shot.error_message = str(exc)
            db.commit()

            if shot.retries <= shot.max_retries:
                countdown = min(120, 2 ** shot.retries)
                raise self.retry(exc=exc, countdown=countdown)

            job = db.get(Job, shot.job_id)
            job.status = JobStatus.FAILED
            job.error_message = f"Shot {shot.idx} failed after retries: {exc}"
            db.commit()
