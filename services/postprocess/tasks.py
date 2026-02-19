import subprocess
from pathlib import Path
from common.celery_app import celery_app
from common.db import SessionLocal
from common.models import Job, JobStatus
from common.schemas import PostprocessConfig
from common.storage import StorageClient

storage = StorageClient()


def _run(cmd: list[str]):
    subprocess.run(cmd, check=True)


def _apply_post_pipeline(stitched: str, out_dir: Path, cfg: PostprocessConfig) -> str:
    current = stitched
    if cfg.upscale_enabled:
        upscaled = str(out_dir / "upscaled.mp4")
        _run(["ffmpeg", "-y", "-i", current, "-vf", f"scale={cfg.target_resolution.replace('x',':')}", "-c:v", "libx264", "-c:a", "aac", upscaled])
        current = upscaled

    if cfg.interpolation_enabled:
        interp = str(out_dir / "interpolated.mp4")
        _run(["ffmpeg", "-y", "-i", current, "-vf", f"minterpolate=fps={cfg.target_fps}", "-c:v", "libx264", "-c:a", "aac", interp])
        current = interp

    if cfg.denoise_enabled:
        denoise = str(out_dir / "denoised.mp4")
        _run(["ffmpeg", "-y", "-i", current, "-vf", "hqdn3d", "-c:v", "libx264", "-c:a", "aac", denoise])
        current = denoise

    final = str(out_dir / "final.mp4")
    Path(current).rename(final)
    return final


@celery_app.task(bind=True, name="services.postprocess.tasks.postprocess_job", max_retries=3)
def postprocess_job(self, job_id: str):
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if not job:
            return
        if job.result_path and job.status == JobStatus.DONE:
            return
        job.status = JobStatus.POSTPROCESSING
        db.commit()
        shots = sorted(job.shots, key=lambda s: s.idx)
        cfg = PostprocessConfig(**(job.postprocess_config or {}))

    try:
        workdir = Path(f"/tmp/vira_post/{job_id}")
        workdir.mkdir(parents=True, exist_ok=True)
        concat_file = workdir / "concat.txt"
        with concat_file.open("w", encoding="utf-8") as f:
            for shot in shots:
                local_shot = shot.output_path
                if storage.s3_enabled:
                    local_shot = storage.fetch_to_local(shot.output_path, str(workdir / f"{shot.idx}.mp4"))
                f.write(f"file '{local_shot}'\n")

        stitched = str(workdir / "stitched.mp4")
        _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", stitched])
        final = _apply_post_pipeline(stitched, workdir, cfg)

        object_path = f"jobs/{job_id}/final.mp4"
        stored = storage.upload_file(final, object_path)

        with SessionLocal() as db:
            job = db.get(Job, job_id)
            job.result_path = stored
            job.status = JobStatus.DONE
            db.commit()
    except Exception as exc:
        with SessionLocal() as db:
            job = db.get(Job, job_id)
            if job:
                job.error_message = f"Postprocess failed: {exc}"
                db.commit()
        raise self.retry(exc=exc, countdown=10)
