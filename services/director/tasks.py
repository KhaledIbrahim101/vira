from common.celery_app import celery_app
from common.db import SessionLocal
from common.models import Job, JobStatus, Shot, ShotStatus
from common.planner import make_plan


@celery_app.task(name="services.director.tasks.plan_job")
def plan_job(job_id: str):
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if not job:
            return

        if job.plan and job.shots:
            return _enqueue_missing_shots(db, job)

        plan = make_plan(job.prompt, job.duration_sec, job.aspect_ratio)
        job.plan = plan.model_dump()
        job.character_profile = plan.character
        job.status = JobStatus.PLANNED
        db.flush()

        for idx, shot in enumerate(plan.shots):
            db.add(
                Shot(
                    job_id=job.id,
                    idx=idx,
                    duration_sec=shot.duration_sec,
                    prompt=shot.shot_prompt,
                    negative_prompt=shot.negative_prompt,
                    camera=shot.camera,
                    action=shot.action,
                    environment=shot.environment,
                    seed=shot.seed,
                    resolution=shot.resolution,
                    fps_internal=shot.fps_internal,
                    continuity_mode=shot.continuity_mode,
                    input_ref_image_path=shot.input_ref_image_path,
                    status=ShotStatus.QUEUED,
                )
            )
        job.status = JobStatus.SHOTS_QUEUED
        db.commit()
        _enqueue_missing_shots(db, job)


def _enqueue_missing_shots(db, job: Job):
    shots = db.query(Shot).filter(Shot.job_id == job.id).order_by(Shot.idx.asc()).all()
    queued = [s for s in shots if s.status in (ShotStatus.QUEUED, ShotStatus.FAILED)]
    if not queued and all(s.status == ShotStatus.DONE for s in shots):
        job.status = JobStatus.SHOTS_DONE
        db.commit()
        celery_app.send_task("services.postprocess.tasks.postprocess_job", kwargs={"job_id": str(job.id)})
        return

    if queued:
        job.status = JobStatus.SHOTS_RENDERING
        db.commit()
        for shot in queued:
            celery_app.send_task("services.worker_gpu.tasks.render_shot", kwargs={"shot_id": str(shot.id)})


@celery_app.task(name="services.director.tasks.resume_job")
def resume_job(job_id: str):
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if not job:
            return
        if job.status == JobStatus.DONE:
            return
        _enqueue_missing_shots(db, job)
