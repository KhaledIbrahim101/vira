from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from common.celery_app import celery_app
from common.config import settings
from common.db import SessionLocal
from common.logging import configure_logging
from common.models import Job, JobStatus, ShotStatus
from common.schemas import CreateJobRequest, CreateJobResponse, JobResponse
from common.storage import StorageClient
from common.validation import PromptValidationError, validate_prompt

configure_logging()
app = FastAPI(title="Vira API")
storage = StorageClient()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    with SessionLocal() as db:
        total = db.query(Job).count()
        done = db.query(Job).filter(Job.status == JobStatus.DONE).count()
        active = db.query(Job).filter(Job.status.in_([JobStatus.SUBMITTED, JobStatus.PLANNED, JobStatus.SHOTS_QUEUED, JobStatus.SHOTS_RENDERING, JobStatus.POSTPROCESSING])).count()
    body = f"vira_jobs_total {total}\nvira_jobs_done {done}\nvira_jobs_active {active}\n"
    return PlainTextResponse(body)


def _enforce_limits(db):
    active = db.query(Job).filter(Job.status.in_([JobStatus.SUBMITTED, JobStatus.PLANNED, JobStatus.SHOTS_QUEUED, JobStatus.SHOTS_RENDERING, JobStatus.POSTPROCESSING])).count()
    if active >= settings.max_concurrent_jobs:
        raise HTTPException(status_code=429, detail="Max concurrent jobs reached")

    since = datetime.utcnow() - timedelta(minutes=1)
    recent = db.query(Job).filter(Job.created_at >= since).count()
    if recent >= settings.submission_rate_limit_per_minute:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


@app.post("/jobs", response_model=CreateJobResponse)
def create_job(payload: CreateJobRequest):
    try:
        validate_prompt(payload.prompt)
    except PromptValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with SessionLocal() as db:
        _enforce_limits(db)
        job = Job(
            prompt=payload.prompt,
            duration_sec=payload.duration_sec,
            aspect_ratio=payload.aspect_ratio,
            postprocess_config=payload.postprocess.model_dump(),
        )
        db.add(job)
        db.commit()
        db.refresh(job)

    celery_app.send_task("services.director.tasks.plan_job", kwargs={"job_id": str(job.id)})
    return CreateJobResponse(job_id=str(job.id))


@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str):
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        total = max(len(job.shots), 1)
        done = sum(1 for s in job.shots if s.status == ShotStatus.DONE)
        progress = 1.0 if job.status == JobStatus.DONE else done / total
        return JobResponse(
            job_id=str(job.id),
            status=job.status.value,
            progress=progress,
            error_message=job.error_message,
            result_path=job.result_path,
        )


@app.post("/jobs/{job_id}/resume")
def resume_job(job_id: str):
    celery_app.send_task("services.director.tasks.resume_job", kwargs={"job_id": job_id})
    return {"status": "queued"}


@app.get("/jobs/{job_id}/result")
def get_result(job_id: str):
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status != JobStatus.DONE or not job.result_path:
            raise HTTPException(status_code=409, detail="Result not ready")

    if storage.s3_enabled:
        return {"signed_url": storage.sign_url(job.result_path)}
    return FileResponse(storage.get_local_path(job.result_path), media_type="video/mp4", filename=f"{job_id}.mp4")
