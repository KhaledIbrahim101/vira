from celery import Celery
from common.config import settings

celery_app = Celery(
    "vira",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "services.director.tasks",
        "services.worker_gpu.tasks",
        "services.postprocess.tasks",
    ],
)
celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"
celery_app.conf.accept_content = ["json"]
celery_app.conf.task_routes = {
    "services.director.tasks.plan_job": {"queue": "director"},
    "services.director.tasks.resume_job": {"queue": "director"},
    "services.worker_gpu.tasks.render_shot": {"queue": "gpu"},
    "services.postprocess.tasks.postprocess_job": {"queue": "postprocess"},
}
