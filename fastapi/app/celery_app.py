from celery import Celery

from .config import get_settings

settings = get_settings()

celery = Celery("abyss")

celery.conf.update(
    broker_url=settings.celery_broker,
    result_backend=settings.celery_backend,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)

celery.autodiscover_tasks(["app.services.tasks", "app.services.bathymetry_tasks"])
