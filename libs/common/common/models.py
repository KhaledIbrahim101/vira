import enum
import uuid
from datetime import datetime
from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from common.db import Base


class JobStatus(str, enum.Enum):
    SUBMITTED = "SUBMITTED"
    PLANNED = "PLANNED"
    SHOTS_QUEUED = "SHOTS_QUEUED"
    SHOTS_RENDERING = "SHOTS_RENDERING"
    SHOTS_DONE = "SHOTS_DONE"
    POSTPROCESSING = "POSTPROCESSING"
    DONE = "DONE"
    FAILED = "FAILED"


class ShotStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    RENDERING = "RENDERING"
    DONE = "DONE"
    FAILED = "FAILED"


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    duration_sec: Mapped[int] = mapped_column(Integer, nullable=False)
    aspect_ratio: Mapped[str] = mapped_column(String(16), default="16:9")
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.SUBMITTED)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    plan: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    character_profile: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    postprocess_config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    result_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    shots: Mapped[list["Shot"]] = relationship(back_populates="job", cascade="all, delete-orphan")


class Shot(Base):
    __tablename__ = "shots"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("jobs.id"))
    idx: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_sec: Mapped[int] = mapped_column(Integer, nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    negative_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    camera: Mapped[str] = mapped_column(String(128), nullable=False)
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    environment: Mapped[str] = mapped_column(String(128), nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    resolution: Mapped[str] = mapped_column(String(32), nullable=False)
    fps_internal: Mapped[int] = mapped_column(Integer, nullable=False)
    continuity_mode: Mapped[str | None] = mapped_column(String(32), nullable=True)
    input_ref_image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[ShotStatus] = mapped_column(Enum(ShotStatus), default=ShotStatus.QUEUED)
    retries: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    job: Mapped[Job] = relationship(back_populates="shots")
