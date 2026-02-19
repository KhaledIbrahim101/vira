from pydantic import BaseModel, Field
from typing import Literal


class PostprocessConfig(BaseModel):
    upscale_enabled: bool = True
    interpolation_enabled: bool = True
    denoise_enabled: bool = False
    target_resolution: str = "1920x1080"
    target_fps: int = 30


class CreateJobRequest(BaseModel):
    prompt: str
    duration_sec: int = Field(ge=10, le=30)
    aspect_ratio: str = "16:9"
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)


class CreateJobResponse(BaseModel):
    job_id: str


class JobResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    error_message: str | None = None
    result_path: str | None = None


class ShotPlan(BaseModel):
    duration_sec: int
    shot_prompt: str
    camera: str
    action: str
    environment: str
    seed: int
    negative_prompt: str
    resolution: Literal["960x540", "1280x720"]
    fps_internal: int = 24
    continuity_mode: Literal["none", "last_frame"] = "none"
    input_ref_image_path: str | None = None


class JobPlan(BaseModel):
    style_block: str
    character: dict
    shots: list[ShotPlan]
