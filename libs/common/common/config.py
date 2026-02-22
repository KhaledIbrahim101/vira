from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://postgres:postgres@postgres:5432/vira"
    redis_url: str = "redis://redis:6379/0"

    storage_backend: str = "local"  # local | s3
    storage_root: str = "/data/storage"
    output_bucket: str = "vira-output"
    aws_region: str = "us-east-1"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_endpoint_url: str | None = None
    signed_url_ttl_sec: int = 3600

    max_concurrent_jobs: int = 4
    submission_rate_limit_per_minute: int = 30

    model_backend: str = "dummy"  # dummy | wan
    wan_model_path: str = "/models/wan2"
    wan_device: str = "cuda"
    wan_dtype: str = "float16"  # float16 | bfloat16 | float32
    wan_vram_mode: str = "safe"  # safe | balanced | max | 24g | auto
    wan_try_full_resolution_first: bool = False  # when True, skip pre-cap; try requested res first (for 24GB+)
    wan_num_inference_steps: int = 25  # fewer steps = faster, less quality (default 50 in pipeline)

    class Config:
        env_file = ".env"


settings = Settings()
