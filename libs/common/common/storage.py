import shutil
from pathlib import Path
from urllib.parse import quote
import boto3
from common.config import settings


class StorageClient:
    """Storage backend with local filesystem and S3 support."""

    def __init__(self) -> None:
        self.backend = settings.storage_backend
        self.root = Path(settings.storage_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.bucket = settings.output_bucket
        self.s3 = None
        if self.backend == "s3":
            self.s3 = boto3.client(
                "s3",
                region_name=settings.aws_region,
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
                endpoint_url=settings.aws_endpoint_url,
            )

    @property
    def s3_enabled(self) -> bool:
        return self.backend == "s3"

    def upload_file(self, local_path: str, object_path: str) -> str:
        if self.s3_enabled:
            self.s3.upload_file(local_path, self.bucket, object_path)
            return object_path
        dst = self.root / object_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dst)
        return str(dst)

    def fetch_to_local(self, object_or_path: str, local_path: str) -> str:
        local = Path(local_path)
        local.parent.mkdir(parents=True, exist_ok=True)
        if self.s3_enabled:
            self.s3.download_file(self.bucket, object_or_path, str(local))
        else:
            shutil.copy2(object_or_path, local)
        return str(local)

    def sign_url(self, object_path: str) -> str:
        if self.s3_enabled:
            return self.s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": object_path},
                ExpiresIn=settings.signed_url_ttl_sec,
            )
        return f"/storage/{quote(object_path)}"

    def get_local_path(self, object_or_path: str) -> str:
        if self.s3_enabled:
            return str(self.root / object_or_path)
        return object_or_path
