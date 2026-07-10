"""MinIO object storage operations."""

from __future__ import annotations

import io
from typing import BinaryIO

from minio import Minio
from minio.error import S3Error
from minio.sseconfig import Rule, SSEConfig

from ..config import get_settings

settings = get_settings()

BUCKETS = ["raw", "cog", "jph", "vectors", "thumbnails", "exports", "depth", "reports", "s2cache"]


def _get_client() -> Minio:
    return Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.minio_secret,
        secure=False,
    )


def ensure_buckets() -> None:
    """Create required buckets (with SSE-S3 at-rest encryption) if they don't exist."""
    client = _get_client()
    for bucket in BUCKETS:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
        # Encrypt object payloads (imagery, depth rasters, reports) at rest via
        # SSE-S3. Requires MinIO KMS (MINIO_KMS_SECRET_KEY); when KMS is absent the
        # call errors — swallow it so buckets still function unencrypted.
        try:
            client.set_bucket_encryption(bucket, SSEConfig(Rule.new_sse_s3_rule()))
        except S3Error:
            pass


def upload_file(
    bucket: str,
    path: str,
    file_data: BinaryIO,
    content_type: str = "application/octet-stream",
    length: int = -1,
) -> str:
    """Upload a file to MinIO. Returns the object path."""
    client = _get_client()
    part_size = 10 * 1024 * 1024  # 10 MB parts
    client.put_object(
        bucket,
        path,
        file_data,
        length=length,
        content_type=content_type,
        part_size=part_size if length == -1 else 0,
    )
    return path


def download_file(bucket: str, path: str) -> bytes:
    """Download an entire file from MinIO and return its bytes."""
    client = _get_client()
    response = client.get_object(bucket, path)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def get_byte_range(bucket: str, path: str, offset: int, length: int) -> bytes:
    """Read a byte range from an object in MinIO (for HTJ2K streaming)."""
    client = _get_client()
    response = client.get_object(bucket, path, offset=offset, length=length)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def delete_file(bucket: str, path: str) -> None:
    """Delete a file from MinIO."""
    client = _get_client()
    client.remove_object(bucket, path)


def file_exists(bucket: str, path: str) -> bool:
    """Check if a file exists in MinIO."""
    client = _get_client()
    try:
        client.stat_object(bucket, path)
        return True
    except S3Error:
        return False


def get_file_size(bucket: str, path: str) -> int:
    """Return the size of an object in bytes."""
    client = _get_client()
    stat = client.stat_object(bucket, path)
    return stat.size


def list_prefix(bucket: str, prefix: str) -> list[str]:
    """List object keys under a prefix (recursive)."""
    client = _get_client()
    return [obj.object_name for obj in client.list_objects(bucket, prefix=prefix, recursive=True)]


def get_presigned_url(bucket: str, path: str, expires: int = 3600) -> str:
    """Generate a presigned download URL.

    Args:
        expires: URL lifetime in seconds (default 1 hour).
    """
    from datetime import timedelta

    client = _get_client()
    return client.presigned_get_object(bucket, path, expires=timedelta(seconds=expires))
