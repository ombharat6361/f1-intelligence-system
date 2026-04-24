"""
Upload race reports to Cloudflare R2.

R2 is S3-compatible, so boto3 is used with a custom endpoint:
  https://<account_id>.r2.cloudflarestorage.com

Returns a public URL if R2_PUBLIC_URL is configured (custom domain or the
R2-managed public bucket URL), otherwise returns the endpoint path.
"""

import logging
import boto3
from botocore.exceptions import BotoCoreError, ClientError

from src.utils.config import settings

logger = logging.getLogger(__name__)


def _r2_client():
    if not settings.r2_account_id:
        raise RuntimeError("Missing required environment variable: R2_ACCOUNT_ID")
    if not settings.r2_access_key_id or not settings.r2_secret_access_key:
        raise RuntimeError("Missing R2_ACCESS_KEY_ID or R2_SECRET_ACCESS_KEY")

    endpoint = f"https://{settings.r2_account_id}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        region_name="auto",
    )


async def upload_report(report: str, filename: str) -> str:
    """Upload a markdown report to R2 and return its URL."""
    import asyncio

    def _upload():
        client = _r2_client()
        client.put_object(
            Bucket=settings.r2_bucket_name,
            Key=filename,
            Body=report.encode("utf-8"),
            ContentType="text/markdown; charset=utf-8",
        )

    try:
        await asyncio.get_event_loop().run_in_executor(None, _upload)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"R2 upload failed: {exc}") from exc

    if settings.r2_public_url:
        base = settings.r2_public_url.rstrip("/")
        return f"{base}/{filename}"

    return f"https://{settings.r2_account_id}.r2.cloudflarestorage.com/{settings.r2_bucket_name}/{filename}"
