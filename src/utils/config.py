"""Central settings, loaded once from the environment."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    aws_access_key_id: str | None
    aws_secret_access_key: str | None
    aws_region: str
    s3_bucket_name: str
    chroma_persist_dir: str
    openf1_base_url: str
    log_level: str


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


settings = Settings(
    groq_api_key=_require("GROQ_API_KEY"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_region=os.getenv("AWS_REGION", "us-east-1"),
    s3_bucket_name=os.getenv("S3_BUCKET_NAME", "f1-intelligence-reports"),
    chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./knowledge_base/chroma_db"),
    openf1_base_url=os.getenv("OPENF1_BASE_URL", "https://api.openf1.org/v1"),
    log_level=os.getenv("LOG_LEVEL", "INFO"),
)
