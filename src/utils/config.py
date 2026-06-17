"""Central settings, loaded once from the environment."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    # Cloudflare R2
    r2_access_key_id: str | None
    r2_secret_access_key: str | None
    r2_account_id: str | None
    r2_bucket_name: str
    r2_public_url: str | None   # optional custom domain or R2 public URL base
    disable_r2: bool
    chroma_persist_dir: str
    fastf1_cache_dir: str
    log_level: str


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


settings = Settings(
    groq_api_key=os.getenv("GROQ_API_KEY", ""),
    r2_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
    r2_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
    r2_account_id=os.getenv("R2_ACCOUNT_ID"),
    r2_bucket_name=os.getenv("R2_BUCKET_NAME", "f1-intelligence-reports"),
    r2_public_url=os.getenv("R2_PUBLIC_URL"),
    disable_r2=os.getenv("DISABLE_R2", "").lower() in ("1", "true", "yes"),
    chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./knowledge_base/chroma_db"),
    fastf1_cache_dir=os.getenv("FASTF1_CACHE_DIR", "./cache/fastf1"),
    log_level=os.getenv("LOG_LEVEL", "INFO"),
)
