"""Configuration read from the environment (.env at the project root)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Repository root: src/emporio/config.py -> src/emporio -> src -> root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_ID = "google.gemma-3-4b-it"
DEFAULT_EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"


@dataclass(frozen=True)
class Settings:
    region: str
    model_id: str
    embed_model_id: str
    data_dir: Path
    # Which credential the environment offers -- only so the CLI can say
    # something useful when configuration is missing.
    has_bearer_token: bool
    has_iam_keys: bool

    @property
    def has_credentials(self) -> bool:
        return self.has_bearer_token or self.has_iam_keys


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")

    data_dir = Path(os.getenv("DATA_DIR", "./data"))
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    return Settings(
        region=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-2",
        model_id=os.getenv("BEDROCK_MODEL_ID") or DEFAULT_MODEL_ID,
        embed_model_id=os.getenv("BEDROCK_EMBED_MODEL_ID") or DEFAULT_EMBED_MODEL_ID,
        data_dir=data_dir,
        has_bearer_token=bool(os.getenv("AWS_BEARER_TOKEN_BEDROCK")),
        has_iam_keys=bool(os.getenv("AWS_ACCESS_KEY_ID")),
    )
