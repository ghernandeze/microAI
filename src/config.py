from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_processed_dir: Path
    ranking_file: Path
    model_timeout_seconds: int
    model_enabled: bool
    model_api_url: str | None
    model_api_key: str | None
    model_name: str | None
    tavily_api_key: str | None


def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parent.parent
    data_processed_dir = project_root / "data" / "processed"
    ranking_file = data_processed_dir / "ranking_oportunidad.csv"

    api_url = os.getenv("MODEL_API_URL")
    api_key = os.getenv("MODEL_API_KEY")
    model_name = os.getenv("MODEL_NAME", "meta-llama/llama-3.3-70b-instruct:free")
    timeout = int(os.getenv("MODEL_TIMEOUT_SECONDS", "45"))
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    return Settings(
        project_root=project_root,
        data_processed_dir=data_processed_dir,
        ranking_file=ranking_file,
        model_timeout_seconds=timeout,
        model_enabled=bool(api_url),
        model_api_url=api_url,
        model_api_key=api_key,
        model_name=model_name,
        tavily_api_key=tavily_api_key,
    )
