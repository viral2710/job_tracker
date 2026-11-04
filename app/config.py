from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    app_name: str = "Job Tracker API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Storage paths
    data_dir: Path = Path("data")
    delta_metrics_path: Path = Path("data/delta/model_metrics")
    jobs_db_path: Path = Path("data/jobs.json")
    
    # Matching settings
    job_tracker_default_model: str = "gemini"
    default_model: str = "gemini"
    top_k_matches: int = 10

    # Gemini LLM settings
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-3.5-flash-lite"

settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.delta_metrics_path.parent.mkdir(parents=True, exist_ok=True)
