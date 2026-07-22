import os

from pydantic import Field
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    engine_url: str = Field(
        default="http://0.0.0.0:8000/predict",
        description="URL of the Quran Muaalem engine predict endpoint.",
    )
    host: str = Field(
        default="0.0.0.0",
        description="Bind address for the server.",
    )
    port: int = Field(
        default=8001,
        description="Port number to listen on.",
        ge=1,
        le=65535,
    )
    error_ratio: float = Field(
        default=0.1,
        description="Maximum allowed Levenshtein distance as a fraction of query length.",
        ge=0.0,
        le=1.0,
    )
    max_workers_phonetic_search: int = Field(
        default=max(1, (os.cpu_count() or 4) // 2),
        description="Number of worker threads for phonetic search executor.",
        ge=1,
    )
    max_workers_phonetization: int = Field(
        default=max(1, (os.cpu_count() or 4) // 2),
        description="Number of worker threads for phonetization and error explanation executor.",
        ge=1,
    )
