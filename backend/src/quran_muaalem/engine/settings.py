from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import torch


class EngineSettings(BaseSettings):
    """Configuration settings for the Quran Muaalem model server."""

    # Model configuration
    model_name_or_path: str = Field(
        default="obadx/muaalem-model-v3_2",
        description="HuggingFace model identifier or local path to the pre-trained model.",
    )
    dtype: Literal["float32", "float16", "bfloat16"] = Field(
        default="bfloat16",
        description="Data type for model weights and inference (supports float32, float16, bfloat16).",
    )
    max_audio_seconds: float = Field(
        default=15,
        description="Maximum Input audio in seconds",
        gt=1.0,
    )

    # Batching configuration
    max_batch_size: int = Field(
        default=128,
        description="Maximum number of requests to batch together. Must be >= 1.",
        ge=2,
    )
    batch_timeout: float = Field(
        default=0.4,
        description="Maximum time (in seconds) to wait for a batch to fill before sending.",
        gt=0.0,
    )

    # Server configuration
    host: str = Field(
        default="0.0.0.0",
        description="Bind address for the server. Use '0.0.0.0' for all interfaces.",
    )
    port: int = Field(
        default=8000,
        description="Port number to listen on.",
        ge=1,
        le=65535,
    )
    accelerator: Literal["cuda", "cpu", "mps"] = Field(
        default="cuda",
        description="Hardware accelerator to use (cuda, cpu, or mps).",
    )
    devices: int = Field(
        default=1,
        description="Number of accelerator devices to use (e.g., number of GPUs).",
        ge=1,
    )

    # Additional LitServer options
    workers_per_device: int = Field(
        default=1,
        description="Number of worker processes per device for handling requests.",
        ge=1,
    )
    timeout: float = Field(
        default=90.0,
        description="Request timeout in seconds.",
        gt=0.0,
    )

    @property
    def torch_dtype(self) -> torch.dtype:
        """Convert the string dtype to a PyTorch dtype."""
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return mapping[self.dtype]
