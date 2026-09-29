"""Validated project configuration with environment-variable overrides."""

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_root: Path = Field(default_factory=_project_root)
    selected_season: int | None = Field(default=None, ge=1950)
    selected_round: int | None = Field(default=None, ge=1)
    api_base_url: str = "https://api.jolpi.ca/ergast/f1"
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    model_dir: Path = Path("models")
    default_simulations: int = Field(default=1000, ge=1)
    random_seed: int = 42
    api_timeout_seconds: float = Field(default=30.0, gt=0)
    api_page_size: int = Field(default=1000, ge=1, le=1000)
    max_api_pages: int = Field(default=100, ge=1)
    model_parameters: dict[str, int | float | str | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def resolve_project_paths(self) -> "AppConfig":
        self.project_root = self.project_root.expanduser().resolve()
        for field_name in ("raw_data_dir", "processed_data_dir", "model_dir"):
            path = getattr(self, field_name).expanduser()
            if not path.is_absolute():
                path = self.project_root / path
            setattr(self, field_name, path.resolve())
        if self.selected_round is not None and self.selected_season is None:
            raise ValueError("selected_season is required when selected_round is set")
        return self

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Build configuration from F1_* environment variables."""
        values: dict[str, object] = {}
        env_fields = {
            "F1_PROJECT_ROOT": "project_root",
            "F1_SEASON": "selected_season",
            "F1_ROUND": "selected_round",
            "F1_API_BASE_URL": "api_base_url",
            "F1_RAW_DATA_DIR": "raw_data_dir",
            "F1_PROCESSED_DATA_DIR": "processed_data_dir",
            "F1_MODEL_DIR": "model_dir",
            "F1_NUM_SIMULATIONS": "default_simulations",
            "F1_RANDOM_SEED": "random_seed",
            "F1_API_TIMEOUT_SECONDS": "api_timeout_seconds",
            "F1_API_PAGE_SIZE": "api_page_size",
            "F1_MAX_API_PAGES": "max_api_pages",
        }
        integer_fields = {
            "selected_season",
            "selected_round",
            "default_simulations",
            "random_seed",
            "api_page_size",
            "max_api_pages",
        }
        for env_name, field_name in env_fields.items():
            value = os.getenv(env_name)
            if value is not None:
                values[field_name] = (
                    int(value)
                    if field_name in integer_fields
                    else float(value)
                    if field_name == "api_timeout_seconds"
                    else value
                )
        return cls(**values)