"""Runtime configuration, loaded from the environment (prefix ``PRAXIS_``)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["fake", "openai", "anthropic"]


class Settings(BaseSettings):
    """Process configuration.

    Defaults are chosen so a fresh checkout runs end to end with no API keys:
    ``llm_provider="fake"`` returns deterministic canned structured output.
    """

    model_config = SettingsConfigDict(env_prefix="PRAXIS_", env_file=".env", extra="ignore")

    llm_provider: Provider = "fake"
    strong_model: str = "openai:gpt-4o"
    fast_model: str = "openai:gpt-4o-mini"

    # Roles that run on the cheaper/faster tier (calibration + synthesis work).
    fast_roles: tuple[str, ...] = ("editor", "red_team")

    max_gap_loops: int = 1
    otel_enabled: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
