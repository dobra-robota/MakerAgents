"""Environment-backed configuration for MakerAgents."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Self

from makeragents.schemas import MakerAgentsModel


class AppConfig(MakerAgentsModel):
    """Application configuration loaded from process environment variables."""

    openai_api_key: str | None = None
    deepseek_api_key: str | None = None
    default_llm_provider: str = "openai"
    default_llm_model: str | None = None
    deepseek_model: str = "deepseek-chat"
    brave_search_api_key: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Self:
        """Build config from env vars without requiring any real API keys."""

        source = os.environ if env is None else env
        return cls(
            openai_api_key=_blank_to_none(source.get("OPENAI_API_KEY")),
            deepseek_api_key=_blank_to_none(source.get("DEEPSEEK_API_KEY")),
            default_llm_provider=source.get("DEFAULT_LLM_PROVIDER", "openai"),
            default_llm_model=_blank_to_none(source.get("DEFAULT_LLM_MODEL")),
            deepseek_model=source.get("DEEPSEEK_MODEL", "deepseek-chat"),
            brave_search_api_key=_blank_to_none(source.get("BRAVE_SEARCH_API_KEY")),
        )


def load_config(env: Mapping[str, str] | None = None) -> AppConfig:
    """Load the current MakerAgents app configuration."""

    return AppConfig.from_env(env)


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if stripped == "":
        return None
    return stripped
