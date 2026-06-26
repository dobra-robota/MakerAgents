"""DeepSeek API client for LLM-backed agent reasoning.

Provides a small, typed interface for chat completions with support for
JSON-structured responses, request-level retry with backoff on transient
errors, and configuration-driven provider/model selection.

PRD: §3, §14, §19.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from makeragents.config import AppConfig, load_config


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


class ChatMessage:
    """A single message in a chat conversation."""

    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content

    def to_dict(self) -> dict[str, str]:
        """Render this message as an OpenAI-compatible message dict."""
        return {"role": self.role, "content": self.content}


class ChatResponse:
    """A chat completion response from the provider."""

    def __init__(
        self,
        content: str,
        model: str,
        usage: dict[str, int] | None = None,
    ) -> None:
        self.content = content
        self.model = model
        self.usage = usage or {}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMClientError(Exception):
    """Base error for LLM client operations."""


class LLMConfigError(LLMClientError):
    """Configuration is invalid (e.g. missing API key)."""


class LLMProviderError(LLMClientError):
    """The provider returned an error or the request failed."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class LLMClient:
    """DeepSeek API client with retry/backoff and JSON-mode support.

    The client reads ``deepseek_api_key``, ``deepseek_model``, and
    ``default_llm_model`` from :class:`~makeragents.config.AppConfig`.
    """

    BASE_URL = "https://api.deepseek.com/v1"
    _RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        *,
        config: AppConfig | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        cfg = config if config is not None else load_config()
        self._api_key = cfg.deepseek_api_key
        self._model = cfg.default_llm_model or cfg.deepseek_model
        self._http = http_client or httpx.Client(
            timeout=httpx.Timeout(120.0),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> ChatResponse:
        """Send a chat completion request and return the model response.

        Args:
            messages: The conversation messages.
            temperature: Sampling temperature (0–2).
            max_tokens: Maximum tokens in the response.
            response_format: Optional OpenAI-compatible format spec
                (e.g. ``{"type": "json_object"}``).
            max_retries: Number of retries on transient errors.

        Returns:
            A :class:`ChatResponse` with the model's reply.

        Raises:
            LLMConfigError: If ``DEEPSEEK_API_KEY`` is not set.
            LLMProviderError: On non-retryable API errors or after
                exhausting retries.
        """
        self._check_config()

        url = f"{self.BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = self._http.post(url, headers=headers, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    choice = data["choices"][0]
                    return ChatResponse(
                        content=choice["message"]["content"],
                        model=data.get("model", self._model),
                        usage=data.get("usage"),
                    )
                if response.status_code in self._RETRYABLE_STATUSES:
                    last_error = LLMProviderError(
                        f"DeepSeek API returned {response.status_code}: "
                        f"{response.text[:200]}"
                    )
                else:
                    raise LLMProviderError(
                        f"DeepSeek API returned {response.status_code}: "
                        f"{response.text[:200]}"
                    )
            except httpx.TimeoutException as exc:
                last_error = LLMProviderError(
                    f"DeepSeek API request timed out: {exc}"
                )
            except httpx.RequestError as exc:
                last_error = LLMProviderError(
                    f"DeepSeek API request failed: {exc}"
                )

            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # 1s, 2s, 4s, …

        raise last_error  # type: ignore[misc]

    def chat_json(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Send a chat request and parse the response as JSON.

        Uses ``response_format={"type": "json_object"}`` to instruct the
        model to return valid JSON.
        """
        response = self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(response.content)
        except json.JSONDecodeError as exc:
            raise LLMProviderError(
                f"Failed to parse LLM JSON response: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_config(self) -> None:
        """Raise :class:`LLMConfigError` if the API key is missing."""
        if not self._api_key:
            raise LLMConfigError(
                "DEEPSEEK_API_KEY is not set. Set it in your environment "
                "or .mise.toml before making LLM calls."
            )
