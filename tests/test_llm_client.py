"""Tests for the LLM provider client (mocked, no live calls)."""

from __future__ import annotations

import json

import httpx
import pytest

from makeragents.config import AppConfig
from makeragents.llm import (
    ChatMessage,
    ChatResponse,
    LLMClient,
    LLMClientError,
    LLMConfigError,
    LLMProviderError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEEPSEEK_OK_PAYLOAD = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "Hello, World!",
            },
            "finish_reason": "stop",
        }
    ],
    "model": "deepseek-chat",
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

DEEPSEEK_OK_JSON_PAYLOAD = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": json.dumps({"key": "value", "count": 42}),
            },
            "finish_reason": "stop",
        }
    ],
    "model": "deepseek-chat",
}


def _make_client(
    *,
    handler: httpx.MockTransport | None = None,
    api_key: str = "test-key",
) -> LLMClient:
    """Return an :class:`LLMClient` with a mocked HTTP transport."""
    transport = handler or _ok_handler()
    client = httpx.Client(transport=transport)
    cfg = AppConfig(deepseek_api_key=api_key)
    return LLMClient(config=cfg, http_client=client)


def _ok_handler(
    payload: dict | None = None,
    status: int = 200,
) -> httpx.MockTransport:
    payload = payload if payload is not None else DEEPSEEK_OK_PAYLOAD

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler)


def _error_handler(status: int, body: str = "") -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=body.encode())

    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# Unit tests — configuration
# ---------------------------------------------------------------------------


class TestConfig:
    def test_missing_api_key_raises_config_error(self) -> None:
        """LLMConfigError when DEEPSEEK_API_KEY is empty/missing."""
        client = httpx.Client(transport=_ok_handler())
        cfg = AppConfig(deepseek_api_key=None)
        llm = LLMClient(config=cfg, http_client=client)
        with pytest.raises(LLMConfigError, match="DEEPSEEK_API_KEY"):
            llm.chat([ChatMessage("user", "hello")])


# ---------------------------------------------------------------------------
# Unit tests — chat
# ---------------------------------------------------------------------------


class TestChat:
    def test_successful_call_returns_content(self) -> None:
        llm = _make_client()
        response = llm.chat([ChatMessage("user", "hello")])
        assert isinstance(response, ChatResponse)
        assert response.content == "Hello, World!"
        assert response.model == "deepseek-chat"
        assert response.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    def test_passes_temperature_and_max_tokens(self) -> None:
        """The request payload includes temperature and max_tokens."""
        captured: dict | None = None

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured
            captured = json.loads(request.read())
            return httpx.Response(200, json=DEEPSEEK_OK_PAYLOAD)

        llm = _make_client(handler=httpx.MockTransport(handler))
        llm.chat(
            [ChatMessage("user", "hi")],
            temperature=0.2,
            max_tokens=1024,
        )
        assert captured is not None
        assert captured["temperature"] == 0.2
        assert captured["max_tokens"] == 1024

    def test_passes_response_format_json_object(self) -> None:
        captured: dict | None = None

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured
            captured = json.loads(request.read())
            return httpx.Response(200, json=DEEPSEEK_OK_PAYLOAD)

        llm = _make_client(handler=httpx.MockTransport(handler))
        llm.chat(
            [ChatMessage("user", "hi")],
            response_format={"type": "json_object"},
        )
        assert captured is not None
        assert captured["response_format"] == {"type": "json_object"}

    def test_system_and_user_messages(self) -> None:
        captured: dict | None = None

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured
            captured = json.loads(request.read())
            return httpx.Response(200, json=DEEPSEEK_OK_PAYLOAD)

        llm = _make_client(handler=httpx.MockTransport(handler))
        llm.chat(
            [
                ChatMessage("system", "You are helpful."),
                ChatMessage("user", "hello"),
            ],
        )
        assert captured is not None
        assert captured["messages"] == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
        ]


# ---------------------------------------------------------------------------
# Unit tests — chat_json
# ---------------------------------------------------------------------------


class TestChatJson:
    def test_returns_parsed_dict(self) -> None:
        llm = _make_client(handler=_ok_handler(DEEPSEEK_OK_JSON_PAYLOAD))
        result = llm.chat_json([ChatMessage("user", "give me json")])
        assert result == {"key": "value", "count": 42}

    def test_invalid_json_raises_provider_error(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "not json at all",
                    },
                }
            ],
        }
        llm = _make_client(handler=_ok_handler(payload))
        with pytest.raises(LLMProviderError, match="Failed to parse LLM JSON"):
            llm.chat_json([ChatMessage("user", "hi")])


# ---------------------------------------------------------------------------
# Unit tests — errors & retry
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_non_retryable_status_raises_immediately(self) -> None:
        """400 should raise without retry."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(400, content=b"Bad request")

        llm = _make_client(handler=httpx.MockTransport(handler))
        with pytest.raises(LLMProviderError, match="400"):
            llm.chat([ChatMessage("user", "hi")])
        assert call_count == 1  # no retry

    def test_retry_on_429(self) -> None:
        """429 triggers retry, succeeds on second attempt."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(429, content=b"Rate limited")
            return httpx.Response(200, json=DEEPSEEK_OK_PAYLOAD)

        llm = _make_client(handler=httpx.MockTransport(handler))
        response = llm.chat([ChatMessage("user", "hi")])
        assert response.content == "Hello, World!"
        assert call_count == 2

    def test_retry_on_500(self) -> None:
        """500 triggers retry."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return httpx.Response(500, content=b"Internal error")
            return httpx.Response(200, json=DEEPSEEK_OK_PAYLOAD)

        llm = _make_client(handler=httpx.MockTransport(handler))
        response = llm.chat([ChatMessage("user", "hi")])
        assert response.content == "Hello, World!"
        assert call_count == 3

    def test_exhausts_retries(self) -> None:
        """After max_retries failures, raises the last error."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, content=b"Unavailable")

        llm = _make_client(handler=httpx.MockTransport(handler))
        with pytest.raises(LLMProviderError, match="503"):
            llm.chat([ChatMessage("user", "hi")], max_retries=2)

    def test_timeout_triggers_retry(self) -> None:
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.TimeoutException("timed out")
            return httpx.Response(200, json=DEEPSEEK_OK_PAYLOAD)

        llm = _make_client(handler=httpx.MockTransport(handler))
        response = llm.chat([ChatMessage("user", "hi")])
        assert response.content == "Hello, World!"
        assert call_count == 2


# ---------------------------------------------------------------------------
# Integration test (skipped by default)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_real_deepseek_call() -> None:
    """End-to-end smoke test against the live DeepSeek API.

    Requires ``DEEPSEEK_API_KEY`` in the environment.
    Skipped in the default ``uv run pytest`` run.
    """
    from makeragents.config import load_config

    cfg = load_config()
    if not cfg.deepseek_api_key:
        pytest.skip("DEEPSEEK_API_KEY not set")

    llm = LLMClient(config=cfg)
    response = llm.chat(
        [ChatMessage("user", "Say hello in exactly three words.")],
        max_tokens=32,
    )
    assert len(response.content) > 0
    assert response.model
