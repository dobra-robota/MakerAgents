"""Unit tests for the search client and providers (no network)."""

from __future__ import annotations

import httpx
import pytest

from makeragents.config import AppConfig
from makeragents.search import (
    BraveProvider,
    DuckDuckGoProvider,
    ProviderResponse,
    SearchClient,
    SearchProviderError,
    SearchResult,
)

BRAVE_PAYLOAD = {
    "web": {
        "results": [
            {
                "title": "Brave Result One",
                "url": "https://example.com/one",
                "description": "First Brave snippet.",
                "extra_provider_field": "kept-verbatim",
            },
            {
                "title": "Brave Result Two",
                "url": "https://example.com/two",
                "description": "Second Brave snippet.",
            },
        ]
    },
    "query": {"original": "potholes glasgow"},
}

DDG_PAYLOAD = [
    {
        "title": "DDG Result One",
        "href": "https://ddg.example.com/one",
        "body": "First DDG snippet.",
    },
    {
        "title": "DDG Result Two",
        "href": "https://ddg.example.com/two",
        "body": "Second DDG snippet.",
    },
]


def _brave_with_payload(payload: dict, status: int = 200) -> BraveProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    return BraveProvider(api_key="test-key", client=client)


def _ddg_with_payload(payload: list) -> DuckDuckGoProvider:
    class _FakeDDGS:
        def text(self, query: str, max_results: int | None = None) -> list:
            return payload

    return DuckDuckGoProvider(ddgs_factory=_FakeDDGS)


def test_brave_normalizes_results() -> None:
    provider = _brave_with_payload(BRAVE_PAYLOAD)

    response = provider.search("potholes glasgow")

    assert isinstance(response, ProviderResponse)
    assert response.provider == "brave"
    assert response.results == [
        SearchResult(
            title="Brave Result One",
            url="https://example.com/one",
            snippet="First Brave snippet.",
        ),
        SearchResult(
            title="Brave Result Two",
            url="https://example.com/two",
            snippet="Second Brave snippet.",
        ),
    ]


def test_brave_passes_raw_payload_verbatim() -> None:
    provider = _brave_with_payload(BRAVE_PAYLOAD)

    response = provider.search("potholes glasgow")

    assert response.raw == BRAVE_PAYLOAD
    assert (
        response.raw["web"]["results"][0]["extra_provider_field"]
        == "kept-verbatim"
    )


def test_brave_without_key_is_unavailable_and_raises() -> None:
    provider = BraveProvider(api_key=None)

    assert provider.available is False
    with pytest.raises(SearchProviderError):
        provider.search("anything")


def test_brave_http_error_raises_provider_error() -> None:
    provider = _brave_with_payload({"error": "rate limited"}, status=429)

    with pytest.raises(SearchProviderError):
        provider.search("anything")


def test_ddg_normalizes_results_and_keeps_raw() -> None:
    provider = _ddg_with_payload(DDG_PAYLOAD)

    response = provider.search("potholes glasgow")

    assert response.provider == "duckduckgo"
    assert response.raw == DDG_PAYLOAD
    assert response.results == [
        SearchResult(
            title="DDG Result One",
            url="https://ddg.example.com/one",
            snippet="First DDG snippet.",
        ),
        SearchResult(
            title="DDG Result Two",
            url="https://ddg.example.com/two",
            snippet="Second DDG snippet.",
        ),
    ]


def test_client_uses_brave_when_available() -> None:
    client = SearchClient(
        primary=_brave_with_payload(BRAVE_PAYLOAD),
        fallback=_ddg_with_payload(DDG_PAYLOAD),
    )

    response = client.search("potholes glasgow")

    assert response.provider == "brave"
    assert response.results[0].title == "Brave Result One"


def test_client_falls_back_when_brave_missing_key() -> None:
    client = SearchClient(
        primary=BraveProvider(api_key=None),
        fallback=_ddg_with_payload(DDG_PAYLOAD),
    )

    response = client.search("potholes glasgow")

    assert response.provider == "duckduckgo"
    assert response.raw == DDG_PAYLOAD
    assert response.results[0].title == "DDG Result One"


def test_client_falls_back_when_brave_errors() -> None:
    client = SearchClient(
        primary=_brave_with_payload({"error": "boom"}, status=500),
        fallback=_ddg_with_payload(DDG_PAYLOAD),
    )

    response = client.search("potholes glasgow")

    assert response.provider == "duckduckgo"
    assert response.results[0].title == "DDG Result One"


def test_client_builds_brave_from_config_key() -> None:
    config = AppConfig(brave_search_api_key="cfg-key")

    client = SearchClient(fallback=_ddg_with_payload(DDG_PAYLOAD), config=config)

    # The default primary is a BraveProvider seeded from config and is
    # considered available because the config supplies a key.
    assert isinstance(client._primary, BraveProvider)
    assert client._primary.available is True


def test_client_default_primary_unavailable_without_key() -> None:
    config = AppConfig(brave_search_api_key=None)

    client = SearchClient(fallback=_ddg_with_payload(DDG_PAYLOAD), config=config)

    response = client.search("potholes glasgow")

    assert response.provider == "duckduckgo"


@pytest.mark.integration
def test_live_duckduckgo_search() -> None:  # pragma: no cover - opt-in only
    """Live DDG smoke test, skipped by default (no network in CI)."""

    pytest.skip("integration test: requires network; run with -m integration")
