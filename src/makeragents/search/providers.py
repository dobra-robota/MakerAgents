"""Search providers and the normalized result model.

Two providers are implemented:

* :class:`BraveProvider` calls the Brave Web Search API over HTTP.
* :class:`DuckDuckGoProvider` wraps the maintained ``ddgs`` library.

Both return :class:`ProviderResponse` objects, which carry normalized
:class:`SearchResult` items *and* the raw provider payload verbatim so it can
be persisted downstream. Neither provider crawls full pages: snippets and
links only.
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx
from pydantic import Field

from makeragents.schemas import MakerAgentsModel

BRAVE_WEB_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
DEFAULT_TIMEOUT = 10.0


class SearchProviderError(RuntimeError):
    """Raised when a provider cannot satisfy a query.

    The :class:`~makeragents.search.client.SearchClient` treats this as a
    signal to fall back to the next provider.
    """


class SearchResult(MakerAgentsModel):
    """A single normalized search result (snippet + link only)."""

    title: str
    url: str
    snippet: str


class ProviderResponse(MakerAgentsModel):
    """Normalized results plus the raw provider payload, verbatim."""

    provider: str
    results: list[SearchResult] = Field(default_factory=list)
    raw: Any = None


class SearchProvider(Protocol):
    """Structural interface every provider implements."""

    name: str

    def search(self, query: str, *, count: int = ...) -> ProviderResponse:
        """Run ``query`` and return a :class:`ProviderResponse`.

        Implementations must raise :class:`SearchProviderError` when the
        provider is unavailable, errors, is rate-limited, or otherwise
        cannot serve the query, so the client can fall back.
        """
        ...


class BraveProvider:
    """Primary provider backed by the Brave Web Search API."""

    name = "brave"

    def __init__(
        self,
        api_key: str | None,
        *,
        client: httpx.Client | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._api_key = api_key
        self._client = client
        self._timeout = timeout

    @property
    def available(self) -> bool:
        """Whether the provider has the credentials it needs to run."""

        return bool(self._api_key)

    def search(self, query: str, *, count: int = 10) -> ProviderResponse:
        if not self.available:
            raise SearchProviderError("Brave API key is not configured")

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self._api_key or "",
        }
        params = {"q": query, "count": count}

        try:
            if self._client is not None:
                response = self._client.get(
                    BRAVE_WEB_SEARCH_URL,
                    headers=headers,
                    params=params,
                    timeout=self._timeout,
                )
            else:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.get(
                        BRAVE_WEB_SEARCH_URL, headers=headers, params=params
                    )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise SearchProviderError(f"Brave request failed: {exc}") from exc
        except ValueError as exc:  # invalid JSON body
            raise SearchProviderError(f"Brave returned invalid JSON: {exc}") from exc

        return ProviderResponse(
            provider=self.name,
            results=_normalize_brave(payload),
            raw=payload,
        )


class DuckDuckGoProvider:
    """Fallback provider backed by the ``ddgs`` metasearch library."""

    name = "duckduckgo"

    def __init__(
        self,
        *,
        ddgs_factory: Any | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._ddgs_factory = ddgs_factory
        self._timeout = timeout

    available = True

    def search(self, query: str, *, count: int = 10) -> ProviderResponse:
        factory = self._ddgs_factory
        if factory is None:
            from ddgs import DDGS  # imported lazily; only needed on fallback

            factory = lambda: DDGS(timeout=int(self._timeout))  # noqa: E731

        try:
            payload = factory().text(query, max_results=count)
        except Exception as exc:  # ddgs raises a variety of error types
            raise SearchProviderError(f"DuckDuckGo request failed: {exc}") from exc

        payload = list(payload or [])
        return ProviderResponse(
            provider=self.name,
            results=_normalize_ddg(payload),
            raw=payload,
        )


def _normalize_brave(payload: Any) -> list[SearchResult]:
    """Map a Brave Web Search payload to normalized results."""

    if not isinstance(payload, dict):
        return []
    web = payload.get("web") or {}
    raw_results = web.get("results") or []
    results: list[SearchResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        results.append(
            SearchResult(
                title=str(item.get("title") or ""),
                url=str(item.get("url") or ""),
                snippet=str(item.get("description") or ""),
            )
        )
    return results


def _normalize_ddg(payload: Any) -> list[SearchResult]:
    """Map a ``ddgs`` text payload to normalized results."""

    if not isinstance(payload, list):
        return []
    results: list[SearchResult] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        results.append(
            SearchResult(
                title=str(item.get("title") or ""),
                url=str(item.get("href") or ""),
                snippet=str(item.get("body") or ""),
            )
        )
    return results
