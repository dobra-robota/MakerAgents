"""Provider-agnostic search client with Brave-primary, DDG-fallback.

The client tries the primary provider (Brave) first and automatically falls
back to DuckDuckGo when Brave is unavailable, errors, is rate-limited, or has
no configured key. Results are normalized and carry the raw provider payload.
"""

from __future__ import annotations

from makeragents.config import AppConfig, load_config
from makeragents.search.providers import (
    BraveProvider,
    DuckDuckGoProvider,
    ProviderResponse,
    SearchProvider,
    SearchProviderError,
)


class SearchClient:
    """Orchestrates Brave (primary) then DuckDuckGo (fallback)."""

    def __init__(
        self,
        *,
        primary: SearchProvider | None = None,
        fallback: SearchProvider | None = None,
        config: AppConfig | None = None,
    ) -> None:
        cfg = config if config is not None else load_config()
        self._primary = (
            primary
            if primary is not None
            else BraveProvider(api_key=cfg.brave_search_api_key)
        )
        self._fallback = (
            fallback if fallback is not None else DuckDuckGoProvider()
        )

    def search(self, query: str, *, count: int = 10) -> ProviderResponse:
        """Search ``query``, falling back from primary to fallback on failure.

        The primary provider is skipped without an error when it reports it is
        unavailable (e.g. no API key). Any :class:`SearchProviderError` from
        the primary triggers the fallback. If the fallback also fails, its
        error propagates to the caller.
        """

        if getattr(self._primary, "available", True):
            try:
                return self._primary.search(query, count=count)
            except SearchProviderError:
                pass

        return self._fallback.search(query, count=count)
