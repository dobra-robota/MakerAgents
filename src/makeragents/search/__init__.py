"""Search client abstraction (Brave primary, DuckDuckGo fallback)."""

from makeragents.search.client import SearchClient
from makeragents.search.providers import (
    BraveProvider,
    DuckDuckGoProvider,
    ProviderResponse,
    SearchProvider,
    SearchProviderError,
    SearchResult,
)

__all__ = [
    "BraveProvider",
    "DuckDuckGoProvider",
    "ProviderResponse",
    "SearchClient",
    "SearchProvider",
    "SearchProviderError",
    "SearchResult",
]
