"""Research Agent: generates queries, searches, and persists raw results.

The :class:`ResearchAgent` is the first agent in the pipeline. It:

1. Generates search queries from a city + community pair, including
   local-language variants derived from the city's location.
2. Searches for pain points and existing help/interventions using the
   :class:`~makeragents.search.client.SearchClient`.
3. Stores raw search results as JSON files under the run folder's sources/
   directory.
4. Returns structured output for downstream agents.
"""

from __future__ import annotations

from pathlib import Path

from makeragents.config import AppConfig, load_config
from makeragents.schemas import MakerAgentsModel
from makeragents.search.client import ProviderResponse, SearchClient

# ---------------------------------------------------------------------------
# City → language mapping — a small, auditable dictionary for v0.
# Extend as needed when new cities are investigated.
# ---------------------------------------------------------------------------
_CITY_LANGUAGES: dict[str, list[str]] = {
    "lodz": ["en", "pl"],
    "krakow": ["en", "pl"],
    "warsaw": ["en", "pl"],
    "wroclaw": ["en", "pl"],
    "poznan": ["en", "pl"],
    "gdansk": ["en", "pl"],
    "sao paulo": ["en", "pt"],
    "santiago": ["en", "es"],
    "tokyo": ["en", "ja"],
    "berlin": ["en", "de"],
    "paris": ["en", "fr"],
    "madrid": ["en", "es"],
    "rome": ["en", "it"],
    "beijing": ["en", "zh"],
    "moscow": ["en", "ru"],
}

# Language → country-code hints for web search queries.
_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "pl": "Polish",
    "pt": "Portuguese",
    "es": "Spanish",
    "ja": "Japanese",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "zh": "Chinese",
    "ru": "Russian",
}

# ---------------------------------------------------------------------------
# Query templates
# ---------------------------------------------------------------------------
_PAIN_QUERIES = (
    "{community} problems in {city}",
    "{community} challenges {city}",
    "{community} complaints {city}",
    "{community} difficulties {city}",
)

_HELP_QUERIES = (
    "{community} support {city}",
    "{community} help {city}",
    "{community} resources {city}",
    "{community} assistance {city}",
)

_LOCAL_LANGUAGE_PAIN_TEMPLATE = (
    "problems {community} in {city} [{language}]",
    "{community} difficulties {city} [{language}]",
)

_LOCAL_LANGUAGE_HELP_TEMPLATE = (
    "help for {community} {city} [{language}]",
    "{community} support {city} [{language}]",
)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
class ResearchQueryResult(MakerAgentsModel):
    """A single query and its provider response."""

    query: str
    provider: str
    results_count: int
    results: list[dict[str, str]]


class SearchResultsOutput(MakerAgentsModel):
    """Structured output from a research search pass.

    This is the return type of :meth:`ResearchAgent.search`.
    """

    query_results: list[ResearchQueryResult]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class ResearchAgent:
    """Generates search queries, runs them, and persists raw output."""

    def __init__(
        self,
        *,
        search_client: SearchClient | None = None,
        config: AppConfig | None = None,
    ) -> None:
        cfg = config if config is not None else load_config()
        self._search_client = (
            search_client if search_client is not None else SearchClient(config=cfg)
        )

    @staticmethod
    def _normalize_city(city: str) -> str:
        """Normalize a city name for language lookup (lowercase, ASCII)."""
        return city.lower().replace("ł", "l")

    @staticmethod
    def _languages_for_city(city: str) -> list[str]:
        """Return the language codes to use when searching for *city*.

        Always includes English; adds local languages when the city is
        recognised.
        """
        key = ResearchAgent._normalize_city(city)
        languages = _CITY_LANGUAGES.get(key, [])
        if "en" not in languages:
            languages = ["en", *languages]
        return languages

    @staticmethod
    def generate_queries(city: str, community: str) -> list[str]:
        """Generate search queries including local-language variants.

        Args:
            city: The city being researched.
            community: The community being researched.

        Returns:
            A list of search query strings. Queries cover pain/challenge
            topics and existing help/support resources, with local-language
            variants derived from the city.
        """
        queries: list[str] = []
        languages = ResearchAgent._languages_for_city(city)
        community_lower = community.lower()

        # English pain queries
        for template in _PAIN_QUERIES:
            queries.append(template.format(city=city, community=community_lower))

        # English help queries
        for template in _HELP_QUERIES:
            queries.append(template.format(city=city, community=community_lower))

        # Local-language variants (skip English, already covered)
        for lang in languages:
            if lang == "en":
                continue
            lang_name = _LANGUAGE_NAMES.get(lang, lang).lower()
            for template in _LOCAL_LANGUAGE_PAIN_TEMPLATE:
                queries.append(
                    template.format(
                        city=city,
                        community=community_lower,
                        language=lang_name,
                    )
                )
            for template in _LOCAL_LANGUAGE_HELP_TEMPLATE:
                queries.append(
                    template.format(
                        city=city,
                        community=community_lower,
                        language=lang_name,
                    )
                )

        return queries

    def search(
        self,
        run_dir: str | Path,
        city: str,
        community: str,
        *,
        queries_per_run: int = 10,
        results_per_query: int = 5,
    ) -> SearchResultsOutput:
        """Execute a research search pass and persist results.

        Steps:
            1. Generate search queries via :meth:`generate_queries`.
            2. Slice the query list to ``queries_per_run``.
            3. Run each query through the configured :class:`SearchClient`,
               requesting ``results_per_query`` results.
            4. Save all raw results as JSON under
               ``<run_dir>/sources/search-results.json``.
            5. Return a :class:`SearchResultsOutput` with the structured
               results.

        Args:
            run_dir: Path to the run folder (created by
                :func:`~makeragents.run.create_run_folder`).
            city: The city being researched.
            community: The community being researched.
            queries_per_run: Maximum number of queries to execute.
                Defaults to 10.
            results_per_query: Maximum number of results to request per
                query. Defaults to 5.

        Returns:
            A :class:`SearchResultsOutput` containing every query result.
        """
        all_queries = self.generate_queries(city, community)
        selected = all_queries[:queries_per_run]
        query_results: list[ResearchQueryResult] = []

        for query in selected:
            response: ProviderResponse = self._search_client.search(
                query, count=results_per_query
            )
            query_results.append(
                ResearchQueryResult(
                    query=query,
                    provider=response.provider,
                    results_count=len(response.results),
                    results=[
                        {"title": r.title, "url": r.url, "snippet": r.snippet}
                        for r in response.results
                    ],
                )
            )

        # Persist raw output
        run_path = Path(run_dir)
        sources_dir = run_path / "sources"
        sources_dir.mkdir(parents=True, exist_ok=True)
        output_path = sources_dir / "search-results.json"
        output = SearchResultsOutput(query_results=query_results)
        output_path.write_text(
            output.model_dump_json(indent=2), encoding="utf-8"
        )

        return output
