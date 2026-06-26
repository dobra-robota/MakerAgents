"""Research Agent: generates queries, searches, and persists raw results.

The :class:`ResearchAgent` is the first agent in the pipeline. It:

1. Generates search queries from a city + community pair, including
   local-language variants derived from the city's location.
2. Searches for pain points and existing help/interventions using the
   :class:`~makeragents.search.client.SearchClient`.
3. Stores raw search results as JSON files under the run folder's sources/
   directory.
4. Returns structured output for downstream agents.

When an :class:`~makeragents.llm.LLMClient` is available the agent can
generate queries via the LLM-backed :meth:`llm_generate_queries`, which
uses the research prompt from :mod:`makeragents.prompts`.  The
heuristic :meth:`generate_queries` remains as a fallback.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pathlib import Path

from makeragents.config import AppConfig, load_config
from makeragents.llm import ChatMessage, LLMClient, LLMClientError
from makeragents.prompts import load_prompt
from makeragents.schemas import MakerAgentsModel
from makeragents.search.client import ProviderResponse, SearchClient
from makeragents.search.providers import SearchResult

_logger = logging.getLogger(__name__)

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
    language: str = "en"
    provider: str
    results_count: int
    results: list[SearchResult]
    raw: Any = None


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
        llm_client: LLMClient | None = None,
        config: AppConfig | None = None,
    ) -> None:
        cfg = config if config is not None else load_config()
        self._search_client = (
            search_client if search_client is not None else SearchClient(config=cfg)
        )
        self._llm_client = llm_client

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
    def _infer_query_language(query: str, city: str) -> str:
        """Infer the language code of a query from its content.

        Heuristic queries tag local-language variants with
        ``[language_name]`` (e.g. ``[polish]``).  LLM-generated queries
        are expected to use ``[lang_code]`` prefixes (e.g. ``[pl]``)
        per the research prompt.

        Falls back to ``"en"`` when no tag is detected.
        """
        query_lower = query.lower()
        # Check for [language_code] tags (e.g. [pl], [de])
        tag_match = re.search(r"\[([a-z]{2})\]", query_lower)
        if tag_match:
            code = tag_match.group(1)
            if code in _LANGUAGE_NAMES:
                return code
        # Check for [language_name] tags (e.g. [polish], [german])
        for code, name in _LANGUAGE_NAMES.items():
            if f"[{name.lower()}]" in query_lower:
                return code
        return "en"

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

    def llm_generate_queries(
        self,
        city: str,
        community: str,
        *,
        max_queries: int = 10,
        temperature: float = 0.3,
    ) -> list[str]:
        """Generate search queries using the LLM and the research prompt.

        Uses :func:`makeragents.prompts.load_prompt` to load the research
        Markdown prompt template, fills in ``city``, ``community``, and
        ``max_queries``, then calls the LLM with ``chat_json`` to get a
        structured ``{"queries": [...]}`` response.

        Falls back to :meth:`generate_queries` if no LLM client is
        configured or if the LLM call fails.

        Args:
            city: The city being researched.
            community: The community being researched.
            max_queries: Maximum number of queries to generate.
            temperature: LLM sampling temperature.

        Returns:
            A list of search query strings.
        """
        if self._llm_client is None:
            _logger.info(
                "No LLM client configured — falling back to heuristic queries"
            )
            return self.generate_queries(city, community)

        try:
            languages = self._languages_for_city(city)
            language_names = ", ".join(
                f"{_LANGUAGE_NAMES.get(l, l)} ({l})" for l in languages
            )
            prompt = load_prompt(
                "research",
                city=city,
                community=community,
                max_queries=str(max_queries),
                languages=language_names,
            )
            messages = [ChatMessage("system", prompt)]
            result = self._llm_client.chat_json(
                messages, temperature=temperature
            )
            raw_queries = result.get("queries", [])
            if not raw_queries:
                raise ValueError("LLM returned an empty queries list")
            # Parse queries — support both dict format (with language)
            # and plain string format (legacy / fallback).
            queries: list[str] = []
            for item in raw_queries:
                if isinstance(item, dict):
                    queries.append(item.get("query", ""))
                elif isinstance(item, str):
                    queries.append(item)
            queries = [q for q in queries if q]
            if not queries:
                raise ValueError("LLM returned an empty queries list")
            # Truncate to max_queries in case the model returns extra
            return queries[:max_queries]
        except (LLMClientError, ValueError, KeyError) as exc:
            _logger.warning(
                "LLM query generation failed (%s) — "
                "falling back to heuristic queries",
                exc,
            )
            return self.generate_queries(city, community)

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
            1. Generate search queries via :meth:`llm_generate_queries`
               (which falls back to :meth:`generate_queries` when no LLM
               client is available or the call fails).
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
        all_queries = self.llm_generate_queries(
            city, community, max_queries=queries_per_run
        )
        selected = all_queries[:queries_per_run]
        query_results: list[ResearchQueryResult] = []

        for query in selected:
            language = self._infer_query_language(query, city)
            response: ProviderResponse = self._search_client.search(
                query, count=results_per_query
            )
            query_results.append(
                ResearchQueryResult(
                    query=query,
                    language=language,
                    provider=response.provider,
                    results_count=len(response.results),
                    results=list(response.results),
                    raw=response.raw,
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
