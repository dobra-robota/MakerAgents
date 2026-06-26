"""Unit tests for the Research Agent (no real API calls)."""

from __future__ import annotations

from pathlib import Path

from makeragents.agents.research import (
    ResearchAgent,
    ResearchQueryResult,
    SearchResultsOutput,
)
from makeragents.search.client import SearchClient
from makeragents.search.providers import (
    ProviderResponse,
    SearchResult,
)


import json

import httpx

from makeragents.config import AppConfig
from makeragents.llm import LLMClient


def _make_mock_llm_client(queries: list[str] | None = None) -> LLMClient:
    """Return an LLMClient whose HTTP transport returns canned JSON."""

    if queries is None:
        queries = [
            "senior citizens problems Łodz",
            "senior citizens support Łodz",
        ]

    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"queries": queries}),
                },
                "finish_reason": "stop",
            }
        ],
        "model": "deepseek-chat",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    cfg = AppConfig(deepseek_api_key="test-key")
    return LLMClient(config=cfg, http_client=client)


def _make_mock_llm_client_error() -> LLMClient:
    """Return an LLMClient that always returns a 500 error."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"Internal error")

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    cfg = AppConfig(deepseek_api_key="test-key")
    return LLMClient(config=cfg, http_client=client)



def _make_mock_llm_client_dict(
    query_dicts: list[dict[str, str]] | None = None,
) -> LLMClient:
    """Return an LLMClient that returns dict-format queries with language."""

    if query_dicts is None:
        query_dicts = [
            {"query": "senior citizens problems Łodz", "language": "en"},
            {"query": "problemy seniorów Łodź", "language": "pl"},
        ]

    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"queries": query_dicts}),
                },
                "finish_reason": "stop",
            }
        ],
        "model": "deepseek-chat",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    cfg = AppConfig(deepseek_api_key="test-key")
    return LLMClient(config=cfg, http_client=client)


def _make_mock_client() -> SearchClient:
    """Return a SearchClient whose primary provider returns canned results."""

    class _MockProvider:
        name = "brave"

        def search(self, query: str, *, count: int = 5) -> ProviderResponse:
            results = [
                SearchResult(
                    title=f"Result {i} for: {query}",
                    url=f"https://example.com/{i}",
                    snippet=f"Snippet {i} for {query}.",
                )
                for i in range(count)
            ]
            return ProviderResponse(
                provider="brave",
                results=results,
                raw={"mock": True, "query": query, "result_count": len(results)},
            )

    return SearchClient(primary=_MockProvider())


class TestGenerateQueries:
    """Test query generation in isolation."""

    def test_generates_english_pain_queries(self) -> None:
        queries = ResearchAgent.generate_queries(
            "Łodz", "senior citizens"
        )
        pain = [q for q in queries if "problems" in q or "challenges" in q]
        assert len(pain) > 0
        assert any("senior citizens" in q for q in pain)

    def test_generates_english_help_queries(self) -> None:
        queries = ResearchAgent.generate_queries(
            "Łodz", "senior citizens"
        )
        help_ = [
            q
            for q in queries
            if "support" in q or "help" in q or "resources" in q or "assistance" in q
        ]
        assert len(help_) > 0

    def test_generates_polish_language_queries_for_lodz(self) -> None:
        queries = ResearchAgent.generate_queries(
            "Łodz", "senior citizens"
        )
        polish = [q for q in queries if "[polish]" in q.lower()]
        assert len(polish) > 0, (
            "Expected at least one query tagged [polish] for Łodz"
        )

    def test_community_is_lowercased_in_queries(self) -> None:
        queries = ResearchAgent.generate_queries("Łodz", "Senior Citizens")
        for q in queries:
            assert "Senior Citizens" not in q, f"Query not lowercased: {q}"

    def test_unknown_city_still_generates_english_queries(self) -> None:
        queries = ResearchAgent.generate_queries(
            "Smallville", "students"
        )
        assert len(queries) > 0
        # No local-language tags for an unmapped city
        tagged = [q for q in queries if "[" in q and "]" in q]
        assert len(tagged) == 0

    def test_generate_queries_includes_city_name(self) -> None:
        queries = ResearchAgent.generate_queries("Łodz", "senior citizens")
        assert any("Łodz" in q for q in queries)


class TestLLMGenerateQueries:
    """Test the LLM-backed query generation."""

    def test_returns_llm_generated_queries(self) -> None:
        llm_queries = [
            "senior citizens problems Łodz",
            "senior citizens support Łodz",
            "senior citizens complaints Łodz",
        ]
        llm = _make_mock_llm_client(llm_queries)
        agent = ResearchAgent(search_client=_make_mock_client(), llm_client=llm)

        result = agent.llm_generate_queries(
            "Łodz", "senior citizens", max_queries=3
        )
        assert result == llm_queries

    def test_falls_back_to_heuristic_on_llm_error(self) -> None:
        llm = _make_mock_llm_client_error()
        agent = ResearchAgent(search_client=_make_mock_client(), llm_client=llm)

        result = agent.llm_generate_queries(
            "Łodz", "senior citizens", max_queries=3
        )
        # Should still return queries (heuristic fallback)
        assert len(result) > 0
        # Heuristic queries include English pain/help queries
        assert any("problems" in q or "challenges" in q for q in result)

    def test_falls_back_when_no_llm_client(self) -> None:
        agent = ResearchAgent(search_client=_make_mock_client())

        result = agent.llm_generate_queries(
            "Łodz", "senior citizens", max_queries=3
        )
        # Should fall back to heuristic queries
        assert len(result) > 0

    def test_truncates_to_max_queries(self) -> None:
        llm_queries = ["q1", "q2", "q3", "q4", "q5", "q6", "q7"]
        llm = _make_mock_llm_client(llm_queries)
        agent = ResearchAgent(search_client=_make_mock_client(), llm_client=llm)

        result = agent.llm_generate_queries(
            "Łodz", "senior citizens", max_queries=4
        )
        assert len(result) == 4
        assert result == ["q1", "q2", "q3", "q4"]


class TestResearchAgentSearchWithLLM:
    """Test the search method with a mocked LLM client."""

    def test_search_uses_llm_queries(self, tmp_path: Path) -> None:
        llm_queries = ["q1", "q2", "q3"]
        llm = _make_mock_llm_client(llm_queries)
        agent = ResearchAgent(
            search_client=_make_mock_client(), llm_client=llm
        )

        result = agent.search(
            run_dir=tmp_path,
            city="Łodz",
            community="senior citizens",
            queries_per_run=3,
            results_per_query=2,
        )

        assert len(result.query_results) == 3
        assert result.query_results[0].query == "q1"
        assert result.query_results[1].query == "q2"
        assert result.query_results[2].query == "q3"

    def test_search_with_llm_saves_output(self, tmp_path: Path) -> None:
        llm_queries = ["q1", "q2"]
        llm = _make_mock_llm_client(llm_queries)
        agent = ResearchAgent(
            search_client=_make_mock_client(), llm_client=llm
        )

        agent.search(
            run_dir=tmp_path,
            city="Łodz",
            community="senior citizens",
            queries_per_run=2,
            results_per_query=1,
        )

        json_path = tmp_path / "sources" / "search-results.json"
        assert json_path.is_file()
        content = json_path.read_text(encoding="utf-8")
        assert "query_results" in content
        assert "q1" in content
        assert "q2" in content

    def test_search_without_llm_still_works(self, tmp_path: Path) -> None:
        """Search still works with no LLM client (heuristic fallback)."""
        agent = ResearchAgent(search_client=_make_mock_client())

        # Add a mock llm_client set to None explicitly to ensure fallback
        result = agent.search(
            run_dir=tmp_path,
            city="Łodz",
            community="senior citizens",
            queries_per_run=3,
            results_per_query=1,
        )

        assert isinstance(result, SearchResultsOutput)
        assert len(result.query_results) == 3


class TestResearchAgentSearch:
    """Test the search method (SearchClient mocked)."""

    def test_search_returns_structured_output(self, tmp_path: Path) -> None:
        agent = ResearchAgent(search_client=_make_mock_client())

        result = agent.search(
            run_dir=tmp_path,
            city="Łodz",
            community="senior citizens",
            queries_per_run=3,
            results_per_query=2,
        )

        assert isinstance(result, SearchResultsOutput)
        assert len(result.query_results) == 3
        for qr in result.query_results:
            assert isinstance(qr, ResearchQueryResult)
            assert qr.results_count == 2
            assert len(qr.results) == 2
            # Each result is a SearchResult with typed attributes
            for r in qr.results:
                assert isinstance(r, SearchResult)
                assert r.title
                assert r.url
            # Raw payload is preserved
            assert qr.raw is not None
            assert qr.raw["mock"] is True

    def test_search_saves_json_to_sources_directory(
        self, tmp_path: Path
    ) -> None:
        agent = ResearchAgent(search_client=_make_mock_client())

        agent.search(
            run_dir=tmp_path,
            city="Łodz",
            community="senior citizens",
            queries_per_run=2,
            results_per_query=1,
        )

        json_path = tmp_path / "sources" / "search-results.json"
        assert json_path.is_file()
        content = json_path.read_text(encoding="utf-8")
        assert "query_results" in content

    def test_no_real_api_calls(self) -> None:
        """Verify the agent works without real API keys."""
        agent = ResearchAgent(search_client=_make_mock_client())

        result = agent.search(
            run_dir="/tmp/_test_no_api",
            city="Łodz",
            community="senior citizens",
            queries_per_run=1,
            results_per_query=1,
        )

        assert len(result.query_results) == 1
        assert result.query_results[0].provider == "brave"

    def test_partial_results_count(self, tmp_path: Path) -> None:
        """When the provider returns fewer results than results_per_query,
        results_count reflects the actual number returned."""

        class _LimitedMockProvider:
            name = "brave"

            def search(self, query: str, *, count: int = 5) -> ProviderResponse:
                # Always returns 2 results regardless of requested count
                return ProviderResponse(
                    provider="brave",
                    results=[
                        SearchResult(title=f"R1 {query}", url="https://a.com", snippet="Snippet 1"),
                        SearchResult(title=f"R2 {query}", url="https://b.com", snippet="Snippet 2"),
                    ],
                    raw={"mock": True},
                )

        agent = ResearchAgent(search_client=SearchClient(primary=_LimitedMockProvider()))

        result = agent.search(
            run_dir=tmp_path,
            city="Łodz",
            community="senior citizens",
            queries_per_run=2,
            results_per_query=10,
        )

        for qr in result.query_results:
            assert qr.results_count == 2  # mock only returns 2
            assert len(qr.results) == 2


class TestInferQueryLanguage:
    """Unit tests for the _infer_query_language static method."""

    def test_english_query_returns_en(self) -> None:
        assert (
            ResearchAgent._infer_query_language(
                "senior citizens problems Łodz", "Łodz"
            )
            == "en"
        )

    def test_polish_name_tag_returns_pl(self) -> None:
        assert (
            ResearchAgent._infer_query_language(
                "problemy seniorów Łodź [polish]", "Łodz"
            )
            == "pl"
        )

    def test_polish_code_tag_returns_pl(self) -> None:
        assert (
            ResearchAgent._infer_query_language(
                "[pl] problemy seniorów Łodź", "Łodz"
            )
            == "pl"
        )

    def test_german_tag_returns_de(self) -> None:
        assert (
            ResearchAgent._infer_query_language(
                "[de] probleme von senioren Berlin", "Berlin"
            )
            == "de"
        )

    def test_unknown_city_english_only_returns_en(self) -> None:
        assert (
            ResearchAgent._infer_query_language(
                "students problems Smallville", "Smallville"
            )
            == "en"
        )

    def test_heuristic_polish_queries_inferred_as_pl(self) -> None:
        """The heuristic generate_queries tags Polish queries with [polish]."""

        queries = ResearchAgent.generate_queries("Łodz", "senior citizens")
        polish_queries = [q for q in queries if "[polish]" in q.lower()]
        assert len(polish_queries) > 0
        for q in polish_queries:
            assert ResearchAgent._infer_query_language(q, "Łodz") == "pl"

    def test_heuristic_english_queries_inferred_as_en(self) -> None:
        queries = ResearchAgent.generate_queries("Łodz", "senior citizens")
        english_queries = [q for q in queries if "[polish]" not in q.lower()]
        assert len(english_queries) > 0
        for q in english_queries:
            assert ResearchAgent._infer_query_language(q, "Łodz") == "en"


class TestLanguageInSearchResults:
    """Test that search results carry language metadata for downstream use."""

    def test_search_results_have_language_field(self, tmp_path: Path) -> None:
        """Every ResearchQueryResult from search() has a language field."""
        agent = ResearchAgent(search_client=_make_mock_client())

        result = agent.search(
            run_dir=tmp_path,
            city="Łodz",
            community="senior citizens",
            queries_per_run=10,
            results_per_query=1,
        )

        languages = {qr.language for qr in result.query_results}
        # Heuristic queries for Łodz produce both en and pl
        assert "en" in languages, (
            f"Expected English queries in results, got languages: {languages}"
        )
        assert "pl" in languages, (
            f"Expected Polish queries in results, got languages: {languages}"
        )

    def test_language_persisted_in_json_output(self, tmp_path: Path) -> None:
        """The language field is serialized in the persisted JSON."""
        agent = ResearchAgent(search_client=_make_mock_client())

        agent.search(
            run_dir=tmp_path,
            city="Łodz",
            community="senior citizens",
            queries_per_run=3,
            results_per_query=1,
        )

        json_path = tmp_path / "sources" / "search-results.json"
        content = json_path.read_text(encoding="utf-8")
        assert '"language"' in content, (
            "language field missing from persisted JSON"
        )
        assert '"en"' in content or '"pl"' in content, (
            "Expected language codes en or pl in JSON output"
        )

    def test_smallville_only_english_language(self, tmp_path: Path) -> None:
        """Unknown city only produces English-language query results."""
        agent = ResearchAgent(search_client=_make_mock_client())

        result = agent.search(
            run_dir=tmp_path,
            city="Smallville",
            community="students",
            queries_per_run=3,
            results_per_query=1,
        )

        languages = {qr.language for qr in result.query_results}
        assert languages == {"en"}, (
            f"Expected only English for unknown city, got: {languages}"
        )


class TestLLMQueryLanguageInference:
    """Test that LLM path preserves language metadata."""

    def test_llm_dict_queries_parsed_correctly(self) -> None:
        """LLM-generated dict-format queries are parsed; language inferred."""
        query_dicts = [
            {"query": "[en] senior citizens problems Łodz", "language": "en"},
            {"query": "[pl] problemy seniorów Łodź", "language": "pl"},
        ]
        llm = _make_mock_llm_client_dict(query_dicts)
        agent = ResearchAgent(
            search_client=_make_mock_client(), llm_client=llm
        )

        result = agent.llm_generate_queries(
            "Łodz", "senior citizens", max_queries=2
        )
        assert len(result) == 2
        # Both queries are returned (stripped of dict wrapper)
        assert "[en] senior citizens problems Łodz" in result
        assert "[pl] problemy seniorów Łodź" in result

    def test_llm_search_results_have_en_and_pl_language(
        self, tmp_path: Path
    ) -> None:
        """Search via LLM path yields both en and pl language results."""
        query_dicts = [
            {"query": "[en] senior citizens problems Łodz", "language": "en"},
            {"query": "[pl] problemy seniorów Łodź", "language": "pl"},
            {"query": "[en] senior citizens help Łodz", "language": "en"},
        ]
        llm = _make_mock_llm_client_dict(query_dicts)
        agent = ResearchAgent(
            search_client=_make_mock_client(), llm_client=llm
        )

        result = agent.search(
            run_dir=tmp_path,
            city="Łodz",
            community="senior citizens",
            queries_per_run=3,
            results_per_query=1,
        )

        languages = {qr.language for qr in result.query_results}
        assert "en" in languages
        assert "pl" in languages

    def test_llm_fallback_produces_polish_queries(
        self, tmp_path: Path
    ) -> None:
        """When LLM errors, fallback to heuristic still produces Polish."""
        llm = _make_mock_llm_client_error()
        agent = ResearchAgent(
            search_client=_make_mock_client(), llm_client=llm
        )

        result = agent.search(
            run_dir=tmp_path,
            city="Łodz",
            community="senior citizens",
            queries_per_run=10,
            results_per_query=1,
        )

        languages = {qr.language for qr in result.query_results}
        assert "pl" in languages, (
            f"Fallback heuristic should produce Polish queries, got: {languages}"
        )
