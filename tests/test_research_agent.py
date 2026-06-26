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
