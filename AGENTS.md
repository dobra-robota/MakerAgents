You are an experienced, pragmatic software engineering AI agent. Do not over-engineer a solution when a simple one is possible. Keep edits minimal. If you want an exception to ANY rule, you MUST stop and get permission first.

# AGENTS.md

## Project Overview

MakerAgents is an early-stage, local-first research and analysis tool for finding where real communities experience pain, friction, confusion, exploitation, or unmet needs, then evaluating safe opportunities to add more value than they take.

The first implementation target is a Python CLI named `maker`. A run investigates exactly one city plus one community, for example:

```bash
maker run --city "Łodz" --community "senior citizens"
```

Current v0 foundation and planned stack:

- Language/runtime: Python
- CLI: Typer (dependency pinned; CLI behavior not implemented yet)
- Orchestration: LangGraph (dependency pinned; graph behavior not implemented yet)
- Dependency manager: uv
- Schemas: Pydantic
- Prompts: Markdown files (planned)
- Storage/output: filesystem-only run folders and Markdown reports (planned)
- LLM providers: OpenAI API and DeepSeek API
- Search providers: Brave Search first, DuckDuckGo where useful

This repository contains a `uv`-managed Python package scaffold under `src/makeragents`, Pydantic schemas, environment configuration loading, and tests. Do not invent commands or architecture beyond `README.md` and the issue tracker.

## Reference

Important files:

- `README.md` — product vision, planned workflow, agent roles, verdicts, scoring, output structure, development commands, and design principles.
- `pyproject.toml` / `uv.lock` — uv-managed Python project metadata and pinned dependency lockfile.
- `src/makeragents/schemas.py` — Pydantic models and enums for run metadata, evidence, opportunities, scores, POC types, and verdicts.
- `src/makeragents/config.py` — environment-backed configuration loading.
- `tests/` — pytest coverage for schema validation and config loading; tests must not require real API calls.
- `.gitignore` — Python-oriented ignore template; `uv.lock` is intentionally not ignored.
- `LICENSE` — Apache-2.0.

Planned architecture:

```text
Research Agent
  → Evidence Agent
    → Opportunity Agent
      → Maker Agent / Taker Agent in parallel per opportunity
        → Mediator Agent
          → Cost Checker Agent
            → Report Agent
```

Planned local output structure is `runs/<run-id>/` with `run.yaml`, `final-report.md`, `sources/`, `evidence/`, `opportunities/<slug>/`, and `appendix/` files as described in `README.md`.

## Essential Commands

Use `uv` for project commands.

| Purpose | Current command |
| --- | --- |
| Sync dependencies | `uv sync` |
| Build package | `uv build` |
| Format | TODO: add after formatter configuration exists, likely via `uv run ruff format .` if Ruff is chosen. |
| Lint | TODO: add after lint configuration exists, likely via `uv run ruff check .` if Ruff is chosen. |
| Test | `uv run pytest` |
| Clean | TODO: add after generated paths are defined. Be careful not to delete `runs/` unless explicitly intended. |
| Development server | Not applicable for the planned v0 CLI; there is no web server in scope. |
| Inspect shell scripts | `find . -type f -name '*.sh' -not -path './.git/*' -print` |

Useful repository checks that work now:

```bash
git status --short
git diff --check
grep -RInE "TODO|FIXME|HACK|don't|never|always" --exclude-dir=.git --exclude-dir=.venv . || true
```

## Patterns and Workflows

- Follow `README.md` and issue requirements before adding code. If implementation decisions conflict with documented product scope, stop and ask before changing scope.
- Keep v0 local-first: write auditable files under `runs/<run-id>/`; do not add a database, scheduler, auth, web dashboard, or Docker unless the product scope changes explicitly.
- Use strict schemas for agent inputs/outputs. Planned schema work should use Pydantic and should preserve evidence IDs and confidence levels.
- Preserve the run boundary: one run is one city plus one community. The system discovers domains from research.
- Evidence comes before argument. Claims should be classified as `evidence_based`, `inference`, `assumption`, or `unknown`, and should cite evidence IDs where possible.
- Failed agents must not fail the whole run. Mark failed opportunities `INCOMPLETE` and keep enough state on disk for `maker retry runs/<run-id> --opportunity <opportunity-slug>`.
- Tests should not require real API calls unless explicitly marked as integration tests.

## Safety and Anti-Patterns

Do not implement or encourage:

- Invented sources, unsupported claims, hidden weak evidence, or uncited factual claims.
- Exploitation instructions. The Taker Agent is defensive only: identify risk without enabling abuse.
- Illegal interventions or auto-selected actions. The system recommends; it must never decide to act automatically.
- Login-gated scraping, private-community scraping, or full-page crawling in v0.
- Startup-validation, venture-capital, growth-hacking, or extraction-oriented framing.
- Collection of sensitive personal data unless a future requirement explicitly justifies it and safeguards are documented.

Every opportunity must include a Do No Harm section covering vulnerable groups, negative side effects, abuse risks, legal/terms concerns, misinformation, dependency, gatekeeping, false authority, and safeguards before POC.

## Code Style

No formatter or linter configuration exists yet. For Python code:

- Prefer boring, typed, small modules over clever abstractions.
- Prefer uv-managed commands and commit `uv.lock` for reproducible CLI development unless the project owner decides otherwise.
- Keep prompts in Markdown files rather than embedding long prompts in Python.
- Keep generated run artifacts out of source modules and under the documented `runs/<run-id>/` structure.

## Commit and Pull Request Guidelines

Commit history currently uses concise imperative summaries, e.g. `Initial commit`, `Add project scaffold`, `Implement source registry loading`, `Document retry state`. Continue that style.

Before committing:

1. Run `git status --short` and review every changed file.
2. Run `git diff --check`.
3. Run the relevant format/lint/test commands once they exist.
4. For documentation-only changes, verify links, headings, command examples, and README/issue consistency.

Pull request descriptions should include:

- Summary of what changed and why.
- Validation performed, including exact commands and results.
- Any safety, privacy, or Do No Harm implications.
- Any scope changes from `README.md`, issue requirements, or follow-up TODOs.
