# MakerAgents

**How can we make the world a better place?**

MakerAgents is a local-first research and analysis tool for discovering where real communities experience pain, friction, confusion, exploitation, or unmet needs, and where Makers may be able to add more value than they take.

It is inspired by multi-agent research workflows, but its purpose is not trading, venture capital, growth hacking, or startup theatre.

MakerAgents asks:

> In this city and community, where are people struggling, and what safe interventions could help?

## Status

Early development.

The repository now has a `uv`-managed Python package scaffold with Pydantic schemas, environment configuration loading, and tests. The CLI workflow and agents are still planned; no research behavior is implemented yet. Detailed v0 requirements remain in `PRD.md`.

## Development

We use [mise](https://mise.jdx.dev/) for reproducible dev tooling (Python and uv versions).

One-time setup after cloning:

```bash
# Install mise if you don't have it: https://mise.jdx.dev/getting-started.html
mise trust
mise install
```

After setup, day-to-day workflows:

```bash
mise run sync   # uv sync
mise run test   # uv run pytest
# or run the commands directly:
uv sync
uv run pytest
```

Tests use local fixtures only and do not require real API keys.

### API Keys

Create a `.env` file (not committed) for your API keys. Mise loads it via `.mise.toml`:

```bash
OPENAI_API_KEY="sk-..."
DEEPSEEK_API_KEY="sk-..."
BRAVE_SEARCH_API_KEY="..."
DEFAULT_LLM_PROVIDER="openai"
DEFAULT_LLM_MODEL="gpt-4o-mini"
DEEPSEEK_MODEL="deepseek-chat"
```

## Core Idea

A single MakerAgents run investigates:

```text
one city + one community
```

Example:

```bash
maker run --city "Łodz" --community "senior citizens"
```

The system researches public sources, identifies repeated pain points, extracts opportunities, and evaluates each one through Maker/Taker analysis.

## Maker vs Taker

MakerAgents uses two opposing analysis modes.

### Maker Agent

The Maker Agent argues where value can be added.

It asks:

```text
Who is struggling?
What burden can be reduced?
What confusion can be clarified?
What process can be improved?
What help would create more value than it extracts?
```

### Taker Agent

The Taker Agent is a defensive red-team agent.

It identifies how an opportunity could be exploited, abused, or turned into extraction.

It asks:

```text
Could this exploit vulnerable people?
Could this create gatekeeping?
Could this create false authority?
Could this worsen the original problem?
Could bad actors abuse this intervention?
```

The Taker Agent must not provide exploitation instructions. It exists to expose risk, not enable abuse.

## Intended Workflow

```text
Research Agent
  → Evidence Agent
    → Opportunity Agent
      → Maker Agent / Taker Agent in parallel
        → Mediator Agent
          → Cost Checker Agent
            → Report Agent
```

## Agents

### Research Agent

Generates search queries from the city and community, searches public sources, and stores snippets and links.

v0 research sources:

```text
web search
news
Reddit/forums
government sites
public social sources where accessible
```

v0 does not scrape login-gated sources or crawl full pages.

### Evidence Agent

Classifies, deduplicates, and scores evidence.

Evidence types:

```text
claim
complaint
official_statement
news_report
first_hand_account
second_hand_account
statistic
unknown
```

### Opportunity Agent

Turns evidence into candidate opportunities.

An opportunity may be commercial or non-commercial. The point is value added, not revenue.

Examples:

```text
public guide
coordination process
advocacy report
transparency dashboard
manual service
community support process
software/tooling
institution-facing report
open data/resource
```

### Maker Agent

Creates the value-add argument and assigns a Maker score.

### Taker Agent

Creates the exploitation-risk argument and assigns a Taker score.

A higher Taker score means higher exploitability risk.

### Mediator Agent

Compares Maker and Taker arguments, summarizes value-add versus value-take, assigns a verdict, and produces a Do No Harm section.

### Cost Checker Agent

Estimates the cost of starting a proof of concept.

It includes:

```text
POC type
USD cost estimate
time estimate
risk level
first 3 actions
```

### Report Agent

Produces the final ranked Markdown report.

## Opportunity Verdicts

Each opportunity receives one verdict:

```text
IGNORE
WATCH
RESEARCH_MORE
MANUAL_POC
BUILD_POC
DO_NOT_TOUCH
NON_INTERVENTION
```

Definitions:

```text
IGNORE = weak signal, low impact
WATCH = interesting but insufficient evidence
RESEARCH_MORE = promising, needs validation
MANUAL_POC = test without software first
BUILD_POC = small tool/prototype justified
DO_NOT_TOUCH = illegal, harmful, exploitative, or likely to worsen the situation
NON_INTERVENTION = real issue, but we are not the right actor to intervene
```

The system recommends action. It does not automatically decide to build.

## Scoring

MakerAgents ranks opportunities using:

```text
rank_score =
  people_helped_score        * 0.22 +
  severity_score             * 0.20 +
  validity_score             * 0.18 +
  intervention_ease_score    * 0.14 +
  low_harm_score             * 0.14 +
  ability_to_act_score       * 0.12
```

Where:

```text
low_harm_score = 100 - harm_risk_score
```

The Taker score is shown separately as an exploitability warning metric.

## Do No Harm

Every opportunity must include a Do No Harm section covering:

```text
vulnerable groups affected
possible negative side effects
abuse/exploitation risks
legal or terms-of-service concerns
trust and misinformation risks
dependency risks
gatekeeping risks
false authority risks
safeguards required before POC
```

Illegal solutions are not opportunities.

## v0 Technology Direction

Planned stack:

```text
Language: Python
CLI: Typer
Orchestration: LangGraph
Dependency manager: uv
Schemas: Pydantic
Prompts: Markdown files
Storage: Filesystem
Output: Markdown
```

Planned LLM providers:

```text
OpenAI API
DeepSeek API
```

Planned search providers:

```text
Brave Search
DuckDuckGo
```

## Planned CLI

```bash
maker run --city "Łodz" --community "senior citizens"
maker run --city "Łodz" --community "senior citizens" --max-opportunities 10
maker retry runs/<run-id> --opportunity <opportunity-slug>
```

## Planned Output Structure

```text
runs/<run-id>/
  run.yaml
  final-report.md

  sources/
    source-registry.yaml
    search-results.json

  evidence/
    evidence.json

  opportunities/
    <opportunity-slug>/
      README.md
      opportunity.yaml
      maker.json
      maker.md
      taker.json
      taker.md
      mediator.json
      mediator.md
      cost.json
      cost.md
      status.yaml

  appendix/
    rejected-opportunities.md
    incomplete-opportunities.md
```

## Source Trust

MakerAgents uses a global source trust registry.

Initial defaults:

```yaml
default_unknown_domain_score: 40

source_type_defaults:
  government: 85
  academic: 80
  major_news: 70
  local_news: 60
  ngo: 60
  company_official: 55
  forum: 40
  reddit: 35
  anonymous_social: 20
```

Users may edit trust scores manually.

## Design Principles

```text
Evidence before argument.
Local-first output.
Strict schemas.
No invented sources.
No exploitation instructions.
No illegal interventions.
Failures must be resumable.
Reports must be auditable.
Makers add more value than they take.
```

## Non-Goals

MakerAgents is not:

```text
a startup idea generator
a venture capital validator
a growth hacking tool
a scraping system for private communities
a tool for exploiting community pain
a replacement for direct human engagement
```

## License

Apache-2.0.
