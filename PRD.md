# MakerAgents PRD

## 1. Product Summary

**MakerAgents** is a local-first Python CLI that researches a city and community, discovers pain points, extracts opportunities to add value, and evaluates each opportunity through Maker/Taker analysis.

The tool is inspired by TradingAgents-style multi-agent workflows, but the domain is community value discovery rather than trading.

Primary question:

> Where are real people in this city/community experiencing pain, delay, confusion, exploitation, or unmet needs — and where can more value be added than taken?

This is not a startup validator.
This is not a growth-hacking tool.
This is not a fundability checker.

It is a white-hat community opportunity researcher.

## 2. Repository / CLI

Repository name:

```text
MakerAgents
```

CLI command:

```bash
maker
```

Example v0 command:

```bash
maker run --city "Łodz" --community "senior citizens"
```

Default output directory:

```text
./runs/
```

v0 uses flags only. No interactive mode.

Containerisation is deferred.

## 3. v0 Technology Stack

Use the same architectural direction as TradingAgents where appropriate.

```text
Language: Python
Orchestration: LangGraph
CLI: Typer
Dependency manager: uv
Schemas: Pydantic
Prompts: Markdown files
Storage: Filesystem only
Final report: Markdown
```

LLM providers:

```text
OpenAI API
DeepSeek API
```

Environment variables:

```env
OPENAI_API_KEY=
DEEPSEEK_API_KEY=

DEFAULT_LLM_PROVIDER=openai
DEFAULT_LLM_MODEL=
DEEPSEEK_MODEL=deepseek-chat

BRAVE_SEARCH_API_KEY=
```

DuckDuckGo may be implemented through a suitable library or integration. Prefer Brave Search first if API reliability is better.

## 4. v0 Scope

v0 must be usable, not only scaffolding.

Build order:

```text
Milestone B:
- repo scaffolding
- schemas
- CLI
- file output structure
- config placeholders
- tests

Milestone C:
- minimal real end-to-end run
- real search integration
- Research → Evidence → Opportunity
- then expand to full graph
```

Do not build in v0:

```text
Docker
web dashboard
database
auth
scheduler
background monitoring
full-page crawling
login-gated scraping
multi-user support
```

## 5. Run Boundary

One run investigates:

```text
one city + one community
```

The domain is not supplied. The system discovers domains from research.

Example:

```bash
maker run --city "Łodz" --community "senior citizens"
```

The output is a ranked Markdown report containing opportunities where value can be added.

Each opportunity must include:

```text
pain summary
who benefits
Maker argument
Taker argument
Mediator summary
validity score
Maker score/confidence
Taker score/confidence
impact estimate
POC cost estimate
recommended next action
Do No Harm section
evidence references
```

## 6. Agent Graph

Graph topology:

```text
Research Agent
  → Evidence Agent
    → Opportunity Agent
      → Maker Agent / Taker Agent in parallel per opportunity
        → Mediator Agent
          → Cost Checker Agent
            → Report Agent
```

Multiple opportunities may be processed concurrently.

Default max opportunities:

```text
5
```

Configurable with:

```bash
maker run --city "Łodz" --community "senior citizens" --max-opportunities 10
```

Failed agents must not fail the whole run. Failed opportunities are marked `INCOMPLETE` and can be retried.

Retry command:

```bash
maker retry runs/<run-id> --opportunity <opportunity-slug>
```

## 7. Agents

### 7.1 Research Agent

Responsibilities:

```text
- generate search queries from city + community
- include local language queries based on city/location
- search for pain points
- search for existing help/interventions
- call Brave Search and DuckDuckGo
- store snippets, links, and raw search result payloads
```

Defaults:

```text
search queries per run: 10
results per query: 5
```

Both are configurable.

For Łodz, queries should include English and Polish.

The Research Agent does not perform full-page crawling in v0.

### 7.2 Evidence Agent

Responsibilities:

```text
- classify evidence
- deduplicate evidence
- assign evidence IDs
- apply source trust scores
- calculate validity score
- detect conflict between official and community evidence
```

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

Every evidence item must include:

```text
id
source URL
source domain
source type
snippet
language
claim classification
trust score
recency
confidence
```

### 7.3 Opportunity Agent

Responsibilities:

```text
- convert evidence into candidate opportunities
- require at least 2 sources for normal validity
- allow weak-evidence opportunities if potential impact is high
- mark weak opportunities as speculative
- identify who benefits
- identify affected vulnerable groups where applicable
```

Opportunity types may include:

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

### 7.4 Maker Agent

Responsibilities:

```text
- argue where value can be added
- score value-add potential from 0–100
- provide confidence: low | medium | high
- cite evidence IDs
- classify claims as evidence_based, inference, assumption, or unknown
```

### 7.5 Taker Agent

Responsibilities:

```text
- identify how the opportunity could be exploited
- identify extraction risk, gatekeeping risk, false authority risk, dependency risk, and harm risk
- score exploitability risk from 0–100
- provide confidence: low | medium | high
- cite evidence IDs
```

The Taker Agent must not provide instructions for exploitation. It is a defensive red-team agent only.

### 7.6 Mediator Agent

Responsibilities:

```text
- compare Maker and Taker arguments
- summarize value-add vs value-take
- state when evidence is too weak
- assign verdict
- include Do No Harm section
- recommend safe intervention shape
```

### 7.7 Cost Checker Agent

Responsibilities:

```text
- estimate POC type
- estimate money cost in USD
- estimate time cost
- estimate risk level
- provide first 3 actions
```

POC types:

```text
manual_service
public_guide
dashboard
automation
advocacy_report
software_prototype
coordination_process
open_data_resource
```

Example:

```text
POC cost: $50–$300
Time: 1–2 weekends
Risk: medium
Type: public_guide + verified community report
First actions:
1. Collect 20 verified complaints.
2. Draft public guide.
3. Publish wait-time/reporting form.
```

### 7.8 Report Agent

Responsibilities:

```text
- produce final ranked Markdown report
- include formula used for ranking
- include only evidence references, not full evidence inline
- include rejected opportunities in appendix only
- clearly mark incomplete opportunities
- clearly mark user overrides
```

## 8. Source Collection

Allowed v0 sources:

```text
web search
news
Reddit/forums
government sites
public social sources where accessible
```

Rules:

```text
- avoid login-gated scraping
- no full-page crawling in v0
- collect snippets and links only
- save raw search result payloads locally
- use allowlist/blocklist per run
- maintain global source trust registry
```

## 9. Source Trust Registry

Global file:

```text
source-registry.yaml
```

Default trust model:

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

User may edit trust scores manually.

Final report must show top evidence sources per opportunity.

## 10. Evidence Validity

Minimum normal validity:

```text
2 independent sources
```

Anonymous posts are allowed but very low confidence.

Community evidence can outweigh official statements, but conflict must be highlighted explicitly.

Validity model:

```text
validity_score =
  source_count
  + source_trust
  + recency
  + corroboration
  - conflict_penalty
```

User can override validity score manually in Markdown files. Overrides must be clearly marked.

## 11. Scoring

Scores per opportunity:

```text
validity_score: 0–100
maker_score: 0–100
maker_confidence: low | medium | high
taker_score: 0–100
taker_confidence: low | medium | high
people_helped_score: 0–100
severity_score: 0–100
impact_score: 0–100
intervention_ease_score: 0–100
harm_risk_score: 0–100
ability_to_act_score: 0–100
rank_score: 0–100
```

Taker score means exploitability risk. Higher is worse.

Taker score is shown separately and does not directly reduce rank score.

Ranking formula:

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

Ranking priority:

```text
1. number of people helped
2. severity of pain
3. evidence validity
4. ease of intervention
5. low risk of harm
6. personal ability to act
```

Commercial and non-commercial opportunities rank equally.

POC cost is informational only.

## 12. Verdicts

Allowed verdicts:

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

The system must never auto-select action. It only recommends.

## 13. Safety Boundaries

Illegal solutions are not opportunities.

Morally risky ideas may be analyzed but must be clearly flagged.

Every opportunity must include a Do No Harm section.

Do No Harm must cover:

```text
vulnerable groups affected
possible negative side effects
abuse/exploitation risks
legal/terms-of-service concerns
trust and misinformation risks
dependency risks
gatekeeping risks
false authority risks
safeguards required before POC
```

Example:

```text
Opportunity: Polish visa appointment tracker

Do No Harm:
- Must not hoard appointment slots.
- Must not increase load on embassy systems.
- Must not help brokers/resellers exploit applicants.
- Must not present itself as official.
- Must not collect sensitive passport/visa data unless strictly necessary.
- Must clearly state limits and uncertainty.
```

## 14. Prompt Discipline

Every agent prompt must enforce:

```text
- no unsupported claims
- no invented sources
- every claim classified as:
  - evidence_based
  - inference
  - assumption
  - unknown
- every claim cites evidence IDs where possible
- state uncertainty explicitly
- do not hide weak evidence
```

Taker Agent must be restricted to risk identification only.

## 15. Storage Layout

Each run creates a timestamped folder:

```text
runs/<run-id>/
```

Suggested structure:

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

`README.md` per opportunity must contain a short human-readable summary.

Incomplete opportunities must be resumable from disk.

Retry must not rerun the entire research phase.

## 16. Final Report Shape

`final-report.md` is the ranked executive summary.

It must include:

```text
city
community
run timestamp
ranking formula
ranked opportunities
top evidence sources
Maker/Taker scores
validity scores
POC cost estimates
recommended next actions
appendix references
```

Each ranked opportunity includes:

```text
rank
title
pain summary
who benefits
Maker score/confidence
Taker score/confidence
validity score
impact estimate
intervention ease
harm risk
ability to act
Mediator summary
POC cost estimate
recommended next action
Do No Harm summary
evidence references
```

Rejected opportunities appear only in appendix.

## 17. User Overrides

User may override in Markdown files:

```text
validity score
ranking
verdict
source trust score
```

Overrides must be clearly marked.

Example:

```markdown
> User Override:
> validity_score changed from 61 to 75.
> Reason: user manually confirmed source relevance.
```

## 18. CLI Commands

Required v0 commands:

```bash
maker run --city "Łodz" --community "senior citizens"
maker run --city "Łodz" --community "senior citizens" --max-opportunities 10
maker retry runs/<run-id> --opportunity <opportunity-slug>
```

Useful additional commands:

```bash
maker sources list
maker sources trust <domain> --score 75
maker report runs/<run-id>
```

## 19. Testing Requirements

No offline/stub LLM mode is required for v0, but code must be tested.

Tests should cover:

```text
CLI argument parsing
run folder creation
source registry loading
source trust scoring
evidence schema validation
opportunity schema validation
agent output parsing
rank score calculation
retry state detection
final report generation
```

Tests should not require real API calls unless explicitly marked integration tests.

## 20. Acceptance Criteria for v0

v0 is acceptable when this command works:

```bash
maker run --city "Łodz" --community "senior citizens"
```

And produces:

```text
runs/<timestamp>-lodz-senior-citizens/
  run.yaml
  final-report.md
  sources/search-results.json
  evidence/evidence.json
  opportunities/<slug>/README.md
  opportunities/<slug>/maker.md
  opportunities/<slug>/taker.md
  opportunities/<slug>/mediator.md
  opportunities/<slug>/cost.md
```

The final report must contain at least one ranked opportunity if sufficient evidence is found.

If no valid opportunities are found, the report must explain:

```text
what was searched
what sources were found
why evidence was insufficient
recommended next search direction
```

## 21. Engineering Principles

```text
Boring architecture.
Strict schemas.
Local-first output.
Evidence before argument.
No invented sources.
No exploitation instructions.
No illegal interventions.
Reports must be auditable.
Failures must be resumable.
The system exists to help Makers add more value than they take.
```

