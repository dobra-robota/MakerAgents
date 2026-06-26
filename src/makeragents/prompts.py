"""Markdown prompt loader with variable substitution and discipline scaffolding.

Each generative agent loads its prompt from a packaged Markdown file rather
than an inline string. The loader applies simple ``${variable}`` substitution
and appends the shared prompt-discipline clauses from PRD §14.

PRD: §3, §14.
"""

from __future__ import annotations

from pathlib import Path
from string import Template

# ---------------------------------------------------------------------------
# Shared discipline scaffolding (PRD §14)
# ---------------------------------------------------------------------------

_SHARED_DISCIPLINE = """

## Prompt Discipline (PRD §14)

- Do not make unsupported claims.
- Do not invent sources.
- Classify every claim as one of: `evidence_based`, `inference`, `assumption`, `unknown`.
- Cite evidence IDs where possible.
- State uncertainty explicitly; do not hide weak evidence.
- If you do not know something, say so.
"""

_TAKER_SAFETY_DISCIPLINE = """

## CRITICAL SAFETY CONSTRAINT (PRD §7.5, §14)

- Identify exploitability risks **only**.
- Do **NOT** provide exploitation instructions, attack vectors, or step-by-step abuse methods.
- Frame everything as defensive analysis: what could go wrong, who could exploit, and what safeguards exist.
- If you cannot assess a risk without describing how to exploit it, mark it as `unknown` and move on.
- The output must contain **zero** actionable exploitation content.
"""

# ---------------------------------------------------------------------------
# Prompt directory
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).parent / "data" / "prompts"

# Agents whose prompts should receive the Taker safety addendum.
_TAKER_AGENTS = frozenset({"taker"})


def load_prompt(name: str, **variables: str) -> str:
    """Load a prompt Markdown file and substitute variables.

    Args:
        name: The prompt file name without ``.md`` extension
            (e.g. ``"research"``, ``"evidence"``).
        **variables: Keyword arguments are substituted into the
            template using ``${key}`` syntax.

    Returns:
        The rendered prompt string with shared discipline clauses appended.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(
            f"Prompt file not found: {name}.md (looked in {_PROMPTS_DIR})"
        )

    template_content = path.read_text(encoding="utf-8")

    # Append shared discipline.
    template_content += _SHARED_DISCIPLINE

    # Append Taker safety discipline if applicable.
    if name in _TAKER_AGENTS:
        template_content += _TAKER_SAFETY_DISCIPLINE

    return Template(template_content).safe_substitute(**variables)
