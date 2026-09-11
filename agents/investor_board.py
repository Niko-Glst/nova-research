"""Archetype-persona investor board.

Simulates a panel of fictional investor *archetypes* debating the inputs from
upstream agents (TA, FA, sentiment). Archetypes are deliberately generic and
NOT modeled on any specific real individual investor — see SECURITY.md /
project conventions: use names like "ValueArchetype", "MacroArchetype", etc.

TODO:
- Implement each archetype's "opinion" function using whatever LLM/rules
  backend the project settles on.
- Implement aggregation of the board's opinions into a structured output that
  narrative_synth.py can consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ArchetypeName(str, Enum):
    VALUE = "ValueArchetype"
    MACRO = "MacroArchetype"
    MOMENTUM = "MomentumArchetype"
    CONTRARIAN = "ContrarianArchetype"
    GROWTH = "GrowthArchetype"


@dataclass(frozen=True)
class ArchetypeOpinion:
    archetype: ArchetypeName
    stance: str  # e.g. "bullish" / "bearish" / "neutral"
    reasoning: str
    confidence: float  # 0.0 - 1.0


@dataclass(frozen=True)
class BoardVerdict:
    opinions: list[ArchetypeOpinion]
    consensus_stance: str
    dissent_summary: str


def get_archetype_opinion(archetype: ArchetypeName, inputs: dict) -> ArchetypeOpinion:
    """Produce a single archetype's opinion given upstream analysis inputs.

    TODO: implement per-archetype reasoning.
    """
    raise NotImplementedError


def convene_board(inputs: dict) -> BoardVerdict:
    """Run all archetypes and aggregate into a board verdict.

    TODO: implement aggregation logic (e.g. weighted vote, majority stance).
    """
    raise NotImplementedError
