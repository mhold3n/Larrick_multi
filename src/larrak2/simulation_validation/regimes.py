"""Canonical regime definitions and prerequisite ordering."""

from __future__ import annotations

import enum


class CanonicalRegime(str, enum.Enum):  # noqa: UP042
    """Five canonical validation layers, ordered from simplest to most integrated."""

    CHEMISTRY = "chemistry"
    SPRAY = "spray"
    REACTING_FLOW = "reacting_flow"
    CLOSED_CYLINDER = "closed_cylinder"
    FULL_HANDOFF = "full_handoff"

    @classmethod
    def ordered(cls) -> list[CanonicalRegime]:
        """Return regimes in canonical validation order."""
        return [
            cls.CHEMISTRY,
            cls.SPRAY,
            cls.REACTING_FLOW,
            cls.CLOSED_CYLINDER,
            cls.FULL_HANDOFF,
        ]

    @classmethod
    def ordered_names(cls) -> list[str]:
        """Return canonical regime names in order."""
        return [regime.value for regime in cls.ordered()]


# Prerequisite DAG: key depends on all values passing first.
PREREQUISITE_MAP: dict[CanonicalRegime, list[CanonicalRegime]] = {
    CanonicalRegime.CHEMISTRY: [],
    CanonicalRegime.SPRAY: [],
    CanonicalRegime.REACTING_FLOW: [CanonicalRegime.CHEMISTRY],
    CanonicalRegime.CLOSED_CYLINDER: [],
    CanonicalRegime.FULL_HANDOFF: [
        CanonicalRegime.SPRAY,
        CanonicalRegime.REACTING_FLOW,
        CanonicalRegime.CLOSED_CYLINDER,
    ],
}


def canonical_prerequisite_names() -> dict[str, list[str]]:
    """Return the canonical prerequisite DAG keyed by regime name."""
    return {
        regime.value: [prereq.value for prereq in prereqs]
        for regime, prereqs in PREREQUISITE_MAP.items()
    }
