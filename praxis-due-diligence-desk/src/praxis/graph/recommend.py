"""Derive the overall recommendation + confidence from the gathered state.

Deterministic on purpose: the editor writes the prose, but the verdict is a
transparent function of the evidence balance and the red team's findings, so it
is calibrated the same way for every provider and is unit-testable without an
LLM. Phase 4's eval framework scores this against expert labels.
"""

from __future__ import annotations

from praxis.graph.state import DossierState
from praxis.schemas import PLACEHOLDER_SOURCE_IDS, RUBRIC_SECTIONS, Recommendation


def derive_recommendation(state: DossierState) -> tuple[Recommendation, float]:
    evidence = state["evidence"]
    grounded = [ev for ev in evidence if ev.citation.source_id not in PLACEHOLDER_SOURCE_IDS]
    rubric_n = len(RUBRIC_SECTIONS)

    # Not enough real evidence to take a position.
    if len(grounded) < rubric_n / 2:
        conf = 0.2 + 0.1 * (len(grounded) / rubric_n)
        return "insufficient_evidence", round(min(conf, 0.35), 2)

    supports = sum(1 for ev in grounded if ev.stance == "supports")
    refutes = sum(1 for ev in grounded if ev.stance == "refutes")

    redteam = state["redteam"]
    contradictions = len(redteam.contradictions) if redteam else 0
    unsupported = len(redteam.unsupported_claims) if redteam else 0
    severity = contradictions * 2 + unsupported

    bull_pts = len(state["bull"].points) if state["bull"] else 0
    bear_pts = len(state["bear"].points) if state["bear"] else 0

    if contradictions or refutes > supports or bear_pts > 2 * max(bull_pts, 1):
        rec: Recommendation = "pass"
    elif severity >= 2 or refutes >= 1 or bear_pts >= bull_pts:
        rec = "proceed_with_conditions"
    else:
        rec = "proceed"

    covered = {ev.section for ev in grounded} & set(RUBRIC_SECTIONS)
    coverage = len(covered) / rubric_n
    total_stance = supports + refutes
    agreement = abs(supports - refutes) / total_stance if total_stance else 0.0

    conf = 0.35 + 0.4 * coverage + 0.15 * agreement - 0.05 * severity
    if total_stance == 0:
        conf = min(conf, 0.5)  # coverage without a directional read
    return rec, round(max(0.2, min(0.9, conf)), 2)
