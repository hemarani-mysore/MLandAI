"""Seeded-error detection: does the real `red_team` node actually catch a
planted unsupported claim or contradiction in the bull/bear cases?

    uv run python evals/redteam_eval.py             # live model, report
    uv run python evals/redteam_eval.py --gate       # ALWAYS offline (see below)

`--gate` never makes a network call, regardless of `PRAXIS_LLM_PROVIDER`: it
drives `FakeStructuredLLM` with real verdicts recorded once against an actual
model (`evals/cassettes/redteam_test.json`, captured by
`evals/cassettes/_record_redteam.py`). Without `--gate` it calls the live
`red_team` node via `get_llm()` — useful for calibrating a prompt change
before re-recording the cassette.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from praxis.graph.nodes.red_team import red_team
from praxis.graph.state import initial_state
from praxis.llm import FakeStructuredLLM, get_llm
from praxis.schemas import Citation, Evidence, Finding, RedTeamReport

HERE = Path(__file__).parent
CASSETTE = HERE / "cassettes" / "redteam_test.json"
REDTEAM_FLOOR = 0.6


def _ev(question: str, section: str, claim: str, source_id: str) -> Evidence:
    return Evidence(
        question=question,
        section=section,
        claim=claim,
        stance="neutral",
        citation=Citation(source_id=source_id, title=source_id, locator="p.1", quote=claim),
        confidence=0.8,
    )


# Each fixture plants exactly one false/contradictory claim in the bull or
# bear case that the real evidence does not support — `planted_keyword` is
# what a real red_team pass should name in `unsupported_claims`/`contradictions`.
FIXTURES = [
    {
        "id": "fabricated-funding-round",
        "subject": "Acme Robotics",
        "evidence": [
            _ev(
                "What is Acme's funding history?",
                "Team & Funding",
                "Acme raised a $12M Series A in 2022.",
                "acme-overview",
            ),
            _ev(
                "What is Acme's ARR?",
                "Traction & Financials",
                "Acme's ARR is $3M, growing 40% YoY.",
                "acme-overview",
            ),
        ],
        "bull": Finding(
            stance="bull",
            summary="Strong growth trajectory.",
            points=[
                "Acme secured a $50M Series C round, well beyond typical for its stage.",
                "ARR growth of 40% YoY is healthy for this stage.",
            ],
            evidence_refs=[0, 1],
        ),
        "bear": Finding(
            stance="bear",
            summary="Limited visibility on retention.",
            points=["Public information on customer retention is limited."],
            evidence_refs=[],
        ),
        "planted_keyword": "$50M Series C",
    },
    {
        "id": "contradicted-profitability",
        "subject": "Globex Manufacturing",
        "evidence": [
            _ev(
                "Is Globex profitable?",
                "Traction & Financials",
                "Globex has been profitable on an EBITDA basis since 2021.",
                "globex-financials",
            ),
        ],
        "bull": Finding(
            stance="bull",
            summary="Durable margins.",
            points=["Consistent profitability since 2021 supports a stable outlook."],
            evidence_refs=[0],
        ),
        "bear": Finding(
            stance="bear",
            summary="Recent losses raise concern.",
            points=["Globex reported net losses of $10M last quarter, a concerning reversal."],
            evidence_refs=[],
        ),
        "planted_keyword": "net losses",
    },
    {
        "id": "contradicted-churn",
        "subject": "Nimbus Cloud Storage",
        "evidence": [
            _ev(
                "What is Nimbus's churn rate?",
                "Traction & Financials",
                "Nimbus reports a 15% annual customer churn rate.",
                "nimbus-metrics",
            ),
        ],
        "bull": Finding(
            stance="bull",
            summary="Sticky product.",
            points=["Nimbus has zero customer churn, indicating exceptional product-market fit."],
            evidence_refs=[0],
        ),
        "bear": Finding(
            stance="bear",
            summary="Some competitive pressure.",
            points=["The storage market has several well-funded competitors."],
            evidence_refs=[],
        ),
        "planted_keyword": "churn",
    },
    {
        "id": "contradicted-patents",
        "subject": "Vertex Materials",
        "evidence": [
            _ev(
                "Does Vertex hold any patents?",
                "Technology & IP",
                "Vertex holds 3 issued patents on its core alloy process.",
                "vertex-ip-filing",
            ),
        ],
        "bull": Finding(
            stance="bull",
            summary="Solid fundamentals.",
            points=["Revenue growth has been steady over the past two years."],
            evidence_refs=[],
        ),
        "bear": Finding(
            stance="bear",
            summary="Weak defensibility.",
            points=["Vertex has no patents, leaving its process easily replicable by competitors."],
            evidence_refs=[],
        ),
        "planted_keyword": "patent",
    },
]


def _state_for(fixture: dict) -> dict:
    state = initial_state(fixture["subject"], "standard")
    state["evidence"] = fixture["evidence"]
    state["bull"] = fixture["bull"]
    state["bear"] = fixture["bear"]
    state["iteration"] = 0
    return state


async def _run_one(llm, fixture: dict) -> RedTeamReport:
    state = _state_for(fixture)
    result = await red_team(state, {"configurable": {"llm": llm}})
    return result["redteam"]


def _detected(report: RedTeamReport, keyword: str) -> bool:
    haystack = " ".join(report.unsupported_claims + report.contradictions).lower()
    return keyword.lower() in haystack


def _load_cassette() -> dict[str, dict]:
    return json.loads(CASSETTE.read_text())


def _llm_for_gate(gate: bool):
    if not gate:
        return get_llm()
    cassette = _load_cassette()
    fixtures = sorted(FIXTURES, key=lambda f: f["id"])
    missing = [f["id"] for f in fixtures if f["id"] not in cassette]
    if missing:
        raise KeyError(f"evals/cassettes/redteam_test.json is missing verdicts for: {missing}")
    overrides = {"red_team": [RedTeamReport(**cassette[f["id"]]) for f in fixtures]}
    return FakeStructuredLLM(overrides=overrides)


async def evaluate(*, gate: bool) -> dict:
    llm = _llm_for_gate(gate)
    fixtures = sorted(FIXTURES, key=lambda f: f["id"])
    rows = []
    for fixture in fixtures:
        report = await _run_one(llm, fixture)
        detected = _detected(report, fixture["planted_keyword"])
        rows.append(
            {
                "id": fixture["id"],
                "planted_keyword": fixture["planted_keyword"],
                "detected": detected,
                "unsupported_claims": report.unsupported_claims,
                "contradictions": report.contradictions,
            }
        )
    rate = sum(r["detected"] for r in rows) / len(rows) if rows else 0.0
    return {"n": len(rows), "seeded_error_detection_rate": rate, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    import asyncio

    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true", help="exit non-zero if below the floor")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = asyncio.run(evaluate(gate=args.gate))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"redteam eval — {report['n']} seeded-error fixtures\n")
        print(f"  seeded_error_detection_rate  {report['seeded_error_detection_rate']:.3f}\n")
        for row in report["rows"]:
            mark = "ok " if row["detected"] else "MISSED"
            print(f"  {mark} {row['id']:<28} planted={row['planted_keyword']!r}")

    if args.gate:
        rate = report["seeded_error_detection_rate"]
        if rate < REDTEAM_FLOOR:
            print(f"\nGATE FAILED: rate={rate:.3f} < floor={REDTEAM_FLOOR}", file=sys.stderr)
            return 1
        print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
