"""One-off: capture real `red_team` verdicts for each seeded-error fixture
against an actual model, saved to `redteam_test.json` and replayed offline
(zero network calls) by `redteam_eval.py --gate`. Re-run this whenever
`graph/prompts.py`'s `RED_TEAM` prompt changes meaningfully.

    uv run python evals/cassettes/_record_redteam.py    # needs a real key
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
EVALS = HERE.parent
if str(EVALS) not in sys.path:
    sys.path.insert(0, str(EVALS))
import redteam_eval as rte  # noqa: E402

from praxis.config import get_settings  # noqa: E402
from praxis.llm import get_llm  # noqa: E402


async def main() -> int:
    if get_settings().llm_provider == "fake":
        print("set PRAXIS_LLM_PROVIDER to a real provider before recording", file=sys.stderr)
        return 2

    llm = get_llm()
    fixtures = sorted(rte.FIXTURES, key=lambda f: f["id"])
    verdicts: dict[str, dict] = {}
    for fixture in fixtures:
        report = await rte._run_one(llm, fixture)
        detected = rte._detected(report, fixture["planted_keyword"])
        verdicts[fixture["id"]] = report.model_dump(mode="json")
        mark = "detected" if detected else "MISSED"
        print(f"{fixture['id']}: {mark} (planted: {fixture['planted_keyword']!r})")
        print(f"  unsupported_claims: {report.unsupported_claims}")
        print(f"  contradictions:     {report.contradictions}")

    out = HERE / "redteam_test.json"
    out.write_text(json.dumps(verdicts, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
