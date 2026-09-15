"""One-off: capture real judge verdicts for the `test` split against an
actual model, saved to `judge_test.json` and replayed offline (zero network
calls) by `memo_eval.py --split test`. Re-run this whenever `evals/metric.py`'s
`JUDGE_PROMPT` changes meaningfully — the cassette is a frozen snapshot, not a
live contract.

    uv run python evals/cassettes/_record_judge_test.py    # needs a real key
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
EVALS = HERE.parent
GOLDEN_DIR = EVALS / "datasets" / "golden_dossiers"
for _p in (EVALS, GOLDEN_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import loader as golden  # noqa: E402
from metric import BinaryLLMJudgeMetric  # noqa: E402

from praxis.config import get_settings  # noqa: E402
from praxis.llm import get_llm  # noqa: E402


def main() -> int:
    if get_settings().llm_provider == "fake":
        print("set PRAXIS_LLM_PROVIDER to a real provider before recording", file=sys.stderr)
        return 2

    items = sorted(golden.load_index(split="test"), key=lambda i: i.id)
    judge = BinaryLLMJudgeMetric(get_llm())
    verdicts: dict[str, dict[str, str]] = {}
    for item in items:
        result = judge.run(
            subject=item.id.replace("-", " ").title(),
            sources_text=item.read_sources(),
            memo_text=item.read_memo(),
        )
        verdicts[item.id] = {"label": result.label, "critique": result.critique}
        agree = "agrees" if result.label == item.label else "DISAGREES"
        print(f"{item.id}: expert={item.label} judge={result.label} ({agree}) — {result.critique}")

    out = HERE / "judge_test.json"
    out.write_text(json.dumps(verdicts, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
