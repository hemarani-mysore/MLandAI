"""LLM-as-judge: does a due-diligence memo hold up against its own sources?

`BinaryLLMJudgeMetric` reads `(subject, source material, memo text)` and
returns a structured pass/fail verdict + a short critique. This is *not*
reference-free in the sense of ignoring the sources — a memo can be
internally coherent and still fail if it misrepresents or omits something the
sources plainly state, or invents specifics they never gave it. Catching
either failure mode requires the sources, not just the memo — see
`evals/datasets/golden_dossiers/README.md` for two worked examples
(`driftwood-materials`, `ferrovia-rail-tech`).

There is no separate hand-authored "gold" reference-memo mode: the dataset's
expert `label` (`evals/datasets/golden_dossiers/index.yaml`) is the ground
truth the judge is scored against — see `memo_eval.py`.

`JudgeResult` lives here, not in `src/praxis/schemas.py` — it's an eval-only
concern; `evals/` imports from `praxis`, never the reverse.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from praxis.llm import StructuredLLM
from praxis.schemas import RUBRIC_SECTIONS

JudgeLabel = Literal["pass", "fail"]


class JudgeResult(BaseModel):
    label: JudgeLabel
    critique: str = Field(description="1-3 sentences naming the specific reasoning, not a summary")


JUDGE_PROMPT = f"""\
You are a skeptical senior due-diligence reviewer auditing a memo a junior \
analyst produced. You are given the analyst's SOURCE MATERIAL (the only facts \
they had) and the MEMO they wrote from it. Decide pass or fail.

Naming a real risk and recommending `proceed_with_conditions` on it is \
GOOD, correct behaviour, not a reason to fail a memo — that is exactly what \
`proceed_with_conditions` is for. Do not fail a memo merely because a risk \
exists; fail it only for how the memo handles that risk.

Base your verdict ONLY on concrete, checkable facts — specific numbers, \
named events, and entities — not on subjective tone or adjective choice. If \
the memo states the same specific number, event, or fact the sources do \
(even in fewer or plainer words), that risk has NOT been downplayed. Only \
call something downplayed if a specific fact the sources state (a number, an \
event, a named entity) is *absent* from the memo, or is materially altered —
e.g. the sources name a percentage or a specific incident and the memo \
converts it into a vague generality with no comparable specific in its \
place. Word choice and tone are never themselves a failure reason.

A memo FAILS if it:
- recommends `proceed` (not `proceed_with_conditions`) while a specific, \
material fact from the sources — a number, a named event — is completely \
absent from the memo, or is stated in the memo with no comparable specific \
where the sources gave one;
- states a specific fact, figure, or claim the sources do not support;
- recommends confidently despite sources too thin to support any firm \
recommendation (a defensible memo would have said `insufficient_evidence` \
instead).

A memo PASSES if every specific fact the sources raise about a risk (the \
actual numbers and named events) also appears in the memo, claims are \
traceable to the sources, and the recommendation tier (`proceed` / \
`proceed_with_conditions` / `pass` / `insufficient_evidence`) is proportionate \
to those facts — `proceed_with_conditions` that restates a real risk's actual \
numbers/events and conditions on it is a textbook pass, not a borderline \
case. A pass does not require the company to be risk-free, and does not \
require the memo to editorialize about how serious a risk is beyond stating \
its actual facts.

A complete memo covers these rubric sections: {", ".join(RUBRIC_SECTIONS)}.

Give a 1-3 sentence critique naming the specific claim, omission, or \
overreach that drove your verdict — never a generic summary.
"""


def _user_prompt(subject: str, sources_text: str, memo_text: str) -> str:
    return f"Subject: {subject}\n\nSOURCE MATERIAL:\n{sources_text}\n\nMEMO TO AUDIT:\n{memo_text}"


class BinaryLLMJudgeMetric:
    """Runs on the `judge` role (cheap tier — see `config.py`'s `fast_roles`).

    `PRAXIS_LLM_PROVIDER=fake`'s `FakeStructuredLLM` has no canned
    `JudgeResult` and will raise `ValueError` if asked for one directly (i.e.
    outside a `FakeStructuredLLM(overrides={"judge": [...]})` script) — this
    is deliberate: a meaningless "fake" calibration number would be worse
    than a clear error telling you no real judge ran.
    """

    def __init__(self, llm: StructuredLLM) -> None:
        self._llm = llm

    def run(self, *, subject: str, sources_text: str, memo_text: str) -> JudgeResult:
        return self._llm.generate(
            system=JUDGE_PROMPT,
            user=_user_prompt(subject, sources_text, memo_text),
            schema=JudgeResult,
            role="judge",
        )

    async def arun(self, *, subject: str, sources_text: str, memo_text: str) -> JudgeResult:
        return await self._llm.agenerate(
            system=JUDGE_PROMPT,
            user=_user_prompt(subject, sources_text, memo_text),
            schema=JudgeResult,
            role="judge",
        )
