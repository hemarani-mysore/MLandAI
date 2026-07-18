"""
agents/recording_graph.py — Recording Pipeline Graph
======================================================
Wires the 5 recording-pipeline agents into a LangGraph StateGraph:

    START -> recorder -> spec_gen -> execute
                                        |
                        ┌───────────────┴───────────────┐
                        │ (conditional on execute result) │
                        ▼                                 ▼
                       fix ──────loop back───────────▶ execute
                                        │
                                       (done)
                                        ▼
                                     report -> END

This is the pipeline used by the `record` CLI command — separate from, and
additive to, the existing 5-agent ingest/embed/analyze/generate/push pipeline.
"""

import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.graph import StateGraph, START, END

from config import PLAYWRIGHT_RUNNER_DIR
from agents import recorder_agent, spec_gen_agent, execution_agent, fix_agent, report_agent


class RecordingState(TypedDict):
    url              : str
    scenario_name    : str
    raw_recording    : str
    spec_code        : str
    spec_path        : str
    attempt          : int
    max_attempts     : int
    execution_passed : bool
    execution_output : str
    dom_snapshot     : str
    give_up          : bool
    give_up_reason   : str
    attempts_history : list[dict]
    report_path      : str


def _spec_file_path(scenario_name: str) -> Path:
    return PLAYWRIGHT_RUNNER_DIR / "tests" / f"{scenario_name}.spec.js"


_DOM_CAPTURE_HOOK = """

test.afterEach(async ({ page }, testInfo) => {
  if (testInfo.status !== testInfo.expectedStatus) {
    try {
      const html = await page.content();
      await testInfo.attach('dom-snapshot', { body: html, contentType: 'text/html' });
    } catch (e) {
      // page may already be closed — ignore
    }
  }
});
"""


def _with_dom_capture_hook(spec_code: str) -> str:
    """
    Append a fixed afterEach hook that captures page HTML on failure as a
    Playwright attachment (read back by agents/execution_agent.py). This is
    injected mechanically rather than relied on from the LLM's prompt, so it's
    guaranteed to be present on every attempt regardless of what the LLM writes.
    """
    if "dom-snapshot" in spec_code:
        return spec_code  # already present (e.g. survived a fix-agent rewrite)
    return spec_code.rstrip() + "\n" + _DOM_CAPTURE_HOOK


# ─────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────

def _recorder_node(state: RecordingState) -> dict:
    raw_recording = recorder_agent.run(state["url"], state["scenario_name"])
    return {"raw_recording": raw_recording}


def _spec_gen_node(state: RecordingState) -> dict:
    spec_code = spec_gen_agent.run(state["raw_recording"], state["scenario_name"])
    spec_code = _with_dom_capture_hook(spec_code)
    spec_path = _spec_file_path(state["scenario_name"])
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(spec_code, encoding="utf-8")
    return {"spec_code": spec_code, "spec_path": str(spec_path), "attempt": 1}


def _execute_node(state: RecordingState) -> dict:
    result = execution_agent.run(Path(state["spec_path"]))
    history = state["attempts_history"] + [{
        "attempt_number": state["attempt"],
        "passed"        : result.passed,
        "error_summary" : result.error[:500],
    }]
    return {
        "execution_passed": result.passed,
        "execution_output": result.output if result.passed else result.error,
        "dom_snapshot"    : result.dom_snapshot,
        "attempts_history": history,
    }


def _fix_node(state: RecordingState) -> dict:
    next_attempt = state["attempt"] + 1
    fix = fix_agent.run(
        spec_code    = state["spec_code"],
        error_output = state["execution_output"],
        attempt      = next_attempt,
        max_attempts = state["max_attempts"],
        dom_snapshot = state.get("dom_snapshot", ""),
    )
    fixed_code = _with_dom_capture_hook(fix.fixed_code)
    spec_path = Path(state["spec_path"])
    spec_path.write_text(fixed_code, encoding="utf-8")
    return {
        "spec_code"     : fixed_code,
        "attempt"       : next_attempt,
        "give_up"       : fix.give_up,
        "give_up_reason": fix.reason,
    }


def _report_node(state: RecordingState) -> dict:
    report_path = report_agent.run(state)
    return {"report_path": str(report_path)}


def _route(state: RecordingState) -> str:
    """The one piece of pure routing logic — kept separate for easy unit testing."""
    if state["execution_passed"]:
        return "done"
    if state["give_up"] or state["attempt"] >= state["max_attempts"]:
        return "done"
    return "retry"


# ─────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(RecordingState)

    graph.add_node("recorder", _recorder_node)
    graph.add_node("spec_gen", _spec_gen_node)
    graph.add_node("execute", _execute_node)
    graph.add_node("fix", _fix_node)
    graph.add_node("report", _report_node)

    graph.add_edge(START, "recorder")
    graph.add_edge("recorder", "spec_gen")
    graph.add_edge("spec_gen", "execute")
    graph.add_conditional_edges("execute", _route, {"retry": "fix", "done": "report"})
    graph.add_edge("fix", "execute")
    graph.add_edge("report", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────
# Quick test — run directly
# python agents/recording_graph.py <url> [scenario-name] [max-attempts]
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    url          = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    scenario     = sys.argv[2] if len(sys.argv) > 2 else "manual-test"
    max_attempts = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    initial_state: RecordingState = {
        "url"              : url,
        "scenario_name"    : scenario,
        "raw_recording"    : "",
        "spec_code"        : "",
        "spec_path"        : "",
        "attempt"          : 0,
        "max_attempts"     : max_attempts,
        "execution_passed" : False,
        "execution_output" : "",
        "dom_snapshot"     : "",
        "give_up"          : False,
        "give_up_reason"   : "",
        "attempts_history" : [],
        "report_path"      : "",
    }

    graph = build_graph()
    final_state = graph.invoke(initial_state)
    print(f"\nFinal report: {final_state['report_path']}")
