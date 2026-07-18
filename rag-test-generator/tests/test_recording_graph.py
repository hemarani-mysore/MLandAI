"""
tests/test_recording_graph.py — Unit tests for agents/recording_graph.py
==========================================================================
Tests the conditional routing logic and scenario-name slug generation.
OpenAI calls and browser/subprocess execution are not tested here.

Run with:
    pytest tests/test_recording_graph.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.recording_graph import _route
from main import _slugify_scenario_name


# ─────────────────────────────────────────────
# _route  (conditional edge logic)
# ─────────────────────────────────────────────

def _state(execution_passed=False, give_up=False, attempt=1, max_attempts=5):
    return {
        "execution_passed": execution_passed,
        "give_up"         : give_up,
        "attempt"         : attempt,
        "max_attempts"    : max_attempts,
    }


def test_route_done_when_passed():
    assert _route(_state(execution_passed=True)) == "done"


def test_route_done_when_llm_gives_up():
    assert _route(_state(execution_passed=False, give_up=True)) == "done"


def test_route_done_when_max_attempts_reached():
    assert _route(_state(execution_passed=False, attempt=5, max_attempts=5)) == "done"


def test_route_retry_when_failed_and_attempts_remain():
    assert _route(_state(execution_passed=False, give_up=False, attempt=2, max_attempts=5)) == "retry"


def test_route_passed_takes_priority_over_give_up():
    # Shouldn't happen in practice, but passed should win if both are somehow true
    assert _route(_state(execution_passed=True, give_up=True)) == "done"


# ─────────────────────────────────────────────
# _slugify_scenario_name
# ─────────────────────────────────────────────

def test_slug_uses_explicit_name():
    slug = _slugify_scenario_name("https://example.com/checkout", "My Checkout Flow")
    assert slug == "my-checkout-flow"


def test_slug_auto_derives_from_url_when_no_name():
    slug = _slugify_scenario_name("https://example.com/checkout", None)
    assert slug.startswith("example-com-checkout-")


def test_slug_strips_special_characters():
    slug = _slugify_scenario_name("https://example.com", "Login!! Flow??")
    assert slug == "login-flow"


def test_slug_never_empty():
    slug = _slugify_scenario_name("https://example.com", "!!!")
    assert slug != ""
