"""
tests/test_execution_agent.py — Unit tests for agents/execution_agent.py
==========================================================================
Tests the JSON-reporter-parsing logic against canned Playwright JSON reporter
output. No subprocess or real Playwright execution involved.

Run with:
    pytest tests/test_execution_agent.py -v
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.execution_agent import _parse_reporter_json


def _reporter_json(*, expected=0, unexpected=0, suites=None, errors=None) -> str:
    return json.dumps({
        "stats" : {"expected": expected, "unexpected": unexpected, "skipped": 0, "flaky": 0},
        "suites": suites or [],
        "errors": errors or [],
    })


def test_all_passed():
    raw = _reporter_json(expected=1, unexpected=0)
    result = _parse_reporter_json(raw)
    assert result.passed is True
    assert result.error == ""


def test_no_tests_ran_is_not_passed():
    # expected=0, unexpected=0 — nothing actually ran, shouldn't be reported as "passed"
    raw = _reporter_json(expected=0, unexpected=0)
    result = _parse_reporter_json(raw)
    assert result.passed is False


def test_failed_test_extracts_error_message():
    suites = [{
        "specs": [{
            "title": "checkout flow",
            "tests": [{
                "results": [{
                    "status": "failed",
                    "error": {"message": "Timeout 30000ms exceeded waiting for selector #submit"},
                }],
            }],
        }],
    }]
    raw = _reporter_json(expected=0, unexpected=1, suites=suites)
    result = _parse_reporter_json(raw)
    assert result.passed is False
    assert "checkout flow" in result.error
    assert "Timeout 30000ms exceeded" in result.error


def test_nested_describe_suites_are_walked():
    suites = [{
        "specs": [],
        "suites": [{
            "specs": [{
                "title": "nested test",
                "tests": [{
                    "results": [{"status": "failed", "error": {"message": "boom"}}],
                }],
            }],
        }],
    }]
    raw = _reporter_json(expected=0, unexpected=1, suites=suites)
    result = _parse_reporter_json(raw)
    assert "nested test" in result.error
    assert "boom" in result.error


def test_top_level_fatal_error_is_captured():
    raw = _reporter_json(expected=0, unexpected=0, errors=[{"message": "SyntaxError: Unexpected token"}])
    result = _parse_reporter_json(raw)
    assert result.passed is False
    assert "SyntaxError" in result.error


def test_failed_with_no_message_has_fallback_text():
    suites = [{
        "specs": [{
            "title": "flaky test",
            "tests": [{"results": [{"status": "failed", "error": {}}]}],
        }],
    }]
    raw = _reporter_json(expected=0, unexpected=1, suites=suites)
    result = _parse_reporter_json(raw)
    assert result.passed is False
    assert result.error != ""
