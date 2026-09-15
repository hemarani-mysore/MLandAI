"""Token/cost accounting on `StructuredLLM` instances — pure math (no network,
no real chat-model construction) plus end-to-end wiring into `DossierResponse`."""

from types import SimpleNamespace

from praxis.graph import run_dossier
from praxis.llm import PRICING_USD_PER_1M, FakeStructuredLLM, LangChainStructuredLLM, _price_for
from praxis.schemas import DossierRequest


def _bare_llm() -> LangChainStructuredLLM:
    """A `LangChainStructuredLLM` with no real chat models built (no network,
    no API key needed) — just enough state for `_record_usage()`."""
    llm = object.__new__(LangChainStructuredLLM)
    llm._model_names = {"strong": "openai:gpt-4o", "fast": "openai:gpt-4o-mini"}
    return llm


def test_price_for_strips_the_provider_prefix():
    assert _price_for("openai:gpt-4o") == PRICING_USD_PER_1M["gpt-4o"]
    assert _price_for("gpt-4o-mini") == PRICING_USD_PER_1M["gpt-4o-mini"]


def test_price_for_an_unmapped_model_is_zero_not_a_guess():
    assert _price_for("anthropic:claude-opus-5") == (0.0, 0.0)


def test_record_usage_accumulates_tokens_and_cost():
    llm = _bare_llm()
    raw = SimpleNamespace(usage_metadata={"input_tokens": 1000, "output_tokens": 500})

    llm._record_usage("strong", raw)

    price_in, price_out = PRICING_USD_PER_1M["gpt-4o"]
    expected_cost = (1000 * price_in + 500 * price_out) / 1_000_000
    assert llm.total_prompt_tokens == 1000
    assert llm.total_completion_tokens == 500
    assert llm.total_cost_usd == expected_cost


def test_record_usage_accumulates_across_multiple_calls():
    llm = _bare_llm()

    llm._record_usage(
        "fast", SimpleNamespace(usage_metadata={"input_tokens": 100, "output_tokens": 50})
    )
    llm._record_usage(
        "fast", SimpleNamespace(usage_metadata={"input_tokens": 200, "output_tokens": 75})
    )

    assert llm.total_prompt_tokens == 300
    assert llm.total_completion_tokens == 125


def test_record_usage_tolerates_missing_usage_metadata():
    llm = _bare_llm()

    llm._record_usage("strong", SimpleNamespace())  # no usage_metadata attr at all

    assert llm.total_prompt_tokens == 0
    assert llm.total_cost_usd == 0.0


def test_fake_llm_never_accumulates_any_cost():
    llm = FakeStructuredLLM()
    assert llm.total_prompt_tokens == 0
    assert llm.total_completion_tokens == 0
    assert llm.total_cost_usd == 0.0


def test_dossier_response_carries_zero_cost_on_the_fake_provider():
    resp = run_dossier(DossierRequest(subject="Acme Robotics"), llm=FakeStructuredLLM())
    assert resp.cost_usd == 0.0
    assert resp.tokens == {"prompt": 0, "completion": 0, "total": 0}
