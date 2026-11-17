import json
from pathlib import Path

import pytest

from pondlite.respond import ABSTAIN_MESSAGE, TfidfRetriever, normalize_answer


ROOT = Path(__file__).resolve().parents[1]
DEMO_DATA = ROOT / "data" / "demo" / "v1"


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_normalize_answer_strips_greetings_and_surface_formatting():
    personalized = "Hi Tom, thanks for writing in. Settings > Billing > Cancel. You won't be charged again!"
    plain = "settings billing cancel you won t be charged again"
    assert normalize_answer(personalized) == plain
    assert normalize_answer("Happy to help. Good question. " + personalized.split(". ", 1)[1]) == plain


def test_retriever_returns_grounded_clean_answer_and_source():
    records = [
        {
            "instruction": "How do I delete my account?",
            "input": None,
            "output": "Hi Sam! Go to Settings > Account > Delete account. This can't be undone.",
        },
        {
            "instruction": "Do you ship internationally?",
            "input": None,
            "output": "Happy to help. We ship to 40+ countries.",
        },
    ]
    result = TfidfRetriever(records).answer("Can I delete my account?")

    assert result["answer"] == "Go to Settings > Account > Delete account. This can't be undone."
    assert "Sam" not in result["answer"]
    assert result["confidence"] >= 0.45
    assert result["matched_source"] is records[0]
    assert result["abstained"] is False


def test_retriever_uses_optional_input_and_fit_returns_self():
    records = [
        {"instruction": "How do I change it?", "input": "email address", "output": "Open Account settings."},
        {"instruction": "How do I change it?", "input": "credit card", "output": "Open Billing settings."},
    ]
    retriever = TfidfRetriever()
    assert retriever.fit(records) is retriever

    result = retriever.answer("How do I change it?", input="credit card")
    assert result["answer"] == "Open Billing settings."
    assert result["matched_source"] is records[1]


def test_retriever_abstains_for_unrelated_or_empty_questions():
    retriever = TfidfRetriever([
        {"instruction": "Do you ship internationally?", "input": None, "output": "We ship to 40+ countries."},
    ])
    for question in ("What is the weather tomorrow?", "", "   "):
        result = retriever.answer(question)
        assert result["answer"] == ABSTAIN_MESSAGE
        assert "insufficient dataset information" in result["answer"].lower()
        assert result["matched_source"] is None
        assert result["abstained"] is True
        assert 0.0 <= result["confidence"] < 0.45


def test_demo_holdout_has_eight_normalized_policy_matches():
    retriever = TfidfRetriever(load_jsonl(DEMO_DATA / "train.jsonl"))
    holdout = load_jsonl(DEMO_DATA / "holdout.jsonl")
    assert len(holdout) == 8

    results = [retriever.answer(row["instruction"], input=row.get("input")) for row in holdout]
    matches = [
        normalize_answer(result["answer"]) == normalize_answer(row["output"])
        for result, row in zip(results, holdout)
    ]
    assert sum(matches) == 8, [
        (row["instruction"], result["confidence"], result["matched_source"])
        for row, result, match in zip(holdout, results, matches)
        if not match
    ]
    assert all(not result["abstained"] for result in results)


@pytest.mark.parametrize("question", [
    "What is the weather in Paris tomorrow?",
    "Write a poem about the ocean.",
    "How do I deploy a Kubernetes cluster?",
    "What is the capital of Australia?",
])
def test_demo_retriever_abstains_on_unknown_questions(question):
    retriever = TfidfRetriever(load_jsonl(DEMO_DATA / "train.jsonl"))
    result = retriever.answer(question)

    assert result["abstained"] is True
    assert result["answer"] == ABSTAIN_MESSAGE
    assert "insufficient dataset information" in result["answer"].lower()
    assert result["matched_source"] is None
    assert result["confidence"] < 0.45
