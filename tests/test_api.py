import json
import sys
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

from pondlite import serve
from pondlite.store import resolve_run_dir
from pondlite.version import code_fingerprint


class FakeTokenizer:
    eos_token_id = 0

    def __call__(self, prompt, return_tensors):
        return SimpleNamespace(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.tensor([[1, 1, 1]]),
        )

    def decode(self, ids, skip_special_tokens):
        return "two tokens"


class FakeModel:
    def generate(self, ids, **kwargs):
        return torch.tensor([[1, 2, 3, 4, 5]])


class GroundedTokenizer:
    eos_token_id = 0

    def __call__(self, text, **kwargs):
        return {"input_ids": [1, 2, 3, 4]}


class GroundedRetriever:
    def __init__(self, abstained=False):
        self.abstained = abstained

    def answer(self, question):
        if self.abstained:
            return {"answer": "Insufficient dataset information.", "confidence": 0.05,
                    "matched_source": None, "abstained": True}
        return {"answer": "Open Billing and choose Cancel.", "confidence": 0.82,
                "matched_source": {"instruction": "How do I cancel?"}, "abstained": False}


def test_grounded_answer_does_not_require_generative_model(monkeypatch):
    monkeypatch.setattr(serve, "STATE", {
        "run_dir": "runs/demo/v5", "model": None, "tokenizer": GroundedTokenizer(),
        "metrics": {"perplexity": 10.0}, "device": "cpu", "retriever": GroundedRetriever(),
    })
    response = TestClient(serve.app).post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "How do I cancel?"}],
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "Open Billing and choose Cancel."
    assert payload["pondlite"]["mode"] == "grounded"
    assert payload["pondlite"]["confidence"] == 0.82
    assert payload["pondlite"]["source_instruction"] == "How do I cancel?"


def test_unknown_question_returns_safe_abstention(monkeypatch):
    monkeypatch.setattr(serve, "STATE", {
        "run_dir": "runs/demo/v5", "model": None, "tokenizer": GroundedTokenizer(),
        "metrics": {"perplexity": 10.0}, "device": "cpu",
        "retriever": GroundedRetriever(abstained=True),
    })
    response = TestClient(serve.app).post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "What is the weather?"}],
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["pondlite"]["mode"] == "abstained"
    assert "Insufficient dataset information" in payload["choices"][0]["message"]["content"]
    assert payload["pondlite"]["source_instruction"] is None


def test_real_load_run_builds_retriever(isolated_project, monkeypatch):
    root = isolated_project
    run_dir = root / "runs" / "demo" / "v1"
    (run_dir / "adapter").mkdir(parents=True)
    dataset = root / "data" / "demo" / "v1"
    dataset.mkdir(parents=True)
    record = {"instruction": "How do I cancel?", "input": None, "output": "Open Billing."}
    (dataset / "train.jsonl").write_text(json.dumps(record) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps({"model": "fake", "dataset_dir": "data/demo/v1"}))

    class LoadedModel:
        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(serve, "ROOT", root)
    monkeypatch.setattr(serve, "STATE", {})
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda path: FakeTokenizer()),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda name: object()),
    ))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
        PeftModel=SimpleNamespace(from_pretrained=lambda base, path: LoadedModel()),
    ))
    serve.load_run("runs/demo/v1")
    assert serve.STATE["retriever"].answer("How do I cancel?")["answer"] == "Open Billing."
    assert serve.STATE["device"] == "cpu"


def test_production_startup_loads_configured_run(monkeypatch):
    loaded = []
    monkeypatch.setenv("PONDLITE_RUN_DIR", "runs/demo/v5")
    monkeypatch.setattr(serve, "load_run", lambda path: loaded.append(path))
    serve.load_default_run()
    assert loaded == ["runs/demo/v5"]


def test_health_exposes_current_source_fingerprint(monkeypatch):
    monkeypatch.setattr(serve, "STATE", {
        "run_dir": None, "model": None, "tokenizer": None, "metrics": None, "device": None,
    })
    response = TestClient(serve.app).get("/health")
    assert response.status_code == 200
    assert response.json()["code_fingerprint"] == code_fingerprint()
    assert response.json()["status"] == "no model loaded"


def test_chat_reports_model_token_counts_and_length_finish(monkeypatch):
    monkeypatch.setattr(serve, "STATE", {
        "run_dir": "runs/demo/v1", "model": FakeModel(), "tokenizer": FakeTokenizer(),
        "metrics": {"perplexity": 1.0}, "device": "cpu",
    })
    response = TestClient(serve.app).post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 2,
        "temperature": 0,
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    assert payload["choices"][0]["finish_reason"] == "length"
    assert payload["id"].startswith("chatcmpl-")
    assert payload["id"] != "chatcmpl-pondlite"


def test_chat_validation_and_missing_model(monkeypatch):
    monkeypatch.setattr(serve, "STATE", {
        "run_dir": None, "model": None, "tokenizer": None, "metrics": None, "device": None,
    })
    client = TestClient(serve.app)
    messages = [{"role": "user", "content": "Hello"}]
    assert client.post("/v1/chat/completions", json={"messages": messages}).status_code == 503
    assert client.post("/v1/chat/completions", json={"messages": messages, "max_tokens": 0}).status_code == 422
    assert client.post("/v1/chat/completions", json={"messages": messages, "temperature": 3}).status_code == 422


def test_prompt_includes_supported_roles():
    messages = [
        serve.ChatMessage(role="system", content="Be concise"),
        serve.ChatMessage(role="user", content="Question"),
        serve.ChatMessage(role="assistant", content="Earlier answer"),
    ]
    prompt = serve.build_prompt(messages)
    assert "Be concise" in prompt
    assert "Question" in prompt
    assert "Earlier answer" in prompt
    assert prompt.endswith("### Response:\n")


def test_run_resolution_requires_complete_artifacts(isolated_project):
    run_dir = isolated_project / "runs" / "demo" / "v1"
    (run_dir / "adapter").mkdir(parents=True)
    (run_dir / "metrics.json").write_text("{}")
    assert resolve_run_dir(isolated_project, "runs/demo/v1") == run_dir.resolve()
    (run_dir / "metrics.json").unlink()
    with pytest.raises(ValueError, match="metrics.json"):
        resolve_run_dir(isolated_project, "runs/demo/v1")
