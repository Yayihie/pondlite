import json
import sys
from types import SimpleNamespace

import pytest
import torch

from pondlite import store
from pondlite import train as training


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"

    @classmethod
    def from_pretrained(cls, model_name):
        return cls()

    def __call__(self, text, **kwargs):
        return {"input_ids": [1, 2, 3, 4] if "<|endoftext|>" in text else [1, 2]}

    def save_pretrained(self, directory):
        from pathlib import Path
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "tokenizer.json").write_text("{}")


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, input_ids=None, attention_mask=None):
        logits = torch.zeros(
            (*input_ids.shape, 5), device=input_ids.device
        ) + self.weight
        return SimpleNamespace(logits=logits)

    def save_pretrained(self, directory):
        from pathlib import Path
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "adapter_model.safetensors").write_bytes(b"test")


def test_model_load_failure_is_registered_and_visible(isolated_project, monkeypatch):
    root = isolated_project
    dataset_dir = root / "data" / "demo" / "v1"
    dataset_dir.mkdir(parents=True)
    record = {"instruction": "How do I cancel?", "input": None, "output": "Open Billing and choose Cancel."}
    (dataset_dir / "train.jsonl").write_text(json.dumps(record) + "\n")
    (dataset_dir / "holdout.jsonl").write_text(json.dumps(record) + "\n")
    monkeypatch.setattr(training, "__file__", str(root / "pondlite" / "train.py"))
    observed = {}

    class BrokenTokenizer:
        @classmethod
        def from_pretrained(cls, model_name):
            conn = store.db()
            observed["rows_during_load"] = [dict(row) for row in conn.execute("SELECT * FROM runs")]
            conn.close()
            raise RuntimeError("model download failed")

    fake_transformers = SimpleNamespace(
        AutoTokenizer=BrokenTokenizer,
        AutoModelForCausalLM=SimpleNamespace(),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    with pytest.raises(RuntimeError, match="model download failed"):
        training.train(str(dataset_dir), "demo", epochs=1)

    assert len(observed["rows_during_load"]) == 1
    assert observed["rows_during_load"][0]["status"] == "running"

    conn = store.db()
    rows = [dict(row) for row in conn.execute("SELECT * FROM runs")]
    conn.close()
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"

    run_dir = root / "runs" / "demo" / "v1"
    progress = json.loads((run_dir / "progress.json").read_text())
    assert progress["status"] == "failed"
    assert "model download failed" in progress["error"]
    assert "model download failed" in (run_dir / "error.log").read_text()


def test_successful_run_uses_latest_dataset_and_updates_single_row(isolated_project, monkeypatch):
    root = isolated_project
    for version in (1, 2):
        dataset_dir = root / "data" / "demo" / f"v{version}"
        dataset_dir.mkdir(parents=True)
        record = {"instruction": "How do I cancel?", "input": None, "output": "Open Billing and choose Cancel."}
        (dataset_dir / "train.jsonl").write_text(json.dumps(record) + "\n")
        (dataset_dir / "holdout.jsonl").write_text(json.dumps(record) + "\n")
    monkeypatch.setattr(training, "__file__", str(root / "pondlite" / "train.py"))

    class TinyModelFactory:
        @classmethod
        def from_pretrained(cls, model_name):
            return TinyModel()

    class FakeLoraConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=TinyTokenizer,
        AutoModelForCausalLM=TinyModelFactory,
    ))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
        LoraConfig=FakeLoraConfig,
        get_peft_model=lambda base, config: base,
    ))

    run_dir, metrics = training.train(
        str(root / "data" / "demo" / "missing"), "demo", epochs=1, batch_size=1
    )

    assert metrics["dataset_dir"].endswith("data/demo/v2")
    assert metrics["steps"] == 1
    assert (run_dir / "adapter" / "adapter_model.safetensors").is_file()
    assert json.loads((run_dir / "progress.json").read_text())["status"] == "done"
    assert not (run_dir / "error.log").exists()
    conn = store.db()
    rows = [dict(row) for row in conn.execute("SELECT * FROM runs")]
    conn.close()
    assert len(rows) == 1
    assert rows[0]["status"] == "done"
    assert rows[0]["perplexity"] == metrics["perplexity"]
