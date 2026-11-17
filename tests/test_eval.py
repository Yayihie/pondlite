import json
import sys
from types import SimpleNamespace

import torch

from pondlite import eval as evaluation
from pondlite.store import db


class EvalTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"

    @classmethod
    def from_pretrained(cls, path):
        return cls()

    def __call__(self, text, **kwargs):
        if kwargs.get("return_tensors") == "pt":
            return SimpleNamespace(
                input_ids=torch.tensor([[1, 2, 3]]),
                attention_mask=torch.tensor([[1, 1, 1]]),
            )
        return {"input_ids": [1, 2, 3, 4] if "<|endoftext|>" in text else [1, 2]}

    def decode(self, ids, skip_special_tokens):
        return "two tokens"


class EvalModel(torch.nn.Module):
    def forward(self, input_ids=None, attention_mask=None):
        return SimpleNamespace(logits=torch.zeros((*input_ids.shape, 5)))

    def generate(self, ids, **kwargs):
        return torch.tensor([[1, 2, 3, 4, 5]])


def test_eval_loads_run_and_resolves_relative_dataset_from_run(isolated_project, monkeypatch):
    root = isolated_project
    run_dir = root / "runs" / "demo" / "v1"
    (run_dir / "adapter").mkdir(parents=True)
    dataset = root / "data" / "demo" / "v1"
    dataset.mkdir(parents=True)
    record = {"instruction": "Say something", "input": None, "output": "two tokens"}
    (dataset / "holdout.jsonl").write_text(json.dumps(record) + "\n")
    (dataset / "train.jsonl").write_text(json.dumps(record) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps({
        "dataset_dir": "data/demo/v1", "model": "fake-model",
    }))
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=EvalTokenizer,
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda name: object()),
    ))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
        PeftModel=SimpleNamespace(from_pretrained=lambda base, path: EvalModel()),
    ))
    monkeypatch.chdir(root.parent)  # relative dataset paths must not depend on caller cwd

    result = evaluation.eval_run(str(run_dir), max_samples=1)

    assert result["eval_loss"] == 1.6094
    assert result["perplexity"] == 5.0
    assert result["greedy_prefix_matches"] == 1
    assert result["samples"][0]["generated"] == "two tokens"
    assert result["grounded_policy_matches"] == 1
    assert result["grounded_abstentions"] == 0
    assert result["samples"][0]["grounded_policy_match"] is True
    assert json.loads((run_dir / "eval_progress.json").read_text())["status"] == "done"
    assert (run_dir / "eval.json").is_file()
    conn = db()
    row = conn.execute("SELECT run_dir, samples_scored FROM evals").fetchone()
    conn.close()
    assert tuple(row) == (str(run_dir), 1)


def test_eval_cli_prints_result(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pondlite.eval", "--run-dir", "runs/demo/v1"])
    monkeypatch.setattr(evaluation, "eval_run", lambda path: {"run_dir": path, "eval_loss": 1.0})
    evaluation.main()
    assert json.loads(capsys.readouterr().out) == {"run_dir": "runs/demo/v1", "eval_loss": 1.0}
