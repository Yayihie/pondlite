"""Held-out eval: loss/perplexity + generated-vs-reference comparison table -> eval.json."""

import argparse
import json
import math
import traceback
from pathlib import Path

import torch

from .respond import TfidfRetriever, normalize_answer
from .store import db, load_jsonl, now


def _load_run(run_dir: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    rdir = Path(run_dir)
    metrics = json.loads((rdir / "metrics.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(rdir / "adapter"))
    base = AutoModelForCausalLM.from_pretrained(metrics["model"])
    model = PeftModel.from_pretrained(base, str(rdir / "adapter"))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device).eval()
    return model, tokenizer, metrics, device


def _eval_run(run_dir: str, max_new_tokens: int = 48, max_samples: int = 10):
    from .train import collate, format_example, SFTDataset

    model, tokenizer, metrics, device = _load_run(run_dir)
    rdir = Path(run_dir).resolve()
    dataset_dir = Path(metrics["dataset_dir"])
    if not dataset_dir.is_absolute():
        dataset_dir = rdir.parents[2] / dataset_dir
    holdout = load_jsonl(dataset_dir / "holdout.jsonl")
    retriever = TfidfRetriever(load_jsonl(dataset_dir / "train.jsonl"))
    ds = SFTDataset(holdout, tokenizer)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # held-out loss / perplexity
    loader = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=lambda b: collate(b, pad_id))
    total, ntok = 0.0, 0
    with torch.no_grad():
        for input_ids, labels, attn in loader:
            out = model(input_ids=input_ids.to(device), attention_mask=attn.to(device))
            logits = out.logits[:, :-1, :]
            tgt = labels[:, 1:].to(device)
            losses = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1),
                ignore_index=-100, reduction="none")
            total += losses.sum().item()
            ntok += (tgt != -100).sum().item()
    eval_loss = total / max(ntok, 1)

    # generation samples
    samples = []
    for r in holdout[:max_samples]:
        prompt = format_example(r).split("### Response:\n")[0] + "### Response:\n"
        enc = tokenizer(prompt, return_tensors="pt")
        ids = enc.input_ids.to(device)
        attn = enc.attention_mask.to(device)
        with torch.no_grad():
            gen = model.generate(ids, attention_mask=attn, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
        ref = r["output"]
        match = text.strip()[: len(ref)].strip().lower() == ref.strip().lower()
        grounded = retriever.answer(r["instruction"], input=r.get("input"))
        grounded_match = (
            not grounded["abstained"]
            and normalize_answer(grounded["answer"]) == normalize_answer(ref)
        )
        samples.append({
            "instruction": r["instruction"],
            "input": r.get("input"),
            "reference": ref,
            "generated": text.strip(),
            "exact_match_on_prefix": match,
            "grounded_answer": grounded["answer"],
            "grounded_confidence": round(grounded["confidence"], 4),
            "grounded_abstained": grounded["abstained"],
            "grounded_policy_match": grounded_match,
        })
    n_match = sum(s["exact_match_on_prefix"] for s in samples)

    result = {
        "run_dir": run_dir, "created_at": now(),
        "holdout_records": len(holdout),
        "eval_loss": round(eval_loss, 4),
        "perplexity": round(math.exp(eval_loss), 4),
        "samples": samples,
        "n_samples": len(samples),
        "greedy_prefix_matches": n_match,
        "grounded_policy_matches": sum(s["grounded_policy_match"] for s in samples),
        "grounded_abstentions": sum(s["grounded_abstained"] for s in samples),
        "grounded_mean_confidence": round(
            sum(s["grounded_confidence"] for s in samples) / max(len(samples), 1), 4
        ),
    }
    (rdir / "eval.json").write_text(json.dumps(result, indent=2))

    conn = db()
    conn.execute(
        "INSERT INTO evals(run_dir, eval_loss, perplexity, samples_scored, created_at) VALUES(?,?,?,?,?)",
        (run_dir, result["eval_loss"], result["perplexity"], len(samples), now()),
    )
    conn.commit()
    conn.close()
    return result


def eval_run(run_dir: str, max_new_tokens: int = 48, max_samples: int = 10):
    """Run evaluation with a UI-readable lifecycle and preserved failure details."""
    rdir = Path(run_dir).resolve()
    progress_path = rdir / "eval_progress.json"

    def write_progress(status, **extra):
        progress_path.write_text(json.dumps({"status": status, "updated_at": now(), **extra}, indent=2))

    write_progress("running")
    try:
        result = _eval_run(run_dir, max_new_tokens=max_new_tokens, max_samples=max_samples)
    except Exception as exc:
        (rdir / "eval_error.log").write_text(traceback.format_exc())
        write_progress("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    write_progress("done", eval_loss=result["eval_loss"], perplexity=result["perplexity"],
                   samples=result["n_samples"],
                   grounded_policy_matches=result["grounded_policy_matches"],
                   grounded_abstentions=result["grounded_abstentions"])
    return result


def main():
    ap = argparse.ArgumentParser(description="PondLite eval")
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    print(json.dumps(eval_run(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
