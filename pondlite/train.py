"""SFT training with LoRA on a small HF causal LM (tiny-gpt2 / distilgpt2).

Writes runs/<name>/v<n>/: LoRA adapter, metrics.json, progress.json, and
error.log on failure. Runs are registered before model loading so the web UI
can distinguish active, completed, and failed jobs.
"""

import argparse
import json
import math
import time
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .store import db, latest_dataset_dir, now, run_dir, validate_name

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_CHOICES = {"tiny": "sshleifer/tiny-gpt2", "distil": "distilgpt2", "gpt2": "gpt2"}

PROMPT_TPL = "### Instruction:\n{instruction}\n{input_line}### Response:\n{output}<|endoftext|>"


def format_prompt(r):
    input_line = f"Input: {r['input']}\n" if r.get("input") else ""
    return (f"### Instruction:\n{r['instruction']}\n{input_line}### Response:\n")


def format_example(r):
    return f"{format_prompt(r)}{r['output']}<|endoftext|>"


class SFTDataset(Dataset):
    def __init__(self, records, tokenizer, max_len=256):
        self.examples = []
        for r in records:
            ids = tokenizer(format_example(r), truncation=True, max_length=max_len)["input_ids"]
            prompt_ids = tokenizer(format_prompt(r), truncation=True, max_length=max_len)["input_ids"]
            prompt_len = min(len(prompt_ids), len(ids))
            labels = [-100] * prompt_len + ids[prompt_len:]
            self.examples.append((ids, labels))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ids, labels = self.examples[i]
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)}


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        ids = b["input_ids"].tolist() if hasattr(b["input_ids"], "tolist") else list(b["input_ids"])
        n = len(ids)
        pad = [pad_id] * (maxlen - n)
        input_ids.append(ids + pad)
        labels.append(ids + [-100] * (maxlen - n))
        attn.append([1] * n + [0] * (maxlen - n))
    return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn))


def evaluate_loss(model, loader, device):
    model.eval()
    total, ntok = 0.0, 0
    with torch.no_grad():
        for input_ids, labels, attn in loader:
            out = model(input_ids=input_ids.to(device), attention_mask=attn.to(device))
            logits = out.logits[:, :-1, :]
            tgt = labels[:, 1:].to(device)
            mask = tgt != -100
            losses = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1), ignore_index=-100, reduction="none")
            total += losses.sum().item()
            ntok += mask.sum().item()
    model.train()
    return total / max(ntok, 1)


def train(dataset_dir: str, name: str, epochs: int = 3, batch_size: int = 4,
          lr: float = 2e-4, model_key: str = "tiny"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from .store import load_jsonl, next_version

    root = Path(__file__).resolve().parent.parent
    name = validate_name(name)
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch size must be at least 1")
    ddir = Path(dataset_dir)
    if not ddir.exists():
        ddir = latest_dataset_dir(root, name)
    train_recs = load_jsonl(ddir / "train.jsonl")
    holdout = load_jsonl(ddir / "holdout.jsonl")
    if not train_recs or not holdout:
        raise ValueError("dataset must contain non-empty train.jsonl and holdout.jsonl")

    version = next_version(root, name, "runs")
    rdir = run_dir(root, name, version)
    rdir.mkdir(parents=True, exist_ok=False)
    backend = f"peft-lora/{DEVICE}"
    created_at = now()
    conn = db()
    conn.execute(
        """INSERT INTO runs(name, version, dataset_dir, run_dir, backend, epochs, status,
           final_loss, eval_loss, perplexity, created_at, updated_at)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
        (name, version, str(ddir), str(rdir), backend, epochs, "running",
         None, None, None, created_at, created_at),
    )
    conn.commit()
    conn.close()

    def write_progress(status="running", **kw):
        progress_path = rdir / "progress.json"
        previous = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        prog = {**previous, "status": status, "dataset": str(ddir), "updated_at": now(),
                "epochs": epochs, "device": DEVICE, **kw}
        progress_path.write_text(json.dumps(prog, indent=2))
        conn = db()
        conn.execute(
            "UPDATE runs SET status=?, updated_at=? WHERE name=? AND version=?",
            (status, prog["updated_at"], name, version),
        )
        conn.commit()
        conn.close()

    try:
        write_progress(step="loading model")
        model_name = MODEL_CHOICES[model_key]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        base = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        lcfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                          target_modules=["c_attn"], bias="none",
                          task_type="CAUSAL_LM")
        model = get_peft_model(base, lcfg)
        model.to(DEVICE)
        model.train()
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        train_ds = SFTDataset(train_recs, tokenizer)
        hold_ds = SFTDataset(holdout, tokenizer)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  collate_fn=lambda b: collate(b, pad_id))
        hold_loader = DataLoader(hold_ds, batch_size=batch_size,
                                 collate_fn=lambda b: collate(b, pad_id))

        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
        write_progress(step="training", total_steps=len(train_loader) * epochs,
                       n_trainable_params=n_trainable, model=model_name)

        loss_curve, step = [], 0
        t0 = time.time()
        base_eval = evaluate_loss(model, hold_loader, DEVICE)
        loss_curve.append({"step": 0, "train_loss": None, "holdout_loss": round(base_eval, 4)})

        for epoch in range(epochs):
            for input_ids, labels, attn in train_loader:
                out = model(input_ids=input_ids.to(DEVICE), attention_mask=attn.to(DEVICE))
                logits = out.logits[:, :-1, :]
                tgt = labels[:, 1:].to(DEVICE)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1), ignore_index=-100)
                opt.zero_grad()
                loss.backward()
                opt.step()
                step += 1
                loss_curve.append({"step": step, "train_loss": round(loss.item(), 4)})
                if step % 5 == 0:
                    write_progress(step=step, latest_loss=loss.item(), elapsed_s=round(time.time() - t0, 1))

            hv = evaluate_loss(model, hold_loader, DEVICE)
            loss_curve.append({"step": step, "train_loss": None, "holdout_loss": round(hv, 4)})
            write_progress(step=step, epoch=epoch + 1, latest_holdout_loss=hv)

        final_holdout = evaluate_loss(model, hold_loader, DEVICE)
        final_loss = next((c["train_loss"] for c in reversed(loss_curve) if c["train_loss"] is not None), None)

        metrics = {
            "name": name, "version": version, "dataset_dir": str(ddir),
            "model": model_name, "backend": backend,
            "n_trainable_params": n_trainable, "n_train": len(train_recs),
            "n_holdout": len(holdout), "epochs": epochs, "batch_size": batch_size,
            "lr": lr, "steps": step, "elapsed_s": round(time.time() - t0, 1),
            "loss_curve": loss_curve, "final_train_loss": final_loss,
            "holdout_loss": round(final_holdout, 4),
            "perplexity": round(math.exp(final_holdout), 4),
        }
        (rdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        model.save_pretrained(str(rdir / "adapter"))
        tokenizer.save_pretrained(str(rdir / "adapter"))
        write_progress(status="done", holdout_loss=metrics["holdout_loss"],
                       perplexity=metrics["perplexity"], final_train_loss=final_loss)

        conn = db()
        conn.execute(
            """UPDATE runs SET dataset_dir=?, run_dir=?, backend=?, epochs=?, status=?,
               final_loss=?, eval_loss=?, perplexity=?, updated_at=? WHERE name=? AND version=?""",
            (str(ddir), str(rdir), backend, epochs, "done", final_loss,
             metrics["holdout_loss"], metrics["perplexity"], now(), name, version),
        )
        conn.commit()
        conn.close()
        return rdir, metrics
    except Exception as exc:
        (rdir / "error.log").write_text(traceback.format_exc())
        write_progress(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise


def main():
    ap = argparse.ArgumentParser(description="PondLite train (LoRA SFT)")
    ap.add_argument("--dataset", required=True, help="dataset dir (data/<name>/v<n>)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--model", choices=MODEL_CHOICES, default="distil")
    args = ap.parse_args()
    rdir, metrics = train(args.dataset, args.name, args.epochs, args.batch_size, args.lr, args.model)
    print(json.dumps({"run_dir": str(rdir), **metrics}, indent=2))


if __name__ == "__main__":
    main()
