"""Serve trained runs through an OpenAI-style chat endpoint on port 8420."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

from .auth import auth_enabled, require_production_config, valid_credentials
from .respond import TfidfRetriever
from .store import load_jsonl, resolve_run_dir
from .version import ROOT, code_fingerprint

@asynccontextmanager
async def lifespan(app):
    load_default_run()
    yield


app = FastAPI(title="PondLite Serve", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)
STATE = {"run_dir": None, "model": None, "tokenizer": None, "metrics": None, "device": None, "retriever": None}
basic_auth = HTTPBasic(auto_error=False)


def require_api_auth(credentials: HTTPBasicCredentials = Depends(basic_auth)):
    if auth_enabled() and (
        credentials is None or not valid_credentials(credentials.username, credentials.password)
    ):
        raise HTTPException(
            status_code=401, detail="valid credentials required",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: int = Field(default=96, gt=0, le=1024)
    temperature: float = Field(default=0.7, ge=0, le=2)
    mode: str = "grounded"


def load_run(run_dir: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    rdir = resolve_run_dir(ROOT, run_dir)
    metrics = json.loads((rdir / "metrics.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(rdir / "adapter"))
    base = AutoModelForCausalLM.from_pretrained(metrics["model"])
    model = PeftModel.from_pretrained(base, str(rdir / "adapter"))
    device = "mps" if __import__("torch").backends.mps.is_available() else "cpu"
    model.to(device).eval()
    dataset_dir = Path(metrics["dataset_dir"])
    if not dataset_dir.is_absolute():
        dataset_dir = rdir.parents[2] / dataset_dir
    retriever = TfidfRetriever(load_jsonl(dataset_dir / "train.jsonl"))
    STATE.update(run_dir=str(rdir), model=model, tokenizer=tokenizer, metrics=metrics,
                 device=device, retriever=retriever)


def load_default_run():
    """Load the configured or latest completed model under production servers."""
    if STATE["model"] or STATE.get("retriever"):
        return
    configured = os.environ.get("PONDLITE_RUN_DIR")
    if configured:
        load_run(configured)
        return
    candidates = [
        path for path in (ROOT / "runs").glob("*/*")
        if (path / "metrics.json").is_file() and (path / "adapter").is_dir()
    ]
    if candidates:
        latest = max(candidates, key=lambda path: path.stat().st_mtime)
        load_run(str(latest))


def build_prompt(messages: list[ChatMessage]) -> str:
    """Flatten chat messages into our SFT prompt format."""
    parts = []
    for m in messages:
        if m.role == "system":
            parts.append(f"### Instruction:\n{m.content}")
        elif m.role == "user":
            parts.append(f"### Instruction:\n{m.content}")
        elif m.role == "assistant":
            parts.append(f"### Response:\n{m.content}")
    return "\n".join(parts) + "\n### Response:\n"


@app.get("/health")
def health():
    return {"status": "ok" if STATE["model"] or STATE.get("retriever") else "no model loaded",
            "run_dir": STATE["run_dir"],
            "perplexity": STATE["metrics"]["perplexity"] if STATE["metrics"] else None,
            "serving_mode": "grounded" if STATE.get("retriever") else "generative",
            "code_fingerprint": code_fingerprint(), "pid": os.getpid()}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest, credentials=Depends(require_api_auth)):
    import torch

    if req.mode not in {"grounded", "generate"}:
        raise HTTPException(400, "mode must be 'grounded' or 'generate'")
    if req.mode == "grounded" and STATE.get("retriever"):
        user_messages = [message.content for message in req.messages if message.role == "user"]
        if not user_messages:
            raise HTTPException(400, "at least one user message is required")
        result = STATE["retriever"].answer(user_messages[-1])
        text = result["answer"]
        prompt_tokens = len(STATE["tokenizer"](build_prompt(req.messages))["input_ids"])
        completion_tokens = len(STATE["tokenizer"](text)["input_ids"])
        source = result.get("matched_source") or {}
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
            "created": int(time.time()), "model": STATE["run_dir"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens},
            "pondlite": {
                "mode": "abstained" if result["abstained"] else "grounded",
                "confidence": round(result["confidence"], 4),
                "source_instruction": source.get("instruction"),
            },
        }

    if not STATE["model"]:
        raise HTTPException(503, "no run loaded — POST /v1/runs first or start with --run-dir")

    prompt = build_prompt(req.messages)
    enc = STATE["tokenizer"](prompt, return_tensors="pt")
    ids = enc.input_ids.to(STATE["device"])
    attn = enc.attention_mask.to(STATE["device"])
    with torch.no_grad():
        gen = STATE["model"].generate(
            ids, attention_mask=attn, max_new_tokens=req.max_tokens, do_sample=req.temperature > 0,
            temperature=max(req.temperature, 1e-5), pad_token_id=STATE["tokenizer"].eos_token_id)
    completion_ids = gen[0][ids.shape[1]:]
    completion_tokens = int(completion_ids.shape[0])
    text = STATE["tokenizer"].decode(completion_ids, skip_special_tokens=True)
    text = text.split("###")[0].strip()
    reached_limit = completion_tokens >= req.max_tokens and (
        completion_tokens == 0 or int(completion_ids[-1]) != STATE["tokenizer"].eos_token_id
    )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": STATE["run_dir"],
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": text},
                     "finish_reason": "length" if reached_limit else "stop"}],
        "usage": {"prompt_tokens": int(ids.shape[1]),
                  "completion_tokens": completion_tokens,
                  "total_tokens": int(ids.shape[1]) + completion_tokens},
        "pondlite": {"mode": "generative", "confidence": None, "source_instruction": None},
    }


@app.post("/v1/runs")
def set_run(body: dict, credentials=Depends(require_api_auth)):
    run_dir = body.get("run_dir")
    if not isinstance(run_dir, str) or not run_dir.strip():
        raise HTTPException(400, "run_dir is required")
    try:
        load_run(run_dir)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"failed to load run: {exc}")
    return {"loaded": STATE["run_dir"], "perplexity": STATE["metrics"]["perplexity"]}


def main():
    require_production_config()
    import uvicorn
    ap = argparse.ArgumentParser(description="PondLite serve")
    ap.add_argument("--run-dir", help="trained run dir to serve")
    ap.add_argument("--host", default="127.0.0.1", help="bind address; use 0.0.0.0 only on a trusted network")
    ap.add_argument("--port", type=int, default=8420)
    args = ap.parse_args()
    if args.run_dir:
        load_run(args.run_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
