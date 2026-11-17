# PondLite

A working, local **raw data → trained data → served model** platform. Inspired by OpenPond's
Dataset / Experiment / Model / Serve concepts, shrunk to run on a laptop (tested: Apple M3, 8GB RAM).
It trains **real models** — LoRA fine-tuning on `sshleifer/tiny-gpt2` via HuggingFace `peft` —
and serves them behind an OpenAI-compatible endpoint. Every metric you see comes from a real run.

## Setup and quickstart

Requires Python 3.9+. Install the pinned runtime dependencies, then run the pipeline:

```bash
cd ~/workspace/pondlite
python3 -m pip install --user -r requirements.txt
python3 -m pondlite.ingest demo/sample_raw_data.jsonl && \
  python3 -m pondlite.curate --from-ingested --name demo && \
  python3 -m pondlite.train --dataset data/demo/v1 --name demo --epochs 3 --model tiny
python3 -m pondlite.eval --run-dir runs/demo/v1
python3 -m pondlite.serve --run-dir runs/demo/v1 &      # 127.0.0.1:8420
python3 -m pondlite.web                                  # 127.0.0.1:8421 (UI)
```

Then:

```bash
curl -s localhost:8420/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"How do I cancel my subscription?"}]}'
```

Open http://localhost:8421 for the point-and-click version.

## Architecture

```
raw data (.jsonl/.csv/.txt)              trained artifacts
        │                                       │
        ▼                                       ▼
  pondlite/ingest.py ──► raw/ingested_*.jsonl   runs/<name>/v<n>/
        │                   │                     ├─ adapter/        (LoRA weights)
        ▼                   │                     ├─ metrics.json    (loss curve, ppl)
  pondlite/curate.py ───────┘                     ├─ eval.json       (held-out scores)
        │   dedupe + clean + PII scrub            ├─ progress.json   (live training state)
        ▼   train/holdout split                   └─ pondlite.db     (sqlite registry)
  data/<name>/v<n>/{train,holdout}.jsonl
        │       manifest.json
        ▼
  pondlite/train.py ──► LoRA SFT (peft + transformers, MPS/CPU)
        ▼
  pondlite/eval.py ───► loss/perplexity + generated-vs-reference table
        ▼
  pondlite/serve.py ──► FastAPI  :8420  POST /v1/chat/completions, GET /health
  pondlite/web.py ────► Flask    :8421  Ingest ▸ Curate ▸ Train ▸ Eval ▸ Serve
```

Everything is tracked in `pondlite.db` (sqlite): datasets, runs, evals.

## How it maps to OpenPond

| OpenPond | PondLite |
|---|---|
| **Dataset** (versioned, curated) | `data/<name>/v<n>/` — train/holdout split + `manifest.json` + curation report |
| **Experiment** (training run) | `runs/<name>/v<n>/` — LoRA adapter, loss curve, `progress.json` for live tracking |
| **Model** | trained adapter + base model pair, loadable by `eval.py` / `serve.py` |
| **Serve** | OpenAI-compatible `/v1/chat/completions` on port 8420 |

## Pipeline steps

1. **Ingest** — `python3 -m pondlite.ingest <files...>`. Parses `.jsonl` (instruction/input/output),
   `.csv` (columns), `.txt` (`Q:`/`A:` blocks). Reports every parse error; nothing is dropped silently.
2. **Curate** — exact-hash dedupe, fuzzy near-dupe removal (0.92 threshold), whitespace cleanup,
   PII scrub (emails → `[EMAIL]`, phones → `[PHONE]`), 90/10 train/holdout split, versioned output.
3. **Train** — LoRA (r=8, α=16 on GPT-2 `c_attn`) SFT with masked-padding cross-entropy.
   Writes `progress.json` every few steps so the web UI can poll live progress.
4. **Eval** — held-out loss/perplexity + greedy generations vs references, saved to `eval.json`.
5. **Serve** — `POST /v1/runs {"run_dir": ...}` to hot-swap a model, chat at `/v1/chat/completions`.

## Notes & limitations

- Tiny model on purpose: `sshleifer/tiny-gpt2` (~50MB) so 3 epochs finish in minutes on M3/MPS.
  Pass `--model distil` for `distilgpt2` (~350MB) — noticeably better, slower.
- Output quality is demo-grade: 80 training examples makes a pattern-matcher, not a support bot.
- `--run-dir` on `serve.py` must point at a completed run (`metrics.json` present).
- Runtime and development dependencies are pinned in `requirements.txt` and `requirements-dev.txt`.
- Training runs are registered as `running` before model loading, then marked `done` or `failed`.
  Failures write full tracebacks to `runs/<name>/v<n>/error.log`; web-spawned process output goes to
  `runs/.logs/`. The Train page shows active and failed runs.
- Both servers bind `127.0.0.1` by default. `--host 0.0.0.0` is available for trusted networks, but
  the UI has no authentication and should not be exposed publicly.
- Run loading accepts completed artifacts under `runs/` only. Dataset names and upload filenames are
  validated, and dynamic values in the UI are HTML-escaped.

## Development

```bash
python3 -m pip install --user -r requirements-dev.txt
python3 -m pytest
python3 -m pytest --cov=pondlite --cov-report=term-missing
```

The tests use temporary registries and fake small models for fast unit/integration coverage. They do
not replace a real training run or a fresh-process API smoke test. `/health` exposes a
`code_fingerprint` so callers can verify that a running server matches the local source.
