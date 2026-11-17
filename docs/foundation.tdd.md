# PondLite foundation TDD evidence

## Scope

Journeys derived from the initial project audit and the request to start building:

- As a user, I can see active and failed training runs so training is not a black box.
- As a user, I can inspect failure details instead of losing subprocess errors.
- As a local user, uploads, names, rendered records, and run paths cannot escape their intended boundaries.
- As a developer, I can install pinned dependencies and reproduce the test suite.
- As a developer, I can verify which source build a running API serves.

No plan file or Git repository was supplied; no checkpoint commits were created.

## RED evidence

- Initial `python3 -m pytest`: 9 failed, 3 passed. Product failures included invisible training registration, unsafe names/uploads, unescaped HTML, unrestricted run paths, non-loopback defaults, and invalid JSONL field types crashing ingest.
- Two fixture defects were corrected before implementation: PII is counted before dedupe, and numbered near-identical instructions were not valid unique curation fixtures.
- Evaluation path reproducer: 1 failed, 24 passed because a relative `dataset_dir` depended on caller cwd.
- Mobile viewport reproducer: 1 failed because the HTML had no viewport meta tag.
- Evaluation's later loss assertion was corrected to log(5) = 1.6094 for uniform five-token logits; this was a test-math error, not a product defect.

## GREEN evidence

Final command:

```bash
python3 -m pytest --cov=pondlite --cov-report=term-missing --cov-report=html
```

Result: 26 passed; 89% total statement coverage (756 statements, 85 missed). No skipped tests. The only warning is the macOS system Python LibreSSL warning from urllib3.

| Guarantee | Test area | Result |
|---|---|---|
| Active run exists during model loading; failure updates status, progress, and error.log | `tests/test_training_lifecycle.py` | PASS |
| Successful training uses latest existing dataset and updates one run row | `tests/test_training_lifecycle.py` | PASS |
| Names and upload filenames cannot escape storage paths | `tests/test_security.py` | PASS |
| Dataset HTML and training errors are escaped | `tests/test_security.py`, `tests/test_web_workflow.py` | PASS |
| Run loading rejects directories outside `runs/` and incomplete artifacts | `tests/test_security.py`, `tests/test_api.py` | PASS |
| Servers default to 127.0.0.1 | `tests/test_security.py` | PASS |
| API reports actual generated token counts and build fingerprint | `tests/test_api.py` | PASS |
| Evaluation loads a run and resolves relative dataset paths from the run | `tests/test_eval.py` | PASS |
| Ingest, curation, training launch, eval display, and serve UI workflows | `tests/test_web_workflow.py` | PASS |
| Child output is captured in a real log file | `tests/test_web_workflow.py` | PASS |
| Mobile viewport is declared | `tests/test_security.py` | PASS |

## Real-process verification

- Fresh API health fingerprint matched disk: `e2bb40a8b4e2265e`.
- Real tiny-gpt2 run `runs/demo/v6` was observed in SQLite as `running`, then `done`: 18 steps on MPS, 13.8 seconds, holdout loss 10.8246.
- Real evaluation of v6 completed with the same loss/perplexity (10.8246 / 50240.0578).
- `/train/status` displayed v6 as done with 18/18 steps.
- Real chat on port 8420 completed with 16 prompt + 34 completion = 50 total tokens.
- API rejected `/tmp` as a run directory with HTTP 400.
- Browser sweep loaded Ingest, Curate, Train, Eval, and Serve with their expected content and no desktop horizontal overflow.
- Mobile browser verification after restart: innerWidth 390, clientWidth 390, scrollWidth 390.
- Normal services: API PID 91504 on 127.0.0.1:8420; UI PID 91505 on 127.0.0.1:8421.

## Known gaps

- Unit/integration tests use fake models for speed; one real tiny training run and real serving/eval were verified separately.
- The saved v5 model's answer quality remains demo-grade; this pass did not attempt model-quality tuning.
- No full accessibility audit, multi-user security model, or production WSGI deployment was added.
- The project is still not a Git repository.
