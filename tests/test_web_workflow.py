import json
import sys
from types import SimpleNamespace

from pondlite import store, web


INSTRUCTIONS = [
    "How do I cancel my subscription?",
    "Where is my order right now?",
    "Which credit cards do you accept?",
    "How can I permanently delete my account?",
    "Do you offer international shipping?",
    "Why am I unable to sign in?",
    "Is there a mobile application?",
    "My package arrived damaged",
    "Why did the monthly price increase?",
    "How do I change my billing address?",
]


def insert_run(run_dir, status="done"):
    conn = store.db()
    conn.execute(
        """INSERT INTO runs(name, version, dataset_dir, run_dir, backend, epochs, status,
           final_loss, eval_loss, perplexity, created_at, updated_at)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
        ("demo", 1, "data/demo/v1", str(run_dir), "peft-lora/cpu", 1, status,
         1.0, 1.0, 2.7, store.now(), store.now()),
    )
    conn.commit()
    conn.close()


def test_ingest_curate_and_launch_training(isolated_project, monkeypatch):
    root = isolated_project
    monkeypatch.setattr(web, "ROOT", root)
    client = web.app.test_client()
    raw = "\n".join(json.dumps({
        "instruction": instruction, "output": f"Answer number {index}"
    }) for index, instruction in enumerate(INSTRUCTIONS))

    response = client.post("/ingest", data={"name": "demo", "data": raw})
    assert response.status_code == 200
    assert (root / "raw" / "ingested_demo.jsonl").is_file()

    response = client.post("/curate", data={"name": "demo", "holdout": "20"})
    assert response.status_code == 200
    assert (root / "data" / "demo" / "v1" / "train.jsonl").is_file()
    assert (root / "data" / "demo" / "v1" / "holdout.jsonl").is_file()

    launched = {}

    def fake_spawn(command, prefix):
        launched["command"] = command
        return SimpleNamespace(pid=4242), root / "runs" / ".logs" / "train.log"

    monkeypatch.setattr(web, "spawn_logged", fake_spawn)
    response = client.post("/train", data={"name": "demo", "epochs": "1"})
    assert response.status_code == 200
    assert "4242" in response.get_data(as_text=True)
    assert launched["command"][-4:] == ["--name", "demo", "--epochs", "1"]
    assert client.get("/train/status").get_json()["html"] == "no runs yet"
    assert client.post("/train", data={"name": "demo", "epochs": "0"}).status_code == 400
    assert client.post("/train", data={"name": "missing", "epochs": "1"}).status_code == 400


def test_training_status_and_eval_render_results(isolated_project, monkeypatch):
    root = isolated_project
    monkeypatch.setattr(web, "ROOT", root)
    run_dir = root / "runs" / "demo" / "v1"
    run_dir.mkdir(parents=True)
    insert_run(run_dir, status="failed")
    (run_dir / "metrics.json").write_text("{}")
    (run_dir / "progress.json").write_text(json.dumps({
        "status": "failed", "step": 2, "total_steps": 4, "latest_loss": 2.5,
        "latest_holdout_loss": 3.0, "elapsed_s": 10, "device": "cpu",
        "error": "bad <script>alert(1)</script>",
    }))
    (run_dir / "eval.json").write_text(json.dumps({
        "eval_loss": 1.0, "perplexity": 2.7, "greedy_prefix_matches": 1, "n_samples": 1,
        "samples": [{
            "instruction": "Question <script>alert(1)</script>",
            "reference": "Answer", "generated": "Answer", "exact_match_on_prefix": True,
        }],
    }))

    client = web.app.test_client()
    status = client.get("/train/status").get_json()["html"]
    assert "status=failed" in status
    assert "bad &lt;script&gt;" in status
    assert "error.log" in status
    page = client.get("/eval").get_data(as_text=True)
    assert "greedy prefix matches 1/1" in page
    assert "Question &lt;script&gt;" in page
    assert "Question <script>" not in page


def test_serve_page_chat_health_and_run_launch(isolated_project, monkeypatch):
    root = isolated_project
    monkeypatch.setattr(web, "ROOT", root)
    run_dir = root / "runs" / "demo" / "v1"
    (run_dir / "adapter").mkdir(parents=True)
    (run_dir / "metrics.json").write_text("{}")

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def read(self):
            return json.dumps(self.payload).encode()

    def fake_urlopen(request, timeout):
        if isinstance(request, str):
            return FakeResponse({"status": "ok", "code_fingerprint": "test-build"})
        return FakeResponse({"choices": [{"message": {"content": "safe <script>alert(1)</script>"}}]})

    monkeypatch.setattr(web.urllib.request, "urlopen", fake_urlopen)
    launched = {}

    def fake_spawn(command, prefix):
        launched["command"] = command
        return SimpleNamespace(pid=555), root / "runs" / ".logs" / "serve.log"

    monkeypatch.setattr(web, "spawn_logged", fake_spawn)
    client = web.app.test_client()
    assert "test-build" in client.get("/serve").get_data(as_text=True)
    chat = client.post("/serve", data={"prompt": "Hello"}).get_data(as_text=True)
    assert "safe &lt;script&gt;" in chat
    assert "safe <script>" not in chat
    started = client.post("/serve", data={"run_dir": "runs/demo/v1"})
    assert started.status_code == 200
    assert "Model loaded" in started.get_data(as_text=True)
    assert "command" not in launched  # live API hot-swap must not start a duplicate server
    assert client.post("/serve", data={"run_dir": str(root / "outside")}).status_code == 400


def test_spawn_logged_captures_real_child_output(isolated_project, monkeypatch):
    monkeypatch.setattr(web, "ROOT", isolated_project)
    proc, log_path = web.spawn_logged(
        [sys.executable, "-c", "print('child output')"], "test"
    )
    assert proc.wait(timeout=30) == 0
    assert log_path.read_text().strip() == "child output"


def test_one_click_eval_launches_child_and_shows_progress(isolated_project, monkeypatch):
    root = isolated_project
    monkeypatch.setattr(web, "ROOT", root)
    run_dir = root / "runs" / "demo" / "v1"
    (run_dir / "adapter").mkdir(parents=True)
    (run_dir / "metrics.json").write_text("{}")
    insert_run(run_dir)
    launched = {}

    def fake_spawn(command, prefix):
        launched["command"] = command
        return SimpleNamespace(pid=777), root / "runs" / ".logs" / "eval.log"

    monkeypatch.setattr(web, "spawn_logged", fake_spawn)
    client = web.app.test_client()
    response = client.post("/eval", data={"run_dir": "runs/demo/v1"})
    assert response.status_code == 303
    assert launched["command"][:3] == [sys.executable, "-m", "pondlite.eval"]
    progress = json.loads((run_dir / "eval_progress.json").read_text())
    assert progress["status"] == "running"
    assert "running" in client.get("/eval?run=runs/demo/v1").get_data(as_text=True)


def test_product_shell_uses_external_design_system():
    response = web.app.test_client().get("/ingest")
    page = response.get_data(as_text=True)
    assert '/static/styles.css' in page
    assert 'class="app-shell"' in page
    assert "Small data. Useful models." in page
    assert response.headers["Content-Security-Policy"]


def test_legacy_run_path_is_rebased_to_project_runs(isolated_project, monkeypatch):
    monkeypatch.setattr(web, "ROOT", isolated_project)
    rebased = web.stored_run_path("/old/host/pondlite/runs/demo/v7")
    assert rebased == isolated_project / "runs" / "demo" / "v7"


def test_index_redirects_to_ingest():
    response = web.app.test_client().get("/")
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/ingest")
