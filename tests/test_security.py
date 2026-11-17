import io
import json
import sys

from fastapi.testclient import TestClient

from pondlite import serve, store, web


VALID_JSONL = '{"instruction":"How do I cancel?","output":"Open Billing and choose Cancel."}\n'


def test_ingest_rejects_unsafe_dataset_name(isolated_project, monkeypatch):
    monkeypatch.setattr(web, "ROOT", isolated_project)
    response = web.app.test_client().post(
        "/ingest", data={"name": "../outside", "data": VALID_JSONL}
    )
    assert response.status_code == 400
    assert not (isolated_project / "outside.jsonl").exists()


def test_upload_filename_cannot_escape_upload_directory(isolated_project, monkeypatch):
    monkeypatch.setattr(web, "ROOT", isolated_project)
    response = web.app.test_client().post(
        "/ingest",
        data={
            "name": "demo",
            "file": (io.BytesIO(VALID_JSONL.encode()), "../../outside.jsonl"),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert not (isolated_project / "outside.jsonl").exists()
    assert not (isolated_project / "raw" / "outside.jsonl").exists()


def test_curate_review_escapes_record_html(isolated_project, monkeypatch):
    root = isolated_project
    monkeypatch.setattr(web, "ROOT", root)
    dataset = root / "data" / "demo" / "v1"
    dataset.mkdir(parents=True)
    record = {
        "instruction": "<script>alert('instruction')</script>",
        "input": "<img src=x onerror=alert(1)>",
        "output": "<script>alert('output')</script>",
    }
    (dataset / "train.jsonl").write_text(json.dumps(record) + "\n")
    (dataset / "holdout.jsonl").write_text(json.dumps(record) + "\n")
    conn = store.db()
    conn.execute(
        "INSERT INTO datasets(name, version, dir, n_records, n_holdout, created_at) VALUES(?,?,?,?,?,?)",
        ("demo", 1, str(dataset), 1, 1, store.now()),
    )
    conn.commit()
    conn.close()

    page = web.app.test_client().get("/curate").get_data(as_text=True)
    assert "<script>alert('instruction')</script>" not in page
    assert "&lt;script&gt;" in page
    assert "<img src=x onerror=alert(1)>" not in page


def test_run_loader_rejects_directory_outside_runs(isolated_project):
    outside = isolated_project / "outside"
    outside.mkdir()
    response = TestClient(serve.app).post("/v1/runs", json={"run_dir": str(outside)})
    assert response.status_code == 400


def test_pages_include_mobile_viewport():
    page = web.app.test_client().get("/ingest").get_data(as_text=True)
    assert '<meta name="viewport" content="width=device-width,initial-scale=1">' in page


def test_servers_default_to_loopback(monkeypatch):
    serve_call = {}
    web_call = {}
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: serve_call.update(kwargs))
    monkeypatch.setattr(web.app, "run", lambda *args, **kwargs: web_call.update(kwargs))
    monkeypatch.setattr(sys, "argv", ["pondlite.serve"])
    serve.main()
    monkeypatch.setattr(sys, "argv", ["pondlite.web"])
    web.main()

    assert serve_call["host"] == "127.0.0.1"
    assert web_call["host"] == "127.0.0.1"
