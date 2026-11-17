"""PondLite web UI: authenticated, responsive pipeline workspace."""

import argparse
import base64
import hmac
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename

from .auth import auth_enabled, require_production_config, valid_credentials
from .store import db, dataset_dir, load_jsonl, resolve_run_dir, validate_name

ROOT = Path(__file__).resolve().parent.parent
app = Flask(__name__)
app.config.update(
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("PONDLITE_SECURE_COOKIES") == "1",
)
app.secret_key = os.environ.get("PONDLITE_SECRET_KEY") or secrets.token_hex(32)


@app.context_processor
def template_globals():
    token = ""
    if auth_enabled():
        token = session.setdefault("_csrf_token", secrets.token_urlsafe(32))
    return {
        "auth_enabled": auth_enabled(),
        "csrf_token": token,
        "current_user": session.get("username", ""),
    }


@app.before_request
def protect_workspace():
    if auth_enabled() and request.method == "POST":
        expected = session.get("_csrf_token", "")
        supplied = request.form.get("_csrf_token", "")
        if not expected or not hmac.compare_digest(expected, supplied):
            return error_page(request.endpoint or "ingest", "Your form session expired. Reload and try again.", 400)
    if request.endpoint in {"login", "static", None}:
        return None
    if auth_enabled() and not session.get("username"):
        if request.method == "GET":
            return redirect(url_for("login", next=request.path))
        return error_page("ingest", "Sign in required.", 401)
    return None


@app.after_request
def security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' data:; style-src 'self'; "
        "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    )
    return response


def error_page(active, message, status=400):
    return render_template("error.html", active=active, message=message), status


def runs_db():
    conn = db()
    rows = [dict(r) for r in conn.execute("SELECT * FROM runs ORDER BY id DESC")]
    conn.close()
    return rows


def datasets_db():
    conn = db()
    rows = [dict(r) for r in conn.execute("SELECT * FROM datasets ORDER BY id DESC")]
    conn.close()
    return rows


def spawn_logged(command, prefix):
    log_dir = ROOT / "runs" / ".logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{prefix}-{time.time_ns()}.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
    return proc, log_path


def api_open(request_or_url, timeout):
    request_obj = request_or_url
    if auth_enabled():
        if isinstance(request_obj, str):
            request_obj = urllib.request.Request(request_or_url)
        credentials = base64.b64encode(
            f"{os.environ.get('PONDLITE_USERNAME', '')}:{os.environ.get('PONDLITE_PASSWORD', '')}".encode()
        ).decode()
        request_obj.add_header("Authorization", f"Basic {credentials}")
    return urllib.request.urlopen(request_obj, timeout=timeout)


def load_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def stored_run_path(value):
    """Rebase legacy absolute run paths into the active project root."""
    path = Path(value)
    if path.exists():
        return path
    parts = path.parts
    if "runs" in parts:
        index = parts.index("runs")
        return ROOT.joinpath(*parts[index:])
    return path


@app.route("/")
def index():
    return redirect(url_for("ingest"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        if valid_credentials(username, request.form.get("password", "")):
            session.clear()
            session["username"] = username
            session["_csrf_token"] = secrets.token_urlsafe(32)
            destination = request.form.get("next") or url_for("index")
            if not destination.startswith("/") or destination.startswith("//"):
                destination = url_for("index")
            return redirect(destination)
        error = "Those credentials didn’t work."
        return render_template("login.html", error=error, next=request.form.get("next", "/")), 401
    return render_template("login.html", error=error, next=request.args.get("next", "/"))


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/ingest", methods=["GET", "POST"])
def ingest():
    notice = None
    if request.method == "POST":
        try:
            name = validate_name(request.form.get("name") or "demo")
        except ValueError as exc:
            return error_page("ingest", str(exc))
        from .ingest import ingest as do_ingest
        raw = request.form.get("data", "").strip()
        tmp = ROOT / "raw" / f"upload_{name}"
        tmp.mkdir(parents=True, exist_ok=True)
        if request.files.get("file") and request.files["file"].filename:
            upload = request.files["file"]
            filename = secure_filename(upload.filename)
            if not filename or filename != upload.filename:
                return error_page("ingest", "Use a plain filename without paths.")
            path = tmp / filename
            upload.save(str(path))
        elif raw:
            path = tmp / "pasted.jsonl"
            path.write_text(raw)
        else:
            return error_page("ingest", "Paste JSONL or choose a file to continue.")
        try:
            records, errors, stats = do_ingest([str(path)])
        except (OSError, ValueError) as exc:
            return error_page("ingest", f"Ingest failed: {exc}")
        out = ROOT / "raw" / f"ingested_{name}.jsonl"
        with out.open("w") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        notice = {
            "title": f"{len(records)} records ready",
            "detail": f"{len(errors)} errors · saved to {out.name}",
            "errors": errors[:5],
        }
    return render_template("ingest.html", active="ingest", notice=notice, datasets=datasets_db())


@app.route("/curate", methods=["GET", "POST"])
def curate():
    notice = None
    if request.method == "POST":
        from .curate import run_curate
        try:
            name = validate_name(request.form.get("name") or "demo")
            frac = float(request.form.get("holdout") or 10) / 100.0
            src = ROOT / "raw" / f"ingested_{name}.jsonl"
            if not src.exists():
                src = ROOT / "raw" / "ingested.jsonl"
            if not src.exists():
                return error_page("curate", "Ingest this dataset first.")
            records = load_jsonl(src)
            ddir, manifest = run_curate(ROOT, name, records, holdout_frac=frac)
        except (OSError, ValueError) as exc:
            return error_page("curate", f"Curation failed: {exc}")
        report = manifest["curation_report"]
        notice = {"title": f"Dataset {ddir.name} ready", "detail": (
            f"{report['kept']} kept · {report['exact_dupes'] + report['fuzzy_dupes']} duplicates removed · "
            f"{manifest['n_train']} train / {manifest['n_holdout']} holdout"
        )}
    latest = datasets_db()[:1]
    review, dataset = [], None
    if latest:
        row = latest[0]
        dataset = dataset_dir(ROOT, row["name"], row["version"])
        review = load_jsonl(dataset / "train.jsonl")[:12]
    return render_template("curate.html", active="curate", notice=notice,
                           review=review, dataset=dataset, dataset_row=latest[0] if latest else None)


@app.route("/train", methods=["GET", "POST"])
def train_page():
    notice = None
    datasets = datasets_db()
    if request.method == "POST":
        try:
            name = validate_name(request.form.get("name") or "demo")
            epochs = int(request.form.get("epochs") or 3)
            model = request.form.get("model") or "distil"
            if model not in {"tiny", "distil", "gpt2"}:
                raise ValueError("Choose a supported base model.")
            if not 1 <= epochs <= 100:
                raise ValueError("Epochs must be between 1 and 100.")
        except ValueError as exc:
            return error_page("train", str(exc))
        matches = [row for row in datasets if row["name"] == name]
        if not matches:
            return error_page("train", f"No dataset named {name!r}. Curate it first.")
        row = matches[0]
        ddir = dataset_dir(ROOT, row["name"], row["version"])
        proc, log_path = spawn_logged(
            [sys.executable, "-m", "pondlite.train", "--model", model,
             "--dataset", str(ddir), "--name", name, "--epochs", str(epochs)], f"train-{name}"
        )
        notice = {"title": "Training started", "detail": f"PID {proc.pid} · log {log_path.name}"}
    return render_template("train.html", active="train", notice=notice,
                           datasets=datasets, runs=runs_db()[:8])


@app.route("/train/status")
def train_status():
    parts = []
    for row in runs_db()[:5]:
        path = stored_run_path(row["run_dir"]) / "progress.json"
        progress = load_json(path, {"status": row["status"]})
        from markupsafe import escape
        line = (f'<b>{escape(row["name"])} v{row["version"]}</b>: status={escape(progress["status"])}, '
                f'step={escape(progress.get("step", "-"))}/{escape(progress.get("total_steps", "-"))}, '
                f'loss={escape(progress.get("latest_loss", "-"))}, '
                f'holdout={escape(progress.get("holdout_loss", progress.get("latest_holdout_loss", "-")))}, '
                f'{escape(progress.get("elapsed_s", "-"))}s on {escape(progress.get("device", "?"))}')
        if progress.get("error"):
            line += f'<br><span class=error>{escape(progress["error"])} — see {escape(str(path.parent / "error.log"))}</span>'
        parts.append(line)
    return jsonify({"html": "<br>".join(str(part) for part in parts) or "no runs yet"})


@app.route("/eval", methods=["GET", "POST"])
def eval_page():
    if request.method == "POST":
        try:
            run_dir = resolve_run_dir(ROOT, request.form.get("run_dir", ""))
        except ValueError as exc:
            return error_page("eval", str(exc))
        (run_dir / "eval_progress.json").write_text(json.dumps({"status": "running", "updated_at": time.time()}))
        proc, log_path = spawn_logged(
            [sys.executable, "-m", "pondlite.eval", "--run-dir", str(run_dir)], "eval"
        )
        return redirect(url_for("eval_page", run=str(run_dir)), code=303)

    run_views = []
    for row in runs_db():
        run_dir = stored_run_path(row["run_dir"])
        if not (run_dir / "metrics.json").is_file():
            continue
        run_views.append({
            "row": row,
            "run_dir": str(run_dir),
            "label": f"{row['name']} v{row['version']}",
            "evaluation": load_json(run_dir / "eval.json"),
            "progress": load_json(run_dir / "eval_progress.json"),
        })
    requested_run = request.args.get("run")
    if requested_run:
        try:
            selected_dir = str(resolve_run_dir(ROOT, requested_run))
        except ValueError as exc:
            return error_page("eval", str(exc))
    else:
        selected_dir = run_views[0]["run_dir"] if run_views else ""
    selected = next((view for view in run_views if view["run_dir"] == selected_dir), None)
    return render_template("eval.html", active="eval", runs=run_views, selected=selected)


@app.route("/serve", methods=["GET", "POST"])
def serve_page():
    reply, notice, health = None, None, None
    run_views = [
        {**row, "run_dir": str(stored_run_path(row["run_dir"]))}
        for row in runs_db() if (stored_run_path(row["run_dir"]) / "metrics.json").is_file()
    ]
    if request.method == "POST":
        if request.form.get("run_dir"):
            try:
                run_dir = resolve_run_dir(ROOT, request.form["run_dir"].strip())
            except ValueError as exc:
                return error_page("serve", str(exc))
            body = json.dumps({"run_dir": str(run_dir)}).encode()
            api_request = urllib.request.Request(
                "http://127.0.0.1:8420/v1/runs", data=body, headers={"Content-Type": "application/json"}
            )
            try:
                api_open(api_request, timeout=90).read()
                notice = {"title": "Model loaded", "detail": f"{run_dir.parent.name}/{run_dir.name} is ready to answer."}
            except (OSError, ValueError):
                proc, log_path = spawn_logged(
                    [sys.executable, "-m", "pondlite.serve", "--run-dir", str(run_dir)], "serve"
                )
                notice = {"title": "Server starting", "detail": f"PID {proc.pid} · log {log_path.name}"}
        elif request.form.get("prompt"):
            prompt = request.form["prompt"]
            body = json.dumps({"messages": [{"role": "user", "content": prompt}]}).encode()
            try:
                api_request = urllib.request.Request(
                    "http://127.0.0.1:8420/v1/chat/completions", data=body,
                    headers={"Content-Type": "application/json"}
                )
                response = json.loads(api_open(api_request, timeout=60).read())
                reply = {
                    "text": response["choices"][0]["message"]["content"] or "(empty response)",
                    "meta": response.get("pondlite", {}),
                }
            except Exception as exc:
                return error_page("serve", f"The server could not answer: {exc}")
    try:
        health = json.loads(api_open("http://127.0.0.1:8420/health", timeout=3).read())
    except (OSError, ValueError):
        pass
    return render_template("serve.html", active="serve", runs=run_views, reply=reply,
                           notice=notice, health=health)


def main():
    require_production_config()
    parser = argparse.ArgumentParser(description="PondLite web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8421)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
