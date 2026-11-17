"""Export ARTEX findings, worker traces and exploration-graph facts as PondLite
ingest JSONL (instruction/input/output records).

Pulls from a local ARTEX server (default http://localhost:8787):

  POST /api/auth/login                       -> {"token": ...}  (username ARTEX)
  GET  /api/exploration/findings             -> bare array of FindingDTO
  GET  /api/exploration/findings/{id}/retests-> {"retests": [...]}  (verdict field)
  GET  /api/exploration/graph?task=<id>      -> {"nodes": [...], "edges": [...]}

Each verified finding becomes one training pair:

  instruction: what the model is asked to do (record the finding)
  input:       intent context + target context
  output:      finding write-back (fact) + the latest retest verdict

Graph nodes/edges for the finding's task become short "fact" pairs.

Usage:
  python3 scripts/export-artex.py --password SECRET [--out raw/artex.jsonl]
"""

import argparse
import json
import sys
import urllib.request

DEFAULT_BASE = "http://localhost:8787"
DEFAULT_USER = "ARTEX"

FINDING_INSTRUCTION = (
    "You are a web security testing assistant. Given the intent and target "
    "context, write back the verified finding and its retest verdict."
)
FACT_INSTRUCTION = (
    "You are a web security testing assistant. Given the intent and target "
    "context, write back the discovered exploration fact."
)


def _get(url, token=None):
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def login(base, username, password):
    return _post(base + "/api/auth/login", {"username": username, "password": password})["token"]


def latest_verdict(retests):
    """Most recent finished retest verdict ('reproduced' | 'fixed' | 'inconclusive')."""
    for r in sorted(retests, key=lambda x: x.get("created_at") or ""):
        if r.get("status") == "done" and r.get("verdict"):
            return r["verdict"]
    return None


def target_context(f):
    parts = []
    for a in f.get("assets") or []:
        parts.append(a.get("label") or a.get("id") or "")
    return "; ".join(p for p in parts if p)


def finding_record(f, verdict):
    target = target_context(f)
    lines = [
        "Intent: " + (f.get("task_description") or f.get("intent_id") or "web application penetration test"),
        "Target: " + (target or "unknown"),
        "Vulnerability class: " + (f.get("vulnclass") or "unknown"),
        "Severity: " + (f.get("severity") or "unknown"),
    ]
    out = [
        "Finding: " + (f.get("name") or f.get("vulnclass") or "unknown"),
        "Severity: " + (f.get("severity") or "unknown"),
    ]
    if f.get("summary"):
        out.append("Summary: " + f["summary"])
    if f.get("evidence"):
        out.append("Evidence: " + f["evidence"].strip())
    if verdict:
        out.append("Retest verdict: " + verdict)
    return {
        "instruction": FINDING_INSTRUCTION,
        "input": "\n".join(lines),
        "output": "\n".join(out),
    }


def graph_records(task_id, task_description, graph):
    """Exploration-graph nodes/edges as short fact pairs (capped to keep data small)."""
    records = []
    for node in (graph.get("nodes") or [])[:100]:
        text = json.dumps(node, ensure_ascii=False)
        records.append({
            "instruction": FACT_INSTRUCTION,
            "input": "Intent: %s\nTarget task: %s" % (task_description or "web application penetration test", task_id),
            "output": "Fact: " + text,
        })
    for edge in (graph.get("edges") or [])[:100]:
        text = json.dumps(edge, ensure_ascii=False)
        records.append({
            "instruction": FACT_INSTRUCTION,
            "input": "Intent: %s\nTarget task: %s" % (task_description or "web application penetration test", task_id),
            "output": "Fact: " + text,
        })
    return records


def export(base, username, password, token, status_filter, out_path):
    if not token:
        token = login(base, username, password)
    findings = _get(base + "/api/exploration/findings", token)
    if not isinstance(findings, list):
        findings = findings.get("items", [])

    graph_cache = {}
    records = []
    for f in findings:
        if status_filter and f.get("status") != status_filter:
            continue
        # Latest retest verdict feeds the output; failures are skipped, not hidden.
        try:
            retests = _get("%s/api/exploration/findings/%s/retests" % (base, f["id"]), token)
            verdict = latest_verdict(retests.get("retests", []))
        except Exception as e:
            print("warn: retests for finding %s failed: %s" % (f.get("id"), e), file=sys.stderr)
            verdict = None
        records.append(finding_record(f, verdict))

        task_id = f.get("task_id") or ""
        if task_id and task_id not in graph_cache:
            try:
                graph_cache[task_id] = _get(base + "/api/exploration/graph?task=" + task_id, token)
            except Exception as e:
                print("warn: graph for task %s failed: %s" % (task_id, e), file=sys.stderr)
                graph_cache[task_id] = {"nodes": [], "edges": []}
        records.extend(graph_records(task_id, f.get("task_description") or "", graph_cache[task_id]))

    with open(out_path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return records


def main():
    ap = argparse.ArgumentParser(description="Export ARTEX findings as PondLite training pairs")
    ap.add_argument("--base", default=DEFAULT_BASE, help="ARTEX base URL (default %(default)s)")
    ap.add_argument("--user", default=DEFAULT_USER, help="ARTEX username (default %(default)s)")
    ap.add_argument("--password", default=None, help="ARTEX password (or env ARTEX_PASSWORD)")
    ap.add_argument("--token", default=None, help="existing ARTEX JWT (skips login)")
    ap.add_argument("--status", default="confirmed", help="only export findings with this status (default %(default)s)")
    ap.add_argument("--out", default="raw/artex.jsonl", help="output JSONL path (default %(default)s)")
    args = ap.parse_args()

    import os
    password = args.password or os.environ.get("ARTEX_PASSWORD")
    if not password and not args.token:
        ap.error("--password (or ARTEX_PASSWORD) or --token is required")

    records = export(args.base, args.user, password, args.token, args.status, args.out)
    print("wrote %s (%d records)" % (args.out, len(records)))


if __name__ == "__main__":
    main()
