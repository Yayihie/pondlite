"""Ingest raw data (.jsonl / .csv / .txt pairs) into normalized records.

Normalized record: {"instruction": str, "input": str|None, "output": str}
Reports parse errors; never silently drops malformed data.
"""

import csv
import json
import sys
from pathlib import Path

REQUIRED = {"instruction", "output"}
KNOWN_FIELDS = {"instruction", "input", "output"}


def _norm(instruction, inp, output):
    if not isinstance(instruction, str) or not isinstance(output, str) or (
        inp is not None and not isinstance(inp, str)
    ):
        raise TypeError("instruction, input, and output must be strings")
    return {
        "instruction": (instruction or "").strip(),
        "input": (inp or "").strip() or None,
        "output": (output or "").strip(),
    }


def ingest_file(path: Path):
    """Returns (records, errors, stats)."""
    records, errors = [], []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"not found: {p}")
    suffix = p.suffix.lower()

    if suffix == ".jsonl":
        with open(p, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append({"file": str(p), "line": i, "error": f"invalid JSON: {e}"})
                    continue
                if not isinstance(obj, dict):
                    errors.append({"file": str(p), "line": i, "error": "not a JSON object"})
                    continue
                missing = REQUIRED - set(obj)
                if missing:
                    errors.append({"file": str(p), "line": i, "error": f"missing fields {sorted(missing)}"})
                    continue
                extra = set(obj) - KNOWN_FIELDS
                try:
                    r = _norm(obj.get("instruction"), obj.get("input"), obj.get("output"))
                except TypeError as e:
                    errors.append({"file": str(p), "line": i, "error": str(e)})
                    continue
                if extra:
                    r["_dropped_fields"] = sorted(extra)
                records.append(r)

    elif suffix == ".csv":
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, 2):  # header is line 1
                missing = REQUIRED - set(row)
                if missing:
                    errors.append({"file": str(p), "line": i, "error": f"missing columns {sorted(missing)}"})
                    continue
                records.append(_norm(row.get("instruction"), row.get("input"), row.get("output")))

    elif suffix == ".txt":
        # Pair format: blocks separated by blank lines, "Q: ...\nA: ..." per block.
        text = p.read_text(encoding="utf-8")
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        for i, block in enumerate(blocks, 1):
            q_lines, a_lines, mode = [], [], None
            for line in block.splitlines():
                ls = line.strip()
                if ls.lower().startswith("q:"):
                    mode, q_lines = "q", [ls[2:].strip()]
                elif ls.lower().startswith("a:"):
                    mode, a_lines = "a", [ls[2:].strip()]
                elif ls:
                    (q_lines if mode == "q" else a_lines if mode == "a" else None) and (
                        q_lines if mode == "q" else a_lines
                    ).append(ls)
            if q_lines and a_lines:
                records.append(_norm(" ".join(q_lines), None, " ".join(a_lines)))
            else:
                errors.append({"file": str(p), "block": i, "error": "block missing Q:/A: pair"})
    else:
        raise ValueError(f"unsupported file type: {suffix} (use .jsonl, .csv, or .txt)")

    stats = {"file": str(p), "records": len(records), "errors": len(errors)}
    return records, errors, stats


def ingest(paths):
    """Ingest one or more files. Returns (all_records, all_errors, stats_list)."""
    all_records, all_errors, stats_list = [], [], []
    for path in paths:
        records, errors, stats = ingest_file(Path(path))
        all_records.extend(records)
        all_errors.extend(errors)
        stats_list.append(stats)
    return all_records, all_errors, stats_list


def main():
    import argparse

    ap = argparse.ArgumentParser(description="PondLite ingest")
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--from-artex", metavar="PATH", default=None,
                    help="append an ARTEX export JSONL (from scripts/export-artex.py) to the paths")
    args = ap.parse_args()
    paths = list(args.paths)
    if args.from_artex:
        paths.append(args.from_artex)
    if not paths:
        ap.error("no paths given; pass files or --from-artex PATH")
    records, errors, stats = ingest(paths)
    print(json.dumps({"stats": stats, "records": len(records), "errors": errors}, indent=2, ensure_ascii=False))
    if records:
        out = Path("raw") / "ingested.jsonl"
        out.parent.mkdir(exist_ok=True)
        with open(out, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {out} ({len(records)} records)")


if __name__ == "__main__":
    main()
