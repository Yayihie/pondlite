"""Curate: dedupe (exact hash + fuzzy), clean, scrub PII, split train/holdout,
emit versioned dataset dir data/<name>/v<n>/ with manifest.json.
"""

import argparse
import hashlib
import json
import random
import re
from difflib import SequenceMatcher
from pathlib import Path

from .store import dataset_dir, dump_jsonl, db, now, validate_name

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}")
URL_RE = re.compile(r"https?://\S+")


def scrub_text(text: str):
    """Scrub PII-ish patterns. Returns (clean_text, n_scrubbed)."""
    n = 0
    for pattern, repl in ((EMAIL_RE, "[EMAIL]"), (PHONE_RE, "[PHONE]")):
        matches = len(pattern.findall(text))
        if matches:
            text = pattern.sub(repl, text)
            n += matches
    return text, n


def clean_record(r: dict):
    """Clean one record. Returns (record|None, reason_if_discarded, pii_scrubs)."""
    pii = 0
    for key in ("instruction", "input", "output"):
        if r.get(key):
            r[key], n = scrub_text(r[key])
            pii += n
    instruction, output = r.get("instruction", ""), r.get("output", "")
    if not instruction or not output:
        return None, "empty instruction or output", pii
    if len(instruction) < 5 or len(output) < 2:
        return None, "too short", pii
    r["instruction"] = re.sub(r"\s+", " ", instruction).strip()
    r["output"] = re.sub(r"\s+", " ", output).strip()
    if r.get("input"):
        r["input"] = re.sub(r"\s+", " ", r["input"]).strip()
    return r, None, pii


def fuzzy_key(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def curate(records, fuzzy_threshold: float = 0.92):
    """Returns (kept, report). Dedupe by exact hash then fuzzy near-duplicates."""
    report = {"in": len(records), "discarded": [], "pii_scrubs": 0,
              "exact_dupes": 0, "fuzzy_dupes": 0}
    cleaned, seen_hash = [], set()
    for i, r in enumerate(records):
        r = dict(r)
        r.pop("_dropped_fields", None)
        out, reason, pii = clean_record(r)
        report["pii_scrubs"] += pii
        if out is None:
            report["discarded"].append({"index": i, "reason": reason, "instruction": (r.get("instruction") or "")[:80]})
            continue
        h = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
        if h in seen_hash:
            report["exact_dupes"] += 1
            continue
        seen_hash.add(h)
        cleaned.append(out)

    kept = []
    for r in cleaned:
        dup = False
        for k in kept:
            if fuzzy_key(r["instruction"], k["instruction"]) >= fuzzy_threshold:
                dup = True
                break
        if dup:
            report["fuzzy_dupes"] += 1
        else:
            kept.append(r)
    report["kept"] = len(kept)
    return kept, report


def run_curate(root: Path, name: str, records, holdout_frac: float = 0.1, seed: int = 42):
    name = validate_name(name)
    if not 0 < holdout_frac < 1:
        raise ValueError("holdout fraction must be between 0 and 1")
    kept, report = curate(records)
    if len(kept) < 4:
        raise ValueError(f"too few records after curation ({len(kept)}); need at least 4")
    rng = random.Random(seed)
    shuffled = kept[:]
    rng.shuffle(shuffled)
    n_hold = max(1, round(len(shuffled) * holdout_frac))
    holdout, train = shuffled[:n_hold], shuffled[n_hold:]

    from .store import next_version
    version = next_version(root, name, "data")
    ddir = dataset_dir(root, name, version)
    ddir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(ddir / "train.jsonl", train)
    dump_jsonl(ddir / "holdout.jsonl", holdout)
    manifest = {
        "name": name, "version": version, "created_at": now(),
        "n_train": len(train), "n_holdout": len(holdout),
        "holdout_frac": holdout_frac, "seed": seed, "curation_report": report,
    }
    (ddir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    conn = db()
    conn.execute(
        "INSERT INTO datasets(name, version, dir, n_records, n_holdout, created_at) VALUES(?,?,?,?,?,?)",
        (name, version, str(ddir), len(train), len(holdout), now()),
    )
    conn.commit()
    conn.close()
    return ddir, manifest


def main():
    ap = argparse.ArgumentParser(description="PondLite curate")
    ap.add_argument("--from-ingested", action="store_true", help="use raw/ingested.jsonl")
    ap.add_argument("--input", help="jsonl of normalized records to curate directly")
    ap.add_argument("--name", required=True)
    ap.add_argument("--holdout", type=float, default=0.1)
    args = ap.parse_args()

    if args.from_ingested:
        src = Path("raw/ingested.jsonl")
        records = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    elif args.input:
        records = [json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip()]
    else:
        ap.error("need --from-ingested or --input")

    root = Path(__file__).resolve().parent.parent
    ddir, manifest = run_curate(root, args.name, records, args.holdout)
    print(json.dumps({"dataset_dir": str(ddir), **manifest}, indent=2))


if __name__ == "__main__":
    main()
