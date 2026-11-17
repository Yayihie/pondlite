import json

import pytest

from pondlite.curate import curate, run_curate, scrub_text
from pondlite.ingest import ingest_file
from pondlite.store import db, load_jsonl


def test_ingest_jsonl_reports_bad_rows_and_normalizes_valid_rows(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(
        '{"instruction":"  How do I cancel?  ","output":" Open Billing. ","id":7}\n'
        '{bad json}\n'
        '{"instruction":"Missing output"}\n'
        '["not", "an", "object"]\n'
    )
    records, errors, stats = ingest_file(path)
    assert records == [{
        "instruction": "How do I cancel?",
        "input": None,
        "output": "Open Billing.",
        "_dropped_fields": ["id"],
    }]
    assert len(errors) == 3
    assert stats == {"file": str(path), "records": 1, "errors": 3}


def test_ingest_reports_invalid_field_types_instead_of_crashing(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"instruction":123,"output":"Valid output"}\n')
    records, errors, stats = ingest_file(path)
    assert records == []
    assert len(errors) == 1
    assert "must be strings" in errors[0]["error"]
    assert stats["errors"] == 1


def test_ingest_csv_and_text_formats(tmp_path):
    csv_path = tmp_path / "records.csv"
    csv_path.write_text("instruction,input,output\nQuestion,Context,Answer\n")
    text_path = tmp_path / "records.txt"
    text_path.write_text("Q: First question\ncontinued\nA: First answer\n\nInvalid block\n")

    csv_records, csv_errors, _ = ingest_file(csv_path)
    text_records, text_errors, _ = ingest_file(text_path)
    assert csv_records == [{"instruction": "Question", "input": "Context", "output": "Answer"}]
    assert csv_errors == []
    assert text_records == [{"instruction": "First question continued", "input": None, "output": "First answer"}]
    assert len(text_errors) == 1


def test_curation_scrubs_deduplicates_and_discards():
    records = [
        {"instruction": "Email me at person@example.com", "output": "Call 415-555-0132."},
        {"instruction": "Email me at person@example.com", "output": "Call 415-555-0132."},
        {"instruction": "x", "output": "y"},
        {"instruction": "A different valid question", "output": "A valid answer"},
        {"instruction": "A different valid question?", "output": "A valid answer"},
    ]
    kept, report = curate(records)
    assert report["exact_dupes"] == 1
    assert report["fuzzy_dupes"] == 1
    assert report["pii_scrubs"] == 4  # both input rows are scrubbed before dedupe
    assert len(report["discarded"]) == 1
    assert kept[0]["instruction"] == "Email me at [EMAIL]"
    assert kept[0]["output"] == "Call [PHONE]."


def test_run_curate_writes_versioned_dataset_and_registry(isolated_project):
    instructions = [
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
    records = [
        {"instruction": instruction, "output": f"Answer number {index}"}
        for index, instruction in enumerate(instructions)
    ]
    dataset_dir, manifest = run_curate(isolated_project, "demo", records, holdout_frac=0.2)
    assert dataset_dir.name == "v1"
    assert manifest["n_train"] == 8
    assert manifest["n_holdout"] == 2
    assert len(load_jsonl(dataset_dir / "train.jsonl")) == 8
    assert len(load_jsonl(dataset_dir / "holdout.jsonl")) == 2
    conn = db()
    row = conn.execute("SELECT name, version, n_records, n_holdout FROM datasets").fetchone()
    conn.close()
    assert tuple(row) == ("demo", 1, 8, 2)


def test_scrub_text_leaves_normal_text_unchanged():
    assert scrub_text("No personal information here.") == ("No personal information here.", 0)
