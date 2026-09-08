import csv
import hashlib
import json
from pathlib import Path


def test_supplied_pdf_provenance_and_reference_table_structure():
    paper = Path("dynamic_transformers_reference_paper.pdf")
    assert hashlib.sha256(paper.read_bytes()).hexdigest() == (
        "f9117b8504bb952edc5b05e87f74a5f3f72b3822fb96dc7c9373a0698be97ad4"
    )
    with Path("references/paper_results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 64
    assert len({(r["table"], r["model"], r["dataset"]) for r in rows}) == 64
    assert all(r["provenance"] == "authors_reported" and r["printed_page"] == "447" for r in rows)
    assert sum(r["table"] == "1" for r in rows) == 20
    target = next(
        r for r in rows if (r["table"], r["model"], r["dataset"]) == ("2", "T-DHT", "ag_news")
    )
    assert target["f1"] == "0.945" and target["flop_ratio"] == "0.81"


def test_published_special_options_are_preserved():
    settings = json.loads(Path("references/paper_settings.json").read_text())
    options = settings["P_DHT"]["published_options"]
    assert len(options) == 12
    assert "7L6H+2L12H" in options and "7L9H+2L12H" in options
