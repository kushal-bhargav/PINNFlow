from __future__ import annotations

from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinnflow.codal_engine.knowledge.fetcher import CodalFetcher
from pinnflow.codal_engine.knowledge.parser import CodalParser
from pinnflow.codal_engine.knowledge.rule_store import CodalRuleStore


class _FakePage:
    def __init__(self, text: str):
        self._text = text

    def extract_text(self):
        return self._text


class _FakePdf:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_fetcher_extracts_pdf_text_and_parser_detects_rules(tmp_path, monkeypatch):
    pdf_path = tmp_path / "ASME_B31_3.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake content")

    fake_pdfplumber = types.SimpleNamespace(
        open=lambda _path: _FakePdf(
            [
                _FakePage("ASME\nB31.3 Section 304.1.2"),
                _FakePage("API 14E Section 2.4a"),
            ]
        )
    )
    monkeypatch.setitem(sys.modules, "pdfplumber", fake_pdfplumber)

    fetcher = CodalFetcher(storage_path=str(tmp_path / "standards"))
    text = fetcher.fetch_local_pdf(str(pdf_path))

    parser = CodalParser()
    rules = parser.parse_text_to_rules(text)
    codes = {rule["code"] for rule in rules}

    assert "ASME B31.3" in codes
    assert "API 14E" in codes


def test_fetcher_raises_for_missing_pdf(tmp_path):
    fetcher = CodalFetcher(storage_path=str(tmp_path / "standards"))
    missing_pdf = tmp_path / "missing.pdf"

    try:
        fetcher.fetch_local_pdf(str(missing_pdf))
    except FileNotFoundError:
        return

    raise AssertionError("Expected FileNotFoundError for a missing standards PDF")


def test_fetcher_resolves_api_14e_alias_to_packaged_api_4e(tmp_path, monkeypatch):
    alias_target = tmp_path / "API_4E.pdf"
    alias_target.write_bytes(b"%PDF-1.4 aliased content")

    fake_pdfplumber = types.SimpleNamespace(
        open=lambda _path: _FakePdf(
            [
                _FakePage("API 14E Section 2.4a"),
            ]
        )
    )
    monkeypatch.setitem(sys.modules, "pdfplumber", fake_pdfplumber)

    fetcher = CodalFetcher(storage_path=str(tmp_path / "standards"))

    text = fetcher.fetch_local_pdf(str(tmp_path / "API_14E.pdf"))

    assert "API 14E" in text


def test_fetcher_uses_ocr_fallback_when_text_extraction_is_empty(tmp_path, monkeypatch):
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake scanned content")

    fetcher = CodalFetcher(storage_path=str(tmp_path / "standards"))
    monkeypatch.setattr(fetcher, "_extract_pdf_text", lambda _target: "")
    monkeypatch.setattr(fetcher, "_ocr_pdf_text", lambda _target: "ASME B31.3 OCR text with limit 180 MPa")

    text = fetcher.fetch_local_pdf(str(pdf_path))

    assert "ASME B31.3" in text
    assert "180" in text


def test_parser_handles_split_asme_marker():
    parser = CodalParser()

    rules = parser.parse_text_to_rules("ASME\nB31.3\nwall thickness guidance")
    codes = {rule["code"] for rule in rules}

    assert "ASME B31.3" in codes


def test_parser_extracts_multiple_section_scoped_rules():
    parser = CodalParser()

    text = """
    ASME B31.3
    Section 304.1.2 allowable stress shall not exceed 180 MPa for this service.
    Section 304.1.3 minimum wall thickness 12 mm for straight pipe.
    Section 304.1.4 corrosion allowance 3 mm in corrosive duty.
    """

    rules = parser.parse_text_to_rules(text)
    rule_types = {rule["rule_type"] for rule in rules}
    sections = {rule["conditions"]["section"] for rule in rules}

    assert "stress_limit" in rule_types
    assert "thickness_requirement" in rule_types
    assert "corrosion_allowance" in rule_types
    assert "304.1.2" in sections
    assert "304.1.3" in sections
    assert "304.1.4" in sections


def test_rule_store_retrieves_rule_from_extracted_content():
    parser = CodalParser()
    store = CodalRuleStore()
    store.add_rules(
        parser.parse_text_to_rules(
            """
            ASME B31.3
            Section 304.1.4 corrosion allowance 3 mm for sour service.
            """
        )
    )

    results = store.query({"query": "sour service corrosion allowance"})

    assert results
    assert results[0]["rule_type"] == "corrosion_allowance"


def test_fetcher_skips_ocr_if_exact_cache_exists(tmp_path):
    pdf_path = tmp_path / "standards" / "ASME_B31_3.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4 dummy")

    fetcher = CodalFetcher(storage_path=str(tmp_path / "standards"))
    cache_path = fetcher._cache_path(pdf_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("Existing exact cache content")

    text = fetcher.fetch_local_pdf(str(pdf_path))
    assert text == "Existing exact cache content"


def test_fetcher_skips_ocr_if_same_stem_cache_exists(tmp_path):
    pdf_path = tmp_path / "standards" / "ASME_B31_3.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4 dummy")

    fetcher = CodalFetcher(storage_path=str(tmp_path / "standards"))
    cache_path = fetcher.cache_dir / "ASME_B31_3_oldhash.txt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("Existing stem cache content")

    text = fetcher.fetch_local_pdf(str(pdf_path))
    assert text == "Existing stem cache content"


def test_fetcher_skips_ocr_if_alias_cache_exists(tmp_path):
    pdf_path = tmp_path / "standards" / "API_14E.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4 dummy")

    fetcher = CodalFetcher(storage_path=str(tmp_path / "standards"))
    cache_path = fetcher.cache_dir / "API_4E_oldhash.txt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("Existing alias cache content")

    text = fetcher.fetch_local_pdf(str(pdf_path))
    assert text == "Existing alias cache content"
