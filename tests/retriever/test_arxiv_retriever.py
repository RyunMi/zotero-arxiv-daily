"""Tests for ArxivRetriever."""

import time
from types import SimpleNamespace

import feedparser

from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever, _run_with_hard_timeout
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def test_arxiv_retriever(config, monkeypatch):
    from pathlib import Path
    import requests
    raw = Path("tests/retriever/arxiv_rss_example.xml").read_bytes()
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        return SimpleNamespace(content=raw, raise_for_status=lambda: None)
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", lambda **kw: (_ for _ in ()).throw(AssertionError("No API calls")))
    for kind in ("html", "pdf", "tar"):
        monkeypatch.setattr(arxiv_retriever, f"extract_text_from_{kind}", lambda paper: None)
    expected = [e for e in feedparser.parse(raw).entries if e.arxiv_announce_type == "new"]
    papers = ArxivRetriever(config).retrieve_papers()
    assert [p.title for p in papers] == [e.title for e in expected]
    assert len(calls) == 1
    assert all("rss.arxiv.org" in url for url in calls)
    assert papers[0].authors == ["Alice Smith", "Bob Jones"]
    assert papers[0].abstract == "We propose a neural architecture search method for efficient transformers."
    assert papers[0].pdf_url == "https://arxiv.org/pdf/2508.14001v1"


def test_rss_cross_list_and_duplicates(config, monkeypatch):
    from pathlib import Path
    import requests
    raw = Path("tests/retriever/arxiv_rss_example.xml").read_bytes()
    parsed = feedparser.parse(raw)
    parsed.entries.append(parsed.entries[0])
    monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(content=raw, raise_for_status=lambda: None))
    monkeypatch.setattr(feedparser, "parse", lambda _: parsed)
    config.source.arxiv.include_cross_list = True
    papers = ArxivRetriever(config)._retrieve_raw_papers()
    expected = {e.id for e in parsed.entries if e.arxiv_announce_type in {"new", "cross"}}
    assert len(papers) == len(expected)
    assert papers[0].source_url() == "https://arxiv.org/src/2508.13426v1"


def test_rss_retry_timeout(config, monkeypatch):
    from pathlib import Path
    import requests
    raw = Path("tests/retriever/arxiv_rss_example.xml").read_bytes()
    waits, calls = [], []
    def get(*a, **k):
        calls.append(k)
        if len(calls) < 3:
            raise requests.Timeout("temporary")
        return SimpleNamespace(content=raw, raise_for_status=lambda: None)
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(arxiv_retriever, "sleep", waits.append)
    assert ArxivRetriever(config)._retrieve_raw_papers()
    assert waits == [10, 20]
    assert all(c["timeout"] == arxiv_retriever.DOWNLOAD_TIMEOUT for c in calls)


def test_rss_rejects_http_error_and_invalid_feed(config, monkeypatch):
    import pytest
    import requests
    for status in (403, 406):
        response = requests.Response()
        response.status_code = status
        monkeypatch.setattr(requests, "get", lambda *a, **k: response)
        with pytest.raises(requests.HTTPError):
            ArxivRetriever(config)._retrieve_raw_papers()
    monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(content=b"<html>Unavailable</html>", raise_for_status=lambda: None))
    with pytest.raises(ValueError, match="Invalid arXiv RSS"):
        ArxivRetriever(config)._retrieve_raw_papers()


def test_rss_missing_abstract_is_not_silently_dropped():
    import pytest
    entry = feedparser.FeedParserDict(id="oai:arXiv.org:2609.12345v1", title="Paper", author="Author", summary="")
    with pytest.raises(ValueError, match="Missing title, abstract or authors"):
        arxiv_retriever._result_from_rss(entry)


def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 0.01), timeout=1, operation="test op", paper_title="paper"
    )
    assert result == "done"


def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 1.0), timeout=0.01, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "timed out" in warnings[0]


def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _raise_runtime_error, (), timeout=1, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "boom" in warnings[0]
