from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
import feedparser
import multiprocessing
import os
from queue import Empty
from time import sleep
from datetime import datetime, timezone
import re
from typing import Any, Callable, TypeVar
from loguru import logger
import requests

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


def _result_from_rss(entry: feedparser.FeedParserDict) -> ArxivResult:
    """Convert an official arXiv Atom announcement to the existing result type.

    RSS timestamps describe the announcement, not the original submission.
    Downstream ranking uses title/abstract; full-text URLs retain the version.
    """
    paper_id = entry.get("id", "").removeprefix("oai:arXiv.org:")
    if not re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-zA-Z.-]+/\d{7})(?:v\d+)?", paper_id):
        raise ValueError("Invalid arXiv ID in RSS entry")
    title = re.sub(r"\s+", " ", entry.get("title", "")).strip()
    abstract = re.sub(
        r"^arXiv:.*?\bAbstract:\s*", "", entry.get("summary", ""),
        count=1, flags=re.DOTALL,
    ).strip()
    # dc:creator is a comma-separated author list in arXiv's Atom feed.
    author_text = entry.get("author", "")
    authors = [ArxivResult.Author(name.strip()) for name in author_text.split(",") if name.strip()]
    if not title or not abstract or not authors:
        raise ValueError(f"Missing title, abstract or authors in arXiv RSS entry {paper_id}")
    categories = [tag["term"] for tag in entry.get("tags", []) if tag.get("term")]
    dates = {}
    for key in ("updated", "published"):
        if entry.get(f"{key}_parsed"):
            dates[key] = datetime(*entry[f"{key}_parsed"][:6], tzinfo=timezone.utc)
    return ArxivResult(
        entry_id=f"https://arxiv.org/abs/{paper_id}",
        title=title, summary=abstract, authors=authors,
        categories=categories, primary_category=categories[0] if categories else "",
        links=[ArxivResult.Link(
            href=f"https://arxiv.org/pdf/{paper_id}", title="pdf",
            rel="related", content_type="application/pdf",
        )],
        **dates,
    )


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        # The Atom feed already contains the metadata used by this application.
        # Re-querying every ID through export.arxiv.org introduced a second
        # dependency: one HTTP 406 discarded the entire day's successful work.
        query = '+'.join(self.config.source.arxiv.category)
        url = f"https://rss.arxiv.org/atom/{query}"
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
                response.raise_for_status()
                break
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if attempt == 2 or (status is not None and status != 429 and status < 500):
                    raise
                wait = 10 * (2 ** attempt)
                logger.warning(f"arXiv RSS request failed (status={status}); retrying in {wait}s")
                sleep(wait)

        feed = feedparser.parse(response.content)
        title = feed.feed.get("title", "")
        if feed.get("bozo") or not title or "Feed error for query" in title:
            raise ValueError(f"Invalid arXiv RSS response for {query}; refusing an incomplete digest")

        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        allowed = {"new", "cross"} if include_cross_list else {"new"}
        raw_papers = []
        seen = set()
        for entry in feed.entries:
            if entry.get("arxiv_announce_type", "new") not in allowed:
                continue
            paper = _result_from_rss(entry)
            if paper.entry_id in seen:
                continue
            seen.add(paper.entry_id)
            raw_papers.append(paper)
            if self.config.executor.debug and len(raw_papers) == 10:
                break
        logger.info(f"Retrieved {len(raw_papers)} arXiv papers directly from RSS (no metadata API calls)")
        return raw_papers

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        full_text = extract_text_from_tar(raw_paper)
        if full_text is None:
            full_text = extract_text_from_html(raw_paper)
        if full_text is None:
            full_text = extract_text_from_pdf(raw_paper)
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )


def extract_text_from_html(paper: ArxivResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
