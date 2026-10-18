"""RBI notification pages: fetch raw snapshots, parse the index, parse a document."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
from selectolax.lexbor import LexborHTMLParser as HTMLParser

from legalrag.models import Document
from legalrag.normalize import normalize_text

BASE_URL = "https://www.rbi.org.in/Scripts/NotificationUser.aspx"
USER_AGENT = "legalrag/0.1 (offline retrieval research)"
REQUEST_TIMEOUT_SECONDS = 30.0
REQUEST_DELAY_SECONDS = 1.0

_LINK_ID = re.compile(r"NotificationUser\.aspx\?Id=(\d+)")
_DATE_CELL = "td.tableheader"
_CIRCULAR_NUMBER = re.compile(r"RBI/\d{4}-\d{2}/\d+")
MIN_BODY_CHARS = 200

# Every notification page appends the site's archive tree and inline stylesheets
# after the signed text. Left in, they become most of the corpus.
_CHROME_MARKERS = (
    re.compile(r"\b(?:19|20)\d{2}\s+All Months\b"),
    re.compile(r"/\*added by ni\*/"),
    re.compile(r"#DateSettree\w*"),
    re.compile(r"\.accordion\w*"),
)


def _content_end(text: str) -> int:
    ends = [
        match.start()
        for pattern in _CHROME_MARKERS
        if (match := pattern.search(text)) is not None
    ]
    return min(ends) if ends else len(text)


class ParseError(Exception):
    """A page does not match the shape this source expects."""


@dataclass(frozen=True, slots=True)
class NotificationRef:
    source_id: str
    url: str
    title: str
    published: str | None


def make_client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT_SECONDS,
        follow_redirects=True,
    )


def _date_from_cell(text: str) -> str | None:
    try:
        return datetime.strptime(text.strip(), "%b %d, %Y").date().isoformat()
    except ValueError:
        return None


def parse_index(html: str) -> list[NotificationRef]:
    """Every notification link on the index page, with the date heading above it.

    The index is a table in which one date cell heads the rows below it, so the
    date is carried forward from the most recent heading instead of read per row.
    """
    tree = HTMLParser(html)
    refs: list[NotificationRef] = []
    seen: set[str] = set()
    published: str | None = None

    for row in tree.css("tr"):
        heading = row.css_first(_DATE_CELL)
        if heading is not None:
            published = _date_from_cell(heading.text(strip=True))
            continue

        for anchor in row.css("a"):
            match = _LINK_ID.search(anchor.attributes.get("href") or "")
            if match is None:
                continue
            source_id = match.group(1)
            title = anchor.text(strip=True)
            if source_id in seen or not title:
                continue
            seen.add(source_id)
            refs.append(
                NotificationRef(
                    source_id=source_id,
                    url=f"{BASE_URL}?Id={source_id}&Mode=0",
                    title=title,
                    published=published,
                )
            )

    return refs


def _get(client: httpx.Client, url: str) -> str:
    time.sleep(REQUEST_DELAY_SECONDS)
    response = client.get(url)
    response.raise_for_status()
    return response.text


def _write(snapshot_dir: Path, name: str, text: str) -> Path:
    # Snapshots are re-encoded to UTF-8 so later reads never have to guess.
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_dir / name
    path.write_bytes(text.encode("utf-8"))
    return path


def fetch_index(client: httpx.Client, snapshot_dir: Path) -> Path:
    return _write(snapshot_dir, "index.html", _get(client, BASE_URL))


def fetch_document(client: httpx.Client, ref: NotificationRef, snapshot_dir: Path) -> Path:
    return _write(snapshot_dir, f"{ref.source_id}.html", _get(client, ref.url))


def _body_container(tree: HTMLParser):
    """The content column. A page carries several `container_12` blocks; the body is the longest."""
    containers = tree.css("div.container_12")
    if not containers:
        return None
    return max(containers, key=lambda node: len(node.text(strip=True)))


def _body_offset(text: str, title: str) -> int | None:
    """Where the document starts. Everything above it is site navigation chrome."""
    offsets = []
    if title:
        index = text.find(title)
        if index >= 0:
            offsets.append(index)
    match = _CIRCULAR_NUMBER.search(text)
    if match is not None:
        offsets.append(match.start())
    return min(offsets) if offsets else None


def parse_document(ref: NotificationRef, html: str) -> Document:
    """A notification page as a Document, or ParseError if it is not one.

    An unrecognised page must fail loudly rather than yield an empty document.
    """
    container = _body_container(HTMLParser(html))
    if container is None:
        raise ParseError(f"no content container in rbi:{ref.source_id}")

    text = container.text(separator="\n", strip=True)
    start = _body_offset(text, ref.title)
    if start is not None:
        text = text[start:]
    text = normalize_text(text[: _content_end(text)])

    if len(text) < MIN_BODY_CHARS:
        raise ParseError(
            f"only {len(text)} characters of body text in rbi:{ref.source_id}"
        )

    circular = _CIRCULAR_NUMBER.search(text)
    circular_text = circular.group(0) if circular is not None else ""
    return Document(
        doc_id=f"rbi:{ref.source_id}",
        source="rbi",
        source_url=ref.url,
        title=ref.title,
        published=ref.published,
        text=text,
        metadata={
            "published": ref.published or "",
            "circular": circular_text,
            # Carried onto every chunk so a retrieved result can be cited, and so a
            # golden-set label can be read back as a document rather than a bare id.
            "citation": " ".join(part for part in (circular_text, ref.title) if part),
        },
    )
