"""RBI notification pages: fetch raw snapshots, parse the index, parse a document."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
from selectolax.lexbor import LexborHTMLParser as HTMLParser

BASE_URL = "https://www.rbi.org.in/Scripts/NotificationUser.aspx"
USER_AGENT = "legalrag/0.1 (offline retrieval research)"
REQUEST_TIMEOUT_SECONDS = 30.0
REQUEST_DELAY_SECONDS = 1.0

_LINK_ID = re.compile(r"NotificationUser\.aspx\?Id=(\d+)")
_DATE_CELL = "td.tableheader"


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
