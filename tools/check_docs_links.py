#!/usr/bin/env python3
"""Check internal links and anchors in the exported docs site.

Usage: python tools/check_docs_links.py docs/out --base /peyesim

Every ``href`` (and meta-refresh target) that points into the site must
resolve to an exported page, and a ``#fragment`` must match an ``id`` on
that page. Exit 0 when all resolve, 1 otherwise.
"""

from __future__ import annotations

import argparse
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit


class _Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag == "a" and attrs.get("href"):
            self.links.append(attrs["href"])
        if tag == "meta" and (attrs.get("http-equiv") or "").lower() == "refresh":
            m = re.search(r"url=(.+)$", attrs.get("content") or "", re.I)
            if m:
                self.links.append(m.group(1).strip())


def _page_file(out: Path, path: str) -> Path | None:
    rel = path.lstrip("/")
    for candidate in (out / rel, out / rel / "index.html", out / f"{rel.rstrip('/')}.html"):
        if candidate.is_file():
            return candidate
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--base", default="", help="base path the site is served under")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    base = args.base.rstrip("/")

    pages: dict[Path, _Page] = {}
    for html in out.rglob("*.html"):
        page = _Page()
        page.feed(html.read_text(encoding="utf-8", errors="replace"))
        pages[html] = page

    problems = []
    checked = 0
    for html, page in sorted(pages.items()):
        rel = html.relative_to(out).as_posix()
        here = "/" + (rel[: -len("index.html")] if rel.endswith("index.html") else rel)
        for href in page.links:
            parts = urlsplit(href)
            if parts.scheme or parts.netloc or href.startswith(("mailto:", "javascript:")):
                continue
            url = urlsplit(urljoin(f"{base}{here}", href))
            path = unquote(url.path)
            if base and not path.startswith(base + "/") and path != base:
                problems.append(f"{rel}: {href} is outside the base path {base}")
                continue
            target = _page_file(out, path[len(base):]) if path else html
            checked += 1
            if target is None:
                problems.append(f"{rel}: {href} -> no page")
            elif url.fragment and target.suffix == ".html":
                ids = pages[target].ids if target in pages else set()
                if unquote(url.fragment) not in ids:
                    problems.append(f"{rel}: {href} -> no #{url.fragment} on {target.relative_to(out)}")

    if problems:
        print("\n".join(sorted(set(problems))))
        print(f"check_docs_links: {len(set(problems))} broken of {checked} internal links")
        return 1
    print(f"check_docs_links: {checked} internal links OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
