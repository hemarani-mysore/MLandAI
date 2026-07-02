"""
corpus/scraper.py — Crawl Braintree Developer Docs
====================================================
BFS-crawls pages under developer.paypal.com/braintree/docs/,
extracts main content, and saves each page as a .md file in
corpus/data/raw_docs/.

Usage:
    python corpus/scraper.py                     # crawl up to 300 pages
    python corpus/scraper.py --max-pages 100     # smaller run
    python corpus/scraper.py --delay 1.0         # slower / more polite

After this, run:
    python corpus/parser.py --docs-dir corpus/data/raw_docs
    python corpus/embed.py
"""

import argparse
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

BASE_URL   = "https://developer.paypal.com"
DOCS_PREFIX = "/braintree/docs"
RAW_DIR    = Path(__file__).parent / "data" / "raw_docs"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

SEED_URLS = [
    "/braintree/docs/start/overview",
    "/braintree/docs/guides/transactions",
    "/braintree/docs/guides/customers",
    "/braintree/docs/guides/payment-methods",
    "/braintree/docs/guides/payment-method-nonces",
    "/braintree/docs/guides/authorization/overview",
    "/braintree/docs/guides/credit-cards/overview",
    "/braintree/docs/guides/paypal/overview",
    "/braintree/docs/guides/venmo/overview",
    "/braintree/docs/guides/apple-pay/overview",
    "/braintree/docs/guides/google-pay/overview",
    "/braintree/docs/guides/ach/overview",
    "/braintree/docs/guides/recurring-billing/overview",
    "/braintree/docs/guides/webhooks/overview",
    "/braintree/docs/guides/disputes/overview",
    "/braintree/docs/guides/3d-secure/overview",
    "/braintree/docs/guides/drop-in/overview",
    "/braintree/docs/guides/hosted-fields/overview",
    "/braintree/docs/guides/reports/overview",
    "/braintree/docs/reference/overview",
]


# ─────────────────────────────────────────────────────────────────────
# URL helpers
# ─────────────────────────────────────────────────────────────────────

def _is_docs_url(path: str) -> bool:
    return path.startswith(DOCS_PREFIX) and not any(
        path.endswith(ext) for ext in (".png", ".jpg", ".gif", ".pdf", ".zip", ".svg")
    )


def _normalise(href: str) -> str | None:
    """Return a normalised /braintree/docs/… path, or None if not a docs URL."""
    parsed = urlparse(href)
    # Absolute URL on same host
    if parsed.scheme and parsed.netloc and "paypal.com" not in parsed.netloc:
        return None
    path = parsed.path.rstrip("/")
    if not path:
        return None
    if not _is_docs_url(path):
        return None
    return path


def _path_to_filename(path: str) -> str:
    """Convert /braintree/docs/guides/transactions → guides_transactions.md"""
    slug = path.replace(DOCS_PREFIX, "").strip("/").replace("/", "_")
    return (slug or "index") + ".md"


# ─────────────────────────────────────────────────────────────────────
# Content extraction
# ─────────────────────────────────────────────────────────────────────

_NOISE_TAGS = {"nav", "header", "footer", "script", "style", "noscript", "aside"}
_NOISE_CLASSES = {"sidebar", "nav", "navbar", "breadcrumb", "footer",
                  "toc", "table-of-contents", "feedback", "pagination"}


def _extract_content(soup: BeautifulSoup) -> str:
    """Pull main article text; fall back through <main> → <article> → <body>."""
    # Remove noise elements first
    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()
    for cls in _NOISE_CLASSES:
        for tag in soup.find_all(class_=re.compile(cls, re.I)):
            tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.find("body")
    if not main:
        return ""
    return str(main)


def _html_to_markdown(html: str, page_title: str) -> str:
    text = md(html, heading_style="ATX", bullets="-", strip=["a"])
    # Collapse 3+ blank lines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return f"# {page_title}\n\n{text.strip()}\n"


def _get_page_title(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    title = soup.find("title")
    if title:
        return title.get_text(strip=True).split("|")[0].strip()
    return "Untitled"


# ─────────────────────────────────────────────────────────────────────
# Crawler
# ─────────────────────────────────────────────────────────────────────

def crawl(max_pages: int, delay: float) -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    session   = requests.Session()
    session.headers.update(HEADERS)

    visited: set[str] = set()
    queue   = deque(SEED_URLS)
    saved   = 0

    print(f"Starting crawl — max {max_pages} pages, {delay}s delay\n")

    while queue and saved < max_pages:
        path = queue.popleft()
        if path in visited:
            continue
        visited.add(path)

        url = BASE_URL + path
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code != 200:
                print(f"  [{resp.status_code}] {path}")
                continue
        except Exception as e:
            print(f"  [error] {path}: {e}")
            continue

        soup  = BeautifulSoup(resp.text, "lxml")
        title = _get_page_title(soup)

        # Discover new links before we decompose anything
        for a in soup.find_all("a", href=True):
            norm = _normalise(a["href"])
            if norm and norm not in visited:
                queue.append(norm)

        content_html = _extract_content(soup)
        if not content_html.strip():
            print(f"  [skip-empty] {path}")
            continue

        markdown = _html_to_markdown(content_html, title)

        out_file = RAW_DIR / _path_to_filename(path)
        out_file.write_text(markdown, encoding="utf-8")
        saved += 1
        print(f"  [{saved:>3}] {path}  →  {out_file.name}")

        time.sleep(delay)

    print(f"\nDone — {saved} pages saved to {RAW_DIR}")
    return saved


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl Braintree developer docs.")
    parser.add_argument("--max-pages", type=int, default=300,
                        help="Maximum number of pages to crawl (default 300).")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds to wait between requests (default 0.5).")
    args = parser.parse_args()

    crawl(args.max_pages, args.delay)
