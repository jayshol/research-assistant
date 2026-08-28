"""Fetch a URL and extract readable text from the page.

Falls back gracefully on failures (dead links, blocked scraping, non-HTML
content) since these are common when hitting arbitrary search results.
"""
import requests
from bs4 import BeautifulSoup
from config import MAX_SOURCE_CHARS

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ResearchAgent/0.1; "
        "+https://example.com/bot)"
    )
}

# Tags that are never part of the main readable content.
STRIP_TAGS = ["script", "style", "nav", "header", "footer", "aside", "form"]


def fetch_text(url: str, timeout: int = 10) -> str | None:
    """
    Fetch a URL and return cleaned page text, truncated to MAX_SOURCE_CHARS.
    Returns None if the fetch fails or the content isn't usable text.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException:
        return None

    content_type = resp.headers.get("Content-Type", "")
    if "text/html" not in content_type:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(STRIP_TAGS):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse excess blank lines left over from stripped tags.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = "\n".join(lines)

    if len(cleaned) < 200:
        # Too short to be useful — likely a paywall, JS-rendered page, or block page.
        return None

    return cleaned[:MAX_SOURCE_CHARS]
