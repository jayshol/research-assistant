"""
Source credibility classification.

Classifies each source into one of four types -- academic/journal, news,
blog/company, social media -- primarily via fast, free domain heuristics.
An optional LLM-based fallback exists for ambiguous domains but is OFF by
default: it costs API quota, and given the free-tier cap, domain heuristics
alone are the sane default. Turn it on explicitly (use_llm_fallback=True)
once you have quota headroom.

This is deliberately a *heuristic* classifier, not a rigorous one -- domain
matching is a reasonable proxy for source type, not a guarantee (e.g. a
credentialed researcher's personal blog would still be classified "blog").
Good enough to differentiate "cites a peer-reviewed journal" from "cites a
company blog post" for the reader, which is the actual goal.
"""
import re
from urllib.parse import urlparse

from cache import cache_lookup, cache_store

# --- Domain heuristics -------------------------------------------------

ACADEMIC_DOMAIN_PATTERNS = [
    r"\.edu$", r"\.ac\.[a-z]{2}$", r"\.gov$",
    r"arxiv\.org$", r"ncbi\.nlm\.nih\.gov$", r"pubmed\.ncbi\.nlm\.nih\.gov$",
    r"nature\.com$", r"sciencedirect\.com$", r"(link\.)?springer\.com$",
    r"(onlinelibrary\.)?wiley\.com$", r"ieee\.org$", r"ieeexplore\.ieee\.org$",
    r"(dl\.)?acm\.org$", r"jstor\.org$", r"doi\.org$",
    r"biorxiv\.org$", r"medrxiv\.org$", r"ssrn\.com$",
    r"plos\.org$", r"cell\.com$", r"nejm\.org$", r"thelancet\.com$",
    r"jamanetwork\.com$", r"frontiersin\.org$", r"tandfonline\.com$",
    r"mdpi\.com$", r"pnas\.org$", r"science(mag)?\.org$", r"science\.org$",
    r"researchgate\.net$", r"semanticscholar\.org$",
]

NEWS_DOMAIN_PATTERNS = [
    r"nytimes\.com$", r"washingtonpost\.com$", r"bbc\.(co\.uk|com)$",
    r"reuters\.com$", r"apnews\.com$", r"cnn\.com$", r"npr\.org$",
    r"wsj\.com$", r"bloomberg\.com$", r"theguardian\.com$", r"economist\.com$",
    r"ft\.com$", r"axios\.com$", r"politico\.com$", r"forbes\.com$",
    r"time\.com$", r"usatoday\.com$", r"latimes\.com$", r"newsweek\.com$",
    r"wired\.com$", r"techcrunch\.com$", r"theverge\.com$", r"arstechnica\.com$",
    r"businessinsider\.com$", r"cnbc\.com$", r"scientificamerican\.com$",
    r"vox\.com$", r"aljazeera\.com$", r"thequantuminsider\.com$",
]

SOCIAL_DOMAIN_PATTERNS = [
    r"reddit\.com$", r"(twitter|x)\.com$", r"facebook\.com$", r"linkedin\.com$",
    r"youtube\.com$", r"tiktok\.com$", r"instagram\.com$", r"quora\.com$",
    r"medium\.com$", r"substack\.com$", r"threads\.net$", r"news\.ycombinator\.com$",
]

SOURCE_TYPE_LABELS = {
    "academic": "academic/journal",
    "news": "news",
    "blog": "blog/company",
    "social": "social media",
}


def _domain(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def classify_source_domain(url: str) -> str:
    """
    Pure heuristic, no network/API calls. Returns one of:
    "academic", "news", "blog", "social".

    Anything that doesn't match a known academic/news/social pattern
    defaults to "blog" -- the catch-all for company sites, independent
    blogs, and other unaffiliated web content, which is the majority of
    what general web search turns up.
    """
    domain = _domain(url)
    if not domain:
        return "blog"

    for pattern in ACADEMIC_DOMAIN_PATTERNS:
        if re.search(pattern, domain):
            return "academic"
    for pattern in NEWS_DOMAIN_PATTERNS:
        if re.search(pattern, domain):
            return "news"
    for pattern in SOCIAL_DOMAIN_PATTERNS:
        if re.search(pattern, domain):
            return "social"
    return "blog"


def classify_sources(sources: list[dict], use_llm_fallback: bool = False) -> list[dict]:
    """
    Return a NEW list of source dicts (originals untouched), each with
    'source_type' (short key: academic/news/blog/social) and
    'source_type_label' (display string, e.g. "academic/journal") added.

    use_llm_fallback: if True, sources whose domain heuristic is uncertain
    could instead be classified with an LLM call. Currently unused by the
    domain heuristic (which always returns a confident guess), reserved for
    a future version that flags low-confidence domains for an LLM check.
    Left OFF by default to avoid spending API quota -- domain heuristics
    are free and fast.
    """
    classified = []
    for s in sources:
        source_type = classify_source_domain(s["url"])
        classified.append({
            **s,
            "source_type": source_type,
            "source_type_label": SOURCE_TYPE_LABELS[source_type],
        })
    return classified


# --- Optional LLM fallback (off by default, see use_llm_fallback above) ---
# Kept minimal and cached, for later use on domains the heuristic can't
# confidently place (e.g. a bare unfamiliar domain).

_LLM_CLASSIFIER_PROMPT = """Classify the following source into exactly one \
category based on its domain, title, and a short excerpt: "academic", \
"news", "blog", or "social".

- academic: peer-reviewed journals, university/government research pages, preprint servers
- news: professional news outlets and journalism
- blog: company blogs, independent/personal blogs, marketing content
- social: social media posts, forums, video platforms

Respond with ONLY the single lowercase category word, nothing else.

Source title: {title}
Source URL: {url}
Excerpt: {excerpt}
"""


def classify_source_llm(source: dict) -> str:
    """
    LLM-based classification for a single source, cached on disk. Costs one
    Gemini call per uncached source -- use sparingly, and only once domain
    heuristics prove insufficient for your topic set.
    """
    hit, key, cached = cache_lookup(
        "classify_source_llm", source["title"], source["url"], source["text"][:300]
    )
    if hit:
        return cached

    from google import genai
    from google.genai import types
    from config import GEMINI_API_KEY, LLM_MODEL

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = _LLM_CLASSIFIER_PROMPT.format(
        title=source["title"], url=source["url"], excerpt=source["text"][:300]
    )
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=10),
    )
    result = response.text.strip().lower()
    if result not in SOURCE_TYPE_LABELS:
        result = "blog"  # safe default on an unexpected response

    cache_store("classify_source_llm", key, result)
    return result
