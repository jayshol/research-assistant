"""Web search tool, backed by Tavily (built for LLM/agent use cases)."""
from tavily import TavilyClient
from config import TAVILY_API_KEY, RESULTS_PER_QUERY

_client = TavilyClient(api_key=TAVILY_API_KEY)


def search(query: str, max_results: int = RESULTS_PER_QUERY) -> list[dict]:
    """
    Run a web search and return a list of result dicts:
    [{"title": ..., "url": ..., "snippet": ...}, ...]
    """
    response = _client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",  # better relevance, slightly slower
    )
    results = []
    for r in response.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),  # Tavily's short extracted summary
        })
    return results
