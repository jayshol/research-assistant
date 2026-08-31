"""Source tracker: wires the planner's queries into search+fetch, and
assigns each successfully-fetched source a stable, global integer id.

This id is what writer.py cites as [1], [2], ... and what critic.py
refers back to when checking claims against sources.
"""
from planner import plan
from tools.search import search
from tools.fetch import fetch_text
from cache import cache_lookup, cache_store

# Per-query search trace from the most recent gather_sources() call, for
# debug/inspection purposes (see get_last_debug_trace). Not part of the
# public sources contract -- callers that only need the source list can
# ignore this entirely.
_last_debug_trace: list[dict] = []


def get_last_debug_trace() -> list[dict]:
    """
    Returns the per-query search trace from the most recent gather_sources()
    call, including results that were skipped during fetch (not just the
    ones that made it into the final source list).

    Shape: [{"sub_question": ..., "query": ..., "results": [
        {"title", "url", "snippet", "fetched": bool, "chars": int|None}
    ]}, ...]

    Populated on both a fresh run AND a cache hit (the trace is cached
    alongside the sources) -- empty only for cache entries written before
    this feature existed.
    """
    return _last_debug_trace


def gather_sources(topic: str, verbose: bool = True) -> list[dict]:
    """
    Given a topic, return a flat list of source dicts:
    [{"id": 1, "sub_question": ..., "query": ..., "title": ..., "url": ..., "text": ...}, ...]

    ids are assigned globally (not per sub-question) so they can be used
    directly as citation numbers in the final report.

    Cached on disk keyed by topic alone (not `verbose`, so both verbose
    modes share the same cache entry). This is what actually saves Tavily
    credits/search calls -- planner.plan() and the Gemini steps downstream
    are already cached individually, but without this, re-running the same
    topic would still re-search and re-fetch every source from scratch even
    when nothing downstream needed fresh data.
    """
    global _last_debug_trace

    hit, key, cached = cache_lookup("gather_sources", topic)
    if hit:
        if isinstance(cached, dict) and "sources" in cached:
            sources = cached["sources"]
            _last_debug_trace = cached.get("debug_trace", [])
        else:
            # Legacy cache entries written before debug_trace existed --
            # `cached` is directly the list of sources.
            sources = cached
            _last_debug_trace = []
        if verbose:
            print(f"\n[cache hit] gather_sources('{topic}') — skipping search/fetch")
        return sources

    if verbose:
        print(f"\n=== Planning searches for: {topic} ===\n")

    plan_items = plan(topic)
    if verbose:
        for item in plan_items:
            print(f"  - {item['sub_question']}  ->  query: \"{item['query']}\"")

    sources = []
    debug_trace = []
    next_id = 1

    for item in plan_items:
        if verbose:
            print(f"\n=== Searching: {item['query']} ===")
        results = search(item["query"])
        trace_results = []

        for r in results:
            if verbose:
                print(f"  fetching: {r['url']}")
            text = fetch_text(r["url"])
            fetched = text is not None

            trace_results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("snippet", ""),
                "fetched": fetched,
                "chars": len(text) if fetched else None,
            })

            if not fetched:
                if verbose:
                    print("    (skipped — fetch failed or content too short)")
                continue

            sources.append({
                "id": next_id,
                "sub_question": item["sub_question"],
                "query": item["query"],
                "title": r["title"],
                "url": r["url"],
                "text": text,
            })
            if verbose:
                print(f"    ok — {len(text)} chars extracted (source [{next_id}])")
            next_id += 1

        debug_trace.append({
            "sub_question": item["sub_question"],
            "query": item["query"],
            "results": trace_results,
        })

    _last_debug_trace = debug_trace

    # Cached even if empty -- a "no usable sources for this topic" result is
    # still a legitimate, reproducible outcome worth not re-paying for.
    cache_store("gather_sources", key, {"sources": sources, "debug_trace": debug_trace})
    return sources
