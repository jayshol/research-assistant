"""Source tracker: wires the planner's queries into search+fetch, and
assigns each successfully-fetched source a stable, global integer id.

This id is what writer.py cites as [1], [2], ... and what critic.py
refers back to when checking claims against sources.
"""
from planner import plan
from tools.search import search
from tools.fetch import fetch_text


def gather_sources(topic: str, verbose: bool = True) -> list[dict]:
    """
    Given a topic, return a flat list of source dicts:
    [{"id": 1, "sub_question": ..., "query": ..., "title": ..., "url": ..., "text": ...}, ...]

    ids are assigned globally (not per sub-question) so they can be used
    directly as citation numbers in the final report.
    """
    if verbose:
        print(f"\n=== Planning searches for: {topic} ===\n")

    plan_items = plan(topic)
    if verbose:
        for item in plan_items:
            print(f"  - {item['sub_question']}  ->  query: \"{item['query']}\"")

    sources = []
    next_id = 1

    for item in plan_items:
        if verbose:
            print(f"\n=== Searching: {item['query']} ===")
        results = search(item["query"])

        for r in results:
            if verbose:
                print(f"  fetching: {r['url']}")
            text = fetch_text(r["url"])
            if text is None:
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

    return sources