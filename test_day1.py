"""
Day 1-2 smoke test.

Runs: topic -> planner -> search -> fetch, and prints what was gathered.
No writing or critique yet — this just confirms the plumbing works.

Usage:
    python test_day1.py "impact of large language models on scientific research"
"""
import sys
from planner import plan
from tools.search import search
from tools.fetch import fetch_text


def gather_sources(topic: str) -> list[dict]:
    print(f"\n=== Planning searches for: {topic} ===\n")
    plan_items = plan(topic)
    for item in plan_items:
        print(f"  - {item['sub_question']}  ->  query: \"{item['query']}\"")

    sources = []
    for item in plan_items:
        print(f"\n=== Searching: {item['query']} ===")
        results = search(item["query"])

        for r in results:
            print(f"  fetching: {r['url']}")
            text = fetch_text(r["url"])
            if text is None:
                print("    (skipped — fetch failed or content too short)")
                continue
            sources.append({
                "sub_question": item["sub_question"],
                "query": item["query"],
                "title": r["title"],
                "url": r["url"],
                "text": text,
            })
            print(f"    ok — {len(text)} chars extracted")

    return sources


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python test_day1.py "your topic here"')
        sys.exit(1)

    topic = " ".join(sys.argv[1:])
    gathered = gather_sources(topic)

    print(f"\n=== Done: {len(gathered)} usable sources gathered ===")
    for s in gathered:
        print(f"  [{s['sub_question']}] {s['title']} — {s['url']}")
