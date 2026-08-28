"""
Quick inspector: pull the full cached text for a given source id so you can
manually check a claim in the report against what the source actually says.

Usage:
    python inspect_source.py 11
    python inspect_source.py 7 9    # multiple ids at once
"""
import sys
import json

SOURCES_CACHE = "last_run_sources.json"


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_source.py <id> [<id> ...]")
        sys.exit(1)

    try:
        with open(SOURCES_CACHE) as f:
            sources = json.load(f)
    except FileNotFoundError:
        print(f"No cache found at {SOURCES_CACHE} — run test_day45.py first.")
        sys.exit(1)

    by_id = {s["id"]: s for s in sources}
    requested_ids = [int(x) for x in sys.argv[1:]]

    for sid in requested_ids:
        s = by_id.get(sid)
        if s is None:
            print(f"\n[{sid}] not found in cache.")
            continue
        print(f"\n{'=' * 70}")
        print(f"[{sid}] {s['title']}")
        print(f"URL: {s['url']}")
        print(f"Sub-question: {s['sub_question']}")
        print(f"Chars extracted: {len(s['text'])}")
        print(f"{'=' * 70}\n")
        print(s["text"])
        print()


if __name__ == "__main__":
    main()