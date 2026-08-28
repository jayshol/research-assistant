"""
Day 4-5 smoke test.

Runs: topic -> planner -> search -> fetch -> write -> critique -> revise,
and prints each stage so you can see the self-correction loop happen.

Usage:
    python test_day45.py "impact of large language models on scientific research"
"""
import sys
from sources import gather_sources
import json
from writer import write_report, revise_report, format_references
from critic import critique, format_issues_for_revision

SOURCES_CACHE = "last_run_sources.json"
def run(topic: str) -> None:
    sources = gather_sources(topic)
    with open(SOURCES_CACHE, "w") as f:
        json.dump(sources, f, indent=2)
    print(f"\n(sources cached to {SOURCES_CACHE})")
    
    if not sources:
        print("\nNo usable sources gathered — aborting.")
        return

    print(f"\n=== Writing initial draft from {len(sources)} sources ===\n")
    draft = write_report(topic, sources)
    print(draft)
    print("\n" + format_references(sources))

    print("\n=== Critiquing draft ===\n")
    issues = critique(draft, sources)

    if not issues:
        print("No issues found — draft stands as final.")
        final = draft
    else:
        print(f"{len(issues)} issue(s) found:\n")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. [{issue['issue_type']}] \"{issue['claim']}\"")
            print(f"     cited source: {issue.get('cited_source_id')} — {issue['explanation']}")

        revision_notes = format_issues_for_revision(issues)
        print("\n=== Revising draft ===\n")
        final = revise_report(topic, sources, draft, revision_notes)
        print(final)
        print("\n" + format_references(sources))

        print("\n=== Re-critiquing revised draft ===\n")
        second_pass_issues = critique(final, sources)
        if not second_pass_issues:
            print("No issues found in revision.")
        else:
            print(f"{len(second_pass_issues)} issue(s) remain after revision:")
            for i, issue in enumerate(second_pass_issues, 1):
                print(f"  {i}. [{issue['issue_type']}] \"{issue['claim']}\"")
                print(f"     cited source: {issue.get('cited_source_id')} — {issue['explanation']}")

    print("\n=== Done ===")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python test_day45.py "your topic here"')
        sys.exit(1)

    topic = " ".join(sys.argv[1:])
    run(topic)