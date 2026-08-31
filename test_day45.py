"""
Day 4-5 smoke test.

Runs: topic -> planner -> search -> fetch -> write -> [critique -> revise]*,
using the bounded self-correction loop, and prints the convergence trace
(e.g. "Issues found: 3 -> 1 -> 0 after 2 revisions.") along with the final
report.

Usage:
    python test_day45.py "impact of large language models on scientific research"
    python test_day45.py "..." --max-passes 5
"""
import sys
import json

from sources import gather_sources
from revision_loop import run_self_correcting_pipeline
from source_classifier import classify_sources
import usage_tracker
from analytics import assess_report, record_report_metrics

SOURCES_CACHE = "last_run_sources.json"


def run(topic: str, max_passes: int = 3) -> None:
    sources = gather_sources(topic)
    sources = classify_sources(sources)
    with open(SOURCES_CACHE, "w") as f:
        json.dump(sources, f, indent=2)
    print(f"\n(sources cached to {SOURCES_CACHE})")

    if not sources:
        print("\nNo usable sources gathered — aborting.")
        return

    print(f"\n=== Running self-correcting pipeline on {len(sources)} sources "
          f"(max {max_passes} critique passes) ===\n")

    usage_mark = usage_tracker.mark()
    result = run_self_correcting_pipeline(topic, sources, max_passes=max_passes)

    for i, (count, issues) in enumerate(zip(result.issue_counts, result.issues_by_pass), 1):
        print(f"--- Pass {i}: {count} issue(s) ---")
        for j, issue in enumerate(issues, 1):
            print(f"  {j}. [{issue['issue_type']}] \"{issue['claim']}\"")
            print(f"     cited source: {issue.get('cited_source_id')} — {issue['explanation']}")
        print()

    print("=== Final report ===\n")
    print(result.final_report)
    print("\n" + result.references)

    print("\n=== Convergence ===")
    print(result.summary_line())

    metrics = assess_report(topic, sources, result, usage_mark=usage_mark)
    record_report_metrics(metrics)
    print("\n=== Model performance metrics (this run) ===")
    for k, v in metrics.items():
        print(f"  {k:20s} {v}")
    print("  (appended to metrics_log.jsonl — run `python analytics.py` for a cross-run summary)")

    print("=== Done ===")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print('Usage: python test_day45.py "your topic here" [--max-passes N]')
        sys.exit(1)

    max_passes = 3
    if "--max-passes" in args:
        idx = args.index("--max-passes")
        max_passes = int(args[idx + 1])
        del args[idx:idx + 2]

    topic = " ".join(args)
    run(topic, max_passes=max_passes)
