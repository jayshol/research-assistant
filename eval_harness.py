"""
Evaluation harness for the research assistant.

Runs the full self-correcting pipeline (plan -> search -> write -> critique
-> revise, bounded and convergence-tracked via revision_loop.py) across a
fixed set of topics, logs the critic's findings per topic, and writes both a
machine-readable results.json and a summary printed to stdout.

This is what turns "the critic catches hallucinations" into a measured claim:
"across N topics, the critic caught an average of X issues per initial draft,
Y% of topics converged to zero issues within 2 revision passes."

Usage:
    python eval_harness.py
    python eval_harness.py --limit 3                 # quick smoke test
    python eval_harness.py --max-passes 2             # cheaper on quota
    python eval_harness.py --topics-file topics.txt   # custom topic list
    python eval_harness.py --out results/run1.json

Each topic is independent and wrapped in error handling, so one bad topic
(no sources found, API failure, etc.) doesn't kill the whole run -- it's
logged with an "error" field and excluded from the numeric summary.
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

from sources import gather_sources
from revision_loop import run_self_correcting_pipeline
from source_classifier import classify_sources
import usage_tracker
from analytics import assess_report, record_report_metrics

DEFAULT_TOPICS = [
    # Mix of well-documented/factual, fast-moving, and contested/broad topics
    # -- variety matters here since a harness that only tests "easy" topics
    # would overstate the critic's real-world catch rate.
    "impact of large language models on scientific research",
    "CRISPR gene editing ethical concerns",
    "causes of the 2008 financial crisis",
    "quantum computing current state and challenges",
    "effectiveness of remote work on productivity",
    "history and current status of nuclear fusion energy",
    "social media's effect on teenage mental health",
    "electric vehicle battery technology advances",
    "universal basic income pilot program results",
    "AI regulation approaches in the US and EU",
    "microplastics health effects research",
    "self-driving car safety statistics",
    "renewable energy grid storage solutions",
    "gut microbiome and disease research",
    "cybersecurity risks of quantum computing",
]


def _load_topics(topics_file: str | None) -> list[str]:
    if not topics_file:
        return DEFAULT_TOPICS
    path = Path(topics_file)
    topics = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not topics:
        raise ValueError(f"No topics found in {topics_file}")
    return topics


def _issue_type_counts(issues_by_pass: list[list[dict]]) -> Counter:
    """Flatten every issue across every pass into a Counter of issue_type."""
    counts = Counter()
    for issues in issues_by_pass:
        for issue in issues:
            counts[issue.get("issue_type", "unknown")] += 1
    return counts


def run_eval(topics: list[str], max_passes: int, out_path: Path,
             existing_results: dict | None = None) -> dict:
    """
    Runs the pipeline across `topics`, writing results.json to disk after
    EVERY topic (not just at the end) so a Ctrl+C or crash mid-run loses at
    most the one topic currently in flight -- not the whole run's progress.

    If `existing_results` is provided (loaded from a prior results.json),
    topics that already succeeded there (error is None) are skipped and
    their cached entry is reused, so resuming an interrupted run doesn't
    redo work that's already done.
    """
    already_done = {}
    if existing_results:
        for t in existing_results.get("topics", []):
            if t.get("error") is None:
                already_done[t["topic"]] = t

    topic_results = []

    for i, topic in enumerate(topics, 1):
        print(f"\n[{i}/{len(topics)}] {topic}")

        if topic in already_done:
            print("  [resume] already have a successful result — skipping")
            topic_results.append(already_done[topic])
            _write_results(out_path, topic_results, max_passes)
            continue

        start = time.time()

        try:
            sources = gather_sources(topic, verbose=False)
        except Exception as e:
            print(f"  source gathering failed: {e}")
            topic_results.append({"topic": topic, "error": f"source gathering failed: {e}"})
            _write_results(out_path, topic_results, max_passes)
            continue

        if not sources:
            print("  no usable sources — skipping")
            topic_results.append({"topic": topic, "error": "no usable sources"})
            _write_results(out_path, topic_results, max_passes)
            continue

        sources = classify_sources(sources)
        source_type_counts = Counter(s["source_type"] for s in sources)

        usage_mark = usage_tracker.mark()
        try:
            result = run_self_correcting_pipeline(topic, sources, max_passes=max_passes)
        except Exception as e:
            print(f"  pipeline failed: {e}")
            topic_results.append({"topic": topic, "error": f"pipeline failed: {e}"})
            _write_results(out_path, topic_results, max_passes)
            continue

        elapsed = round(time.time() - start, 1)
        type_counts = _issue_type_counts(result.issues_by_pass)

        print(f"  {len(sources)} sources | issues: "
              f"{' -> '.join(str(n) for n in result.issue_counts)} | "
              f"converged: {result.converged} | {elapsed}s")

        metrics = assess_report(topic, sources, result, usage_mark=usage_mark)
        record_report_metrics(metrics)  # accumulates in metrics_log.jsonl across ALL runs, not just this batch

        topic_results.append({
            "topic": topic,
            "num_sources": len(sources),
            "source_type_counts": dict(source_type_counts),
            "issue_counts": result.issue_counts,
            "issues_by_pass": result.issues_by_pass,
            "issue_type_totals": dict(type_counts),
            "num_revisions": result.num_revisions,
            "converged": result.converged,
            "elapsed_seconds": elapsed,
            "num_gemini_calls": metrics["num_gemini_calls"],
            "total_tokens": metrics["total_tokens"],
            "error": None,
        })
        _write_results(out_path, topic_results, max_passes)

    return {
        "max_passes": max_passes,
        "topics": topic_results,
        "summary": _summarize(topic_results),
    }


def _write_results(out_path: Path, topic_results: list[dict], max_passes: int) -> None:
    """Write current progress to disk immediately. Called after every single
    topic (success or failure) so an interrupted run never loses more than
    the one topic in flight when it was stopped."""
    output = {
        "max_passes": max_passes,
        "topics": topic_results,
        "summary": _summarize(topic_results),
    }
    out_path.write_text(json.dumps(output, indent=2))


def _summarize(topic_results: list[dict]) -> dict:
    ok = [t for t in topic_results if t.get("error") is None]
    failed = [t for t in topic_results if t.get("error") is not None]

    if not ok:
        return {
            "num_topics": len(topic_results),
            "num_successful": 0,
            "num_failed": len(failed),
            "note": "no successful runs to summarize",
        }

    first_pass_issues = [t["issue_counts"][0] for t in ok]
    total_issues = [sum(t["issue_counts"]) for t in ok]
    converged = [t for t in ok if t["converged"]]
    converged_within_2 = [t for t in ok if t["converged"] and len(t["issue_counts"]) <= 2]

    combined_type_counts = Counter()
    for t in ok:
        combined_type_counts.update(t["issue_type_totals"])

    combined_source_type_counts = Counter()
    for t in ok:
        combined_source_type_counts.update(t["source_type_counts"])

    return {
        "num_topics": len(topic_results),
        "num_successful": len(ok),
        "num_failed": len(failed),
        "avg_first_pass_issues": round(sum(first_pass_issues) / len(ok), 2),
        "avg_total_issues_across_passes": round(sum(total_issues) / len(ok), 2),
        "avg_revisions": round(sum(t["num_revisions"] for t in ok) / len(ok), 2),
        "pct_converged": round(100 * len(converged) / len(ok), 1),
        "pct_converged_within_2_passes": round(100 * len(converged_within_2) / len(ok), 1),
        "issue_type_distribution": dict(combined_type_counts),
        "source_type_distribution": dict(combined_source_type_counts),
        "avg_elapsed_seconds": round(sum(t["elapsed_seconds"] for t in ok) / len(ok), 1),
    }


def _print_summary(summary: dict) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if summary.get("num_successful", 0) == 0:
        print("No successful runs.")
        return
    print(f"Topics run:                     {summary['num_topics']} "
          f"({summary['num_successful']} ok, {summary['num_failed']} failed)")
    print(f"Avg issues in first draft:      {summary['avg_first_pass_issues']}")
    print(f"Avg total issues across passes: {summary['avg_total_issues_across_passes']}")
    print(f"Avg revisions per topic:        {summary['avg_revisions']}")
    print(f"Converged to 0 issues:          {summary['pct_converged']}%")
    print(f"Converged within 2 passes:      {summary['pct_converged_within_2_passes']}%")
    print(f"Avg time per topic:             {summary['avg_elapsed_seconds']}s")
    print("Issue type distribution:")
    for issue_type, count in sorted(summary["issue_type_distribution"].items(),
                                     key=lambda kv: -kv[1]):
        print(f"  {issue_type:20s} {count}")
    print("Source type distribution:")
    for source_type, count in sorted(summary["source_type_distribution"].items(),
                                      key=lambda kv: -kv[1]):
        print(f"  {source_type:20s} {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the eval harness across a fixed topic set.")
    parser.add_argument("--topics-file", type=str, default=None,
                         help="Path to a text file with one topic per line. "
                              "Defaults to the built-in 15-topic list.")
    parser.add_argument("--max-passes", type=int, default=3,
                         help="Max critique passes per topic (default 3).")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only run the first N topics -- useful for a cheap smoke test "
                              "before spending quota on the full set.")
    parser.add_argument("--out", type=str, default="results.json",
                         help="Output path for the JSON results (default results.json).")
    parser.add_argument("--no-resume", action="store_true",
                         help="Ignore any existing results.json and re-run every topic from "
                              "scratch, even ones that already succeeded.")
    args = parser.parse_args()

    topics = _load_topics(args.topics_file)
    if args.limit:
        topics = topics[:args.limit]

    out_path = Path(args.out)
    existing_results = None
    if out_path.exists() and not args.no_resume:
        try:
            existing_results = json.loads(out_path.read_text())
            num_done = sum(1 for t in existing_results.get("topics", [])
                            if t.get("error") is None)
            if num_done:
                print(f"Found existing {out_path} with {num_done} already-successful "
                      f"topic(s) -- resuming (use --no-resume to ignore this).")
        except (json.JSONDecodeError, OSError):
            existing_results = None  # corrupt/unreadable -- just start fresh

    print(f"Running eval harness on {len(topics)} topic(s), max_passes={args.max_passes}")
    output = run_eval(topics, max_passes=args.max_passes, out_path=out_path,
                       existing_results=existing_results)

    print(f"\nResults written to {out_path}")
    _print_summary(output["summary"])
