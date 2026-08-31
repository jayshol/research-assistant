"""
Post-report analytics: assess and log the performance of the model(s) used
for a single report generation, and later summarize across everything
logged so far.

This is intentionally separate from eval_harness.py: the harness runs one
fixed batch of topics as a point-in-time measurement. This module is meant
to be called after ANY report finishes -- a single ad-hoc test_day45.py
run, a live app.py session, or one topic inside eval_harness.py -- and
appended to a running log (metrics_log.jsonl). That log accumulates over
time, so if you later swap the writer or critic to a different model, you
can compare "before" and "after" using real accumulated usage data instead
of a single fixed batch.
"""
import json
import time
from pathlib import Path
from collections import Counter, defaultdict

from config import LLM_MODEL
import usage_tracker

DEFAULT_LOG_PATH = Path("metrics_log.jsonl")


def assess_report(
    topic: str,
    sources: list[dict],
    result,  # a revision_loop.RevisionResult
    usage_mark: int,
    writer_model: str = LLM_MODEL,
    critic_model: str = LLM_MODEL,
) -> dict:
    """
    Build one metrics record for a completed report run.

    `usage_mark` must be the value returned by usage_tracker.mark() taken
    BEFORE this report's pipeline call started -- that's what isolates this
    report's token usage from anything else that ran before/after it in the
    same process (e.g. other topics in an eval_harness batch).
    """
    usage_entries = usage_tracker.since(usage_mark)
    prompt_tokens = sum(e["prompt_tokens"] for e in usage_entries)
    output_tokens = sum(e["output_tokens"] for e in usage_entries)
    total_tokens = sum(e["total_tokens"] for e in usage_entries)
    # Only entries that actually called Gemini appear here -- cache hits
    # never call track_usage, so this is real API spend, not step count.
    num_gemini_calls = len(usage_entries)

    source_type_counts = Counter(s.get("source_type", "unknown") for s in sources)
    issue_type_totals = Counter()
    for issues in result.issues_by_pass:
        for issue in issues:
            issue_type_totals[issue.get("issue_type", "unknown")] += 1

    return {
        "topic": topic,
        "timestamp": time.time(),
        "writer_model": writer_model,
        "critic_model": critic_model,
        "num_sources": len(sources),
        "source_type_counts": dict(source_type_counts),
        "issue_counts": result.issue_counts,
        "issue_type_totals": dict(issue_type_totals),
        "num_revisions": result.num_revisions,
        "converged": result.converged,
        "num_gemini_calls": num_gemini_calls,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def record_report_metrics(record: dict, log_path: Path = DEFAULT_LOG_PATH) -> None:
    """Append one metrics record as a line to a persistent JSONL log.
    Append-only and safe to call from concurrent short-lived processes
    (each call opens, writes one line, closes)."""
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_metrics_log(log_path: Path = DEFAULT_LOG_PATH) -> list[dict]:
    """Read every record logged so far, oldest first."""
    log_path = Path(log_path)
    if not log_path.exists():
        return []
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarize_by_model(records: list[dict]) -> dict:
    """
    Group logged reports by (writer_model, critic_model) pair and compute
    aggregate stats for each. This is the actual "should I keep or switch
    this model" comparison: once you swap a model in and generate a few
    more reports, a second group appears here to compare against the first.
    """
    groups = defaultdict(list)
    for r in records:
        key = f"writer={r.get('writer_model', 'unknown')} | critic={r.get('critic_model', 'unknown')}"
        groups[key].append(r)

    summary = {}
    for key, group in groups.items():
        n = len(group)
        with_issues = [g for g in group if g.get("issue_counts")]
        summary[key] = {
            "num_reports": n,
            "avg_first_pass_issues": (
                round(sum(g["issue_counts"][0] for g in with_issues) / len(with_issues), 2)
                if with_issues else None
            ),
            "pct_converged": round(100 * sum(1 for g in group if g["converged"]) / n, 1),
            "avg_revisions": round(sum(g["num_revisions"] for g in group) / n, 2),
            "avg_gemini_calls": round(sum(g["num_gemini_calls"] for g in group) / n, 2),
            "avg_total_tokens": round(sum(g["total_tokens"] for g in group) / n, 1),
        }
    return summary


if __name__ == "__main__":
    # Quick CLI: `python analytics.py` prints the current cross-run summary.
    records = load_metrics_log()
    if not records:
        print("No metrics logged yet -- run test_day45.py, app.py, or eval_harness.py first.")
    else:
        print(f"{len(records)} report(s) logged in {DEFAULT_LOG_PATH}\n")
        for key, stats in summarize_by_model(records).items():
            print(f"[{key}]")
            for k, v in stats.items():
                print(f"  {k:24s} {v}")
            print()
