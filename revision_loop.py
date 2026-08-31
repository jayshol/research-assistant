"""
Bounded multi-pass revision loop with convergence tracking.

Wraps write -> critique -> revise into a loop that runs up to `max_passes`
revisions, stopping early once the critic finds zero issues. Tracks the
issue count at each pass so the caller can report/display convergence,
e.g. issue_counts == [3, 1, 0].
"""
from dataclasses import dataclass, field

from writer import write_report, revise_report, format_references
from critic import critique, format_issues_for_revision


@dataclass
class RevisionResult:
    """Full trace of a self-correction run, useful for both the UI and eval logging."""
    topic: str
    final_report: str
    references: str
    issue_counts: list[int]          # e.g. [3, 1, 0] -> critic ran 3 times
    issues_by_pass: list[list[dict]] # raw issue dicts per pass, for eval harness later
    num_revisions: int                # how many revise_report calls were actually made
    converged: bool                   # True if the last pass found 0 issues

    def summary_line(self) -> str:
        """A short human-readable convergence sentence, e.g.
        'Issues found: 3 -> 1 -> 0 after 2 revisions.'"""
        trace = " \u2192 ".join(str(n) for n in self.issue_counts)
        if self.converged:
            return f"Issues found: {trace} after {self.num_revisions} revision(s)."
        return (
            f"Issues found: {trace} after {self.num_revisions} revision(s) "
            f"(did not fully converge within the pass limit)."
        )


def run_self_correcting_pipeline(
    topic: str,
    sources: list[dict],
    max_passes: int = 3,
) -> RevisionResult:
    """
    Runs: write -> critique -> [revise -> critique] up to max_passes times total
    critique calls, stopping early if a critique pass finds zero issues.

    max_passes counts critique calls, not revisions -- e.g. max_passes=3 means
    at most 3 critique calls and at most 2 revise calls (since the first draft
    is critiqued before any revision happens).

    Returns a RevisionResult with the final report and the full convergence
    trace (issue_counts, issues_by_pass) regardless of whether it converged.
    """
    if max_passes < 1:
        raise ValueError("max_passes must be >= 1")

    draft = write_report(topic, sources)

    issue_counts: list[int] = []
    issues_by_pass: list[list[dict]] = []
    num_revisions = 0

    current = draft
    for pass_num in range(1, max_passes + 1):
        issues = critique(current, sources)
        issue_counts.append(len(issues))
        issues_by_pass.append(issues)

        if not issues:
            return RevisionResult(
                topic=topic,
                final_report=current,
                references=format_references(sources),
                issue_counts=issue_counts,
                issues_by_pass=issues_by_pass,
                num_revisions=num_revisions,
                converged=True,
            )

        # Out of passes -- don't revise again, just return what we have.
        if pass_num == max_passes:
            break

        revision_notes = format_issues_for_revision(issues)
        current = revise_report(topic, sources, current, revision_notes)
        num_revisions += 1

    return RevisionResult(
        topic=topic,
        final_report=current,
        references=format_references(sources),
        issue_counts=issue_counts,
        issues_by_pass=issues_by_pass,
        num_revisions=num_revisions,
        converged=False,
    )
