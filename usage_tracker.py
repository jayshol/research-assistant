"""
Lightweight in-process tracker for Gemini token usage.

Every _call_gemini() call site (planner, writer, critic) reports its raw
response here via track_usage(). This captures real token counts per stage
so analytics.py can compare cost/latency across different models later --
not just issue counts -- when deciding whether to keep using a given model
or swap it out.

Cache hits never call Gemini, so they never appear here -- which is
correct: no tokens were spent, so there's nothing to track. This means
"num_gemini_calls" in a metrics record already reflects real API spend,
not the number of pipeline steps.
"""
import time

_log: list[dict] = []


def track_usage(stage: str, model: str, response) -> None:
    """
    Extract token usage from a Gemini response and record it.

    Safe to call even if usage_metadata is missing/partial -- records zeros
    rather than crashing the pipeline over what's just a metrics side-channel.
    """
    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = getattr(usage, "prompt_token_count", None) or 0
    output_tokens = getattr(usage, "candidates_token_count", None) or 0
    total_tokens = getattr(usage, "total_token_count", None) or (prompt_tokens + output_tokens)

    _log.append({
        "stage": stage,  # "plan" | "write_report" | "critique" | "revise_report"
        "model": model,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "timestamp": time.time(),
    })


def mark() -> int:
    """Checkpoint: current log length. Pair with since(mark) to measure
    usage accrued between now and later -- e.g. everything one report's
    pipeline run spent, isolated from any other reports running before/after."""
    return len(_log)


def since(mark_index: int) -> list[dict]:
    """All usage entries recorded after the given mark()."""
    return _log[mark_index:]


def all_entries() -> list[dict]:
    return list(_log)
