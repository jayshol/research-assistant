"""Critic: checks the draft report against the source material and flags
unsupported or overstated claims. This is the self-correction loop."""
import json
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config import GEMINI_API_KEY, LLM_MODEL
from cache import cache_lookup, cache_store
import usage_tracker

_client = genai.Client(api_key=GEMINI_API_KEY)

CRITIC_SYSTEM_PROMPT = """You are a fact-checking critic for a research report.

You will be given a draft report (with inline [n] citations) and the numbered \
source material it was supposed to be based on. Your job is to check the \
report against the sources, not against your own knowledge.

Each source may be labeled with a source type in brackets, e.g. \
"[source type: academic/journal]" or "[source type: blog/company]". Apply \
extra scrutiny to claims cited to lower-credibility source types (blog/company, \
social media) than to higher-credibility ones (academic/journal, established \
news) -- weaker sources are more likely to themselves contain vague, hedged, \
or overstated language that a report can subtly amplify. Do NOT flag a claim \
merely for citing a lower-credibility source; only flag it if it falls into \
one of the four issue types below.

For each issue you find, flag it as one of:
- "unsupported": a claim with a citation [n], but source [n] doesn't actually \
say that.
- "overstated": a claim that goes further than what the cited source says \
(e.g. source says "may reduce" but report says "reduces").
- "missing_citation": a specific factual claim with no citation at all.
- "invalid_citation": a citation number that doesn't correspond to any given \
source.

Respond with ONLY a JSON array of issue objects, no other text, no markdown \
fences. Each object must have:
- "issue_type": one of the four types above
- "claim": the exact sentence or phrase from the draft with the problem
- "cited_source_id": the source id cited (or null if missing_citation)
- "explanation": one sentence on what's wrong

If there are no issues, respond with an empty array: []

Example output:
[
  {"issue_type": "overstated", "claim": "AI has solved climate modeling uncertainty.", "cited_source_id": 3, "explanation": "Source [3] says AI 'reduces' uncertainty, not that it 'solved' it."}
]
"""


def _format_sources_for_prompt(sources: list[dict]) -> str:
    blocks = []
    for s in sources:
        type_tag = f" [source type: {s['source_type_label']}]" if "source_type_label" in s else ""
        blocks.append(f"[{s['id']}] {s['title']} ({s['url']}){type_tag}\n{s['text']}")
    return "\n\n---\n\n".join(blocks)


@retry(
    retry=retry_if_exception_type(genai_errors.ServerError),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    reraise=True,
)
def _call_gemini(prompt: str, max_output_tokens: int):
    return _client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=CRITIC_SYSTEM_PROMPT,
            max_output_tokens=max_output_tokens,
        ),
    )


def _was_truncated(response) -> bool:
    """True if the response was cut off for hitting max_output_tokens rather
    than finishing naturally. A truncated critic response looks exactly like
    a malformed one (invalid JSON) but has a different root cause, so we
    check this explicitly instead of relying on JSONDecodeError alone."""
    try:
        return response.candidates[0].finish_reason == "MAX_TOKENS"
    except (AttributeError, IndexError):
        return False


def critique(draft: str, sources: list[dict]) -> list[dict]:
    """
    Given a draft report and the source list it was based on, return a list
    of flagged issue dicts (see CRITIC_SYSTEM_PROMPT for schema). Empty list
    means no issues found.

    Retries with a larger token budget if the response was truncated. If it's
    still unparseable after exhausting all budgets, raises RuntimeError
    rather than silently returning [] -- a failed critique is NOT the same
    thing as "no issues found," and treating it that way would corrupt
    eval_harness.py's convergence numbers (a topic where the critic never
    successfully ran would get counted as "converged with 0 issues").

    Cached manually (not via @disk_cache) so that a failed run is never
    written to the cache as if it were a genuine result.
    """
    hit, key, cached = cache_lookup("critique", draft, sources)
    if hit:
        return cached

    source_block = _format_sources_for_prompt(sources)
    prompt = f"Draft report:\n\n{draft}\n\n---\n\nSources:\n\n{source_block}"

    token_budgets = [6000, 9000, 12000]
    last_text = ""

    for max_output_tokens in token_budgets:
        response = _call_gemini(prompt, max_output_tokens)
        usage_tracker.track_usage("critique", LLM_MODEL, response)
        text = response.text.strip()
        last_text = text

        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        if _was_truncated(response):
            continue  # don't even try to parse -- known truncated, retry bigger

        try:
            issues = json.loads(text)
        except json.JSONDecodeError:
            continue  # malformed for some other reason -- retry bigger just in case

        cache_store("critique", key, issues)
        return issues

    # Exhausted every budget and still couldn't get valid JSON -- surface
    # this loudly instead of masking it as "no issues found."
    raise RuntimeError(
        f"Critic failed to return valid JSON after {len(token_budgets)} attempts "
        f"(up to {token_budgets[-1]} tokens). Raw tail: ...{last_text[-200:]}"
    )


def format_issues_for_revision(issues: list[dict]) -> str:
    """Turn flagged issues into a plain-text note the writer can revise against."""
    if not issues:
        return "No issues found."
    lines = ["The following issues were found in your draft — please revise:"]
    for i, issue in enumerate(issues, 1):
        lines.append(
            f"{i}. [{issue['issue_type']}] \"{issue['claim']}\" "
            f"(cited source: {issue.get('cited_source_id')}) — {issue['explanation']}"
        )
    return "\n".join(lines)
