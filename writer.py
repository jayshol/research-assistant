"""Writer: synthesizes a cited report from gathered sources."""
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config import GEMINI_API_KEY, LLM_MODEL
from cache import disk_cache
import usage_tracker

_client = genai.Client(api_key=GEMINI_API_KEY)

WRITER_SYSTEM_PROMPT = """You are a research report writer.

You will be given a topic and a numbered list of sources, each with an id, \
title, url, and extracted text. Write a well-organized report on the topic \
that synthesizes the source material.

Rules:
- Every factual claim must be followed by a citation marker like [1] or [2] \
referring to the source id(s) that support it.
- Only cite source ids that were actually given to you. Never invent a \
citation number.
- If a claim isn't supported by any given source, don't include it — do not \
rely on outside/prior knowledge.
- Use multiple citations where a claim is supported by more than one source, \
e.g. [1][3].
- Write in clear prose organized into sections with short headers. Aim for \
a well-rounded report, not a bare list of facts.
- Do not include a references/bibliography section — that will be generated \
separately.

Respond with ONLY the report text (markdown headers are fine), no preamble.
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
def _call_gemini(prompt: str, system_prompt: str, max_output_tokens: int):
    return _client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_output_tokens,
        ),
    )


def _was_truncated(response) -> bool:
    """True if the response was cut off for hitting max_output_tokens rather
    than finishing naturally -- catches silently-truncated reports that would
    otherwise end mid-sentence with no error raised."""
    try:
        return response.candidates[0].finish_reason == "MAX_TOKENS"
    except (AttributeError, IndexError):
        return False


def _generate_report_text(prompt: str, token_budgets: list[int], stage: str) -> str:
    """Call Gemini, retrying with a larger token budget if the response was
    truncated. Reports are long (full prose + citations), so a single fixed
    budget risks cutting a report off mid-sentence. `stage` ("write_report"
    or "revise_report") tags the usage record so analytics can distinguish
    initial drafts from revisions."""
    text = ""
    for max_output_tokens in token_budgets:
        response = _call_gemini(prompt, WRITER_SYSTEM_PROMPT, max_output_tokens)
        usage_tracker.track_usage(stage, LLM_MODEL, response)
        text = response.text.strip()
        if not _was_truncated(response):
            return text
    # Exhausted all budgets and still truncated -- return what we have rather
    # than crashing the pipeline, but this should be rare at the top budget.
    print(f"\n  [warning] Report may be truncated even at {token_budgets[-1]} tokens.")
    return text


@disk_cache
def write_report(topic: str, sources: list[dict]) -> str:
    """
    Given a topic and a flat list of source dicts (with 'id'), return the
    synthesized report text with inline [n] citations.
    """
    if not sources:
        raise ValueError("write_report called with no sources")

    source_block = _format_sources_for_prompt(sources)
    prompt = f"Topic: {topic}\n\nSources:\n\n{source_block}"

    return _generate_report_text(prompt, token_budgets=[6000, 8000], stage="write_report")


@disk_cache
def revise_report(topic: str, sources: list[dict], prior_draft: str, revision_notes: str) -> str:
    """
    Given the prior draft and a plain-text list of issues (from critic.py),
    return a revised report addressing them.
    """
    source_block = _format_sources_for_prompt(sources)
    prompt = (
        f"Topic: {topic}\n\n"
        f"Sources:\n\n{source_block}\n\n"
        f"---\n\nYour previous draft:\n\n{prior_draft}\n\n"
        f"---\n\nRevision notes from the critic:\n\n{revision_notes}\n\n"
        f"Rewrite the full report, fixing every issue listed above. "
        f"Follow the same rules as before (citations, no invented facts)."
    )
    # Revision prompts are longer than the initial write (they carry the
    # prior draft + notes too), so start at the same budget but allow a
    # bigger ceiling on retry.
    return _generate_report_text(prompt, token_budgets=[6000, 9000], stage="revise_report")


def format_references(sources: list[dict]) -> str:
    """Build a markdown reference list matching the [n] citations used in the report.
    Includes each source's credibility type (e.g. "academic/journal", "blog/company")
    when available, so the reader can weigh citations differently -- a claim backed
    by [2] a peer-reviewed journal carries different weight than one backed by
    [4] a company blog post."""
    lines = ["## References"]
    for s in sources:
        type_suffix = f" — *{s['source_type_label']}*" if "source_type_label" in s else ""
        lines.append(f"[{s['id']}] {s['title']} — {s['url']}{type_suffix}")
    return "\n".join(lines)
