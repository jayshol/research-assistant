"""Writer: synthesizes a cited report from gathered sources."""
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config import GEMINI_API_KEY, LLM_MODEL

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
        blocks.append(f"[{s['id']}] {s['title']} ({s['url']})\n{s['text']}")
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


def write_report(topic: str, sources: list[dict]) -> str:
    """
    Given a topic and a flat list of source dicts (with 'id'), return the
    synthesized report text with inline [n] citations.
    """
    if not sources:
        raise ValueError("write_report called with no sources")

    source_block = _format_sources_for_prompt(sources)
    prompt = f"Topic: {topic}\n\nSources:\n\n{source_block}"

    response = _call_gemini(prompt, WRITER_SYSTEM_PROMPT, max_output_tokens=4000)
    return response.text.strip()


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
    response = _call_gemini(prompt, WRITER_SYSTEM_PROMPT, max_output_tokens=4000)
    return response.text.strip()


def format_references(sources: list[dict]) -> str:
    """Build a markdown reference list matching the [n] citations used in the report."""
    lines = ["## References"]
    for s in sources:
        lines.append(f"[{s['id']}] {s['title']} — {s['url']}")
    return "\n".join(lines)