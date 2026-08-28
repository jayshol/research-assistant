"""Critic: checks the draft report against the source material and flags
unsupported or overstated claims. This is the self-correction loop."""
import json
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config import GEMINI_API_KEY, LLM_MODEL

_client = genai.Client(api_key=GEMINI_API_KEY)

CRITIC_SYSTEM_PROMPT = """You are a fact-checking critic for a research report.

You will be given a draft report (with inline [n] citations) and the numbered \
source material it was supposed to be based on. Your job is to check the \
report against the sources, not against your own knowledge.

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
        blocks.append(f"[{s['id']}] {s['title']} ({s['url']})\n{s['text']}")
    return "\n\n---\n\n".join(blocks)


@retry(
    retry=retry_if_exception_type(genai_errors.ServerError),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    reraise=True,
)
def _call_gemini(prompt: str):
    return _client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=CRITIC_SYSTEM_PROMPT,
            max_output_tokens=6000,
        ),
    )


def critique(draft: str, sources: list[dict]) -> list[dict]:
    """
    Given a draft report and the source list it was based on, return a list
    of flagged issue dicts (see CRITIC_SYSTEM_PROMPT for schema). Empty list
    means no issues found.
    """
    source_block = _format_sources_for_prompt(sources)
    prompt = f"Draft report:\n\n{draft}\n\n---\n\nSources:\n\n{source_block}"

    response = _call_gemini(prompt)
    text = response.text.strip()

    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        issues = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"\n  [warning] Critic response was not valid JSON (likely truncated). "
              f"Skipping critique for this run.\n  Raw tail: ...{text[-200:]}")
        return []
    
    return issues


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