"""Planner: turns a research topic into a structured set of search queries."""
import json
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from config import GEMINI_API_KEY, LLM_MODEL, MIN_QUERIES, MAX_QUERIES
from cache import disk_cache
import usage_tracker

_client = genai.Client(api_key=GEMINI_API_KEY)

PLANNER_SYSTEM_PROMPT = f"""You are a research planning assistant.

Given a topic, break it down into {MIN_QUERIES}-{MAX_QUERIES} distinct search queries \
that together would give a well-rounded understanding of the topic. Aim for coverage \
across angles like: background/definition, current state, key players or examples, \
controversies or open questions, and recent developments — but only include angles \
that make sense for the specific topic.

Respond with ONLY a JSON array of objects, no other text, no markdown fences. \
Each object must have:
- "sub_question": a short natural-language question this query investigates
- "query": the actual search engine query string to run

Example output format:
[
  {{"sub_question": "What is X?", "query": "X definition overview"}},
  {{"sub_question": "What are recent developments in X?", "query": "X news 2026"}}
]
"""


@retry(
    retry=retry_if_exception_type(genai_errors.ServerError),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    reraise=True,
)
def _call_gemini(topic: str, max_output_tokens: int):
    return _client.models.generate_content(
        model=LLM_MODEL,
        contents=f"Topic: {topic}",
        config=types.GenerateContentConfig(
            system_instruction=PLANNER_SYSTEM_PROMPT,
            max_output_tokens=max_output_tokens,
        ),
    )


def _parse_plan_json(text: str) -> list[dict]:
    text = text.strip()
    # Defensive cleanup in case the model wraps output in fences despite instructions.
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)  # may raise json.JSONDecodeError


@disk_cache
def plan(topic: str) -> list[dict]:
    """
    Given a topic, return a list of {"sub_question": ..., "query": ...} dicts.

    1000 tokens is tight for a full plan (several sub-questions + queries) and
    can get the response cut off mid-string, producing invalid JSON. If that
    happens, retry once with a larger budget before giving up -- a silently
    truncated plan would otherwise crash the whole pipeline.
    """
    token_budgets = [2000, 3500]
    last_text = ""

    for max_output_tokens in token_budgets:
        response = _call_gemini(topic, max_output_tokens)
        usage_tracker.track_usage("plan", LLM_MODEL, response)
        last_text = response.text.strip()
        try:
            return _parse_plan_json(last_text)
        except json.JSONDecodeError:
            continue  # try again with a bigger budget

    raise ValueError(f"Planner did not return valid JSON:\n{last_text}")
