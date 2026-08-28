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
def _call_gemini(topic: str):
    return _client.models.generate_content(
        model=LLM_MODEL,
        contents=f"Topic: {topic}",
        config=types.GenerateContentConfig(
            system_instruction=PLANNER_SYSTEM_PROMPT,
            max_output_tokens=1000,
        ),
    )


def plan(topic: str) -> list[dict]:
    """
    Given a topic, return a list of {"sub_question": ..., "query": ...} dicts.
    """
    response = _call_gemini(topic)

    text = response.text.strip()
    # Defensive cleanup in case the model wraps output in fences despite instructions.
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        plan_items = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Planner did not return valid JSON:\n{text}") from e

    return plan_items