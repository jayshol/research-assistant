# Research & Summarization Agent

An agentic pipeline: given a topic, it **plans** a set of searches, **executes** them via
web search, **tracks sources**, **writes** a cited report, and runs a **critic** pass to
flag unsupported claims before finalizing.

## Architecture

```
topic
  │
  ▼
[Planner]  ── LLM breaks topic into 3-6 sub-questions / search queries
  │
  ▼
[Search + Fetch]  ── Tavily search API + page text extraction, per sub-question
  │
  ▼
[Source Store]  ── structured list of {query, source_url, title, text}
  │
  ▼
[Writer]  ── LLM drafts report, citing sources as [1], [2], ...
  │
  ▼
[Critic]  ── LLM re-reads draft against sources, flags unsupported claims
  │
  ▼
[Revise]  ── Writer fixes flagged issues (one revision pass)
  │
  ▼
final cited report
```

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Get API keys:
   - **Gemini API key** — https://aistudio.google.com/apikey (free tier, no card required — used for planning/writing/critiquing)
   - **Tavily API key** — https://tavily.com (free tier, built for LLM search use cases)

3. Copy `.env.example` to `.env` and fill in your keys:
   ```
   cp .env.example .env
   ```

## Day 1-2: test search + fetch tooling

```
python test_day1.py "your topic here"
```

This runs the planner to generate search queries, executes them, and prints the
retrieved sources — no writing/critique yet. Confirms the plumbing works before you
build the writer and critic on top.

## Files

- `config.py` — loads API keys, model names, constants
- `tools/search.py` — Tavily search wrapper
- `tools/fetch.py` — fetches a URL and extracts readable text
- `planner.py` — LLM call that turns a topic into a search plan
- `test_day1.py` — Day 1-2 smoke test: plan → search → fetch → print sources

Writer and critic modules come next (Day 3-5) — this scaffold covers the
foundation described in the Day 1-2 plan.
