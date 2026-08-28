"""Central config: API keys and shared constants."""
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Model used for planning / writing / critiquing.
LLM_MODEL = "gemini-3.6-flash"

# How many sub-questions the planner should aim to produce.
MIN_QUERIES = 3
MAX_QUERIES = 6

# How many search results to fetch per sub-question.
RESULTS_PER_QUERY = 3

# Max characters of page text to keep per source (keeps prompts manageable).
MAX_SOURCE_CHARS = 4000

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set. Copy .env.example to .env and fill it in.")
if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY not set. Copy .env.example to .env and fill it in.")
