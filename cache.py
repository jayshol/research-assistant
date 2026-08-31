"""
Lightweight disk cache for Gemini calls during development.

Caches a function's return value to a JSON file, keyed by the function name
plus a hash of its arguments. This means re-running the same topic (or the
same draft + revision notes) during testing hits the cache instead of
re-spending API quota -- useful given the free tier's low daily request cap.

Not meant as a production correctness layer -- just a dev convenience. To
force a fresh call (e.g. because you changed a prompt and want new output),
either delete the relevant file(s) under .cache/, delete the whole .cache/
directory, or set the environment variable RESEARCH_ASSISTANT_NO_CACHE=1 to
bypass caching entirely for a run.
"""
import hashlib
import json
import os
from functools import wraps
from pathlib import Path

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DISABLED = os.environ.get("RESEARCH_ASSISTANT_NO_CACHE") == "1"


def _hash_args(*args, **kwargs) -> str:
    """Deterministic short hash of a function's arguments. Relies on args
    being JSON-serializable (str, list, dict of those) -- true for every
    Gemini-calling function in this project (topic strings, source dicts,
    draft text, issue lists)."""
    payload = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def disk_cache(func):
    """
    Decorator: cache a function's return value to a JSON file on disk under
    .cache/<function_name>/<args_hash>.json. Works for any JSON-serializable
    return value (str for write_report/revise_report, list[dict] for
    plan/critique).

    Two calls with the same arguments (e.g. same topic, or same draft +
    revision_notes) return the cached result instead of calling Gemini again.
    """
    subdir = CACHE_DIR / func.__name__

    @wraps(func)
    def wrapper(*args, **kwargs):
        if CACHE_DISABLED:
            return func(*args, **kwargs)

        subdir.mkdir(parents=True, exist_ok=True)
        key = _hash_args(*args, **kwargs)
        cache_file = subdir / f"{key}.json"

        if cache_file.exists():
            print(f"  [cache hit] {func.__name__} ({key}) — skipping API call")
            with open(cache_file) as f:
                return json.load(f)["result"]

        result = func(*args, **kwargs)

        with open(cache_file, "w") as f:
            json.dump({"result": result}, f, indent=2)

        return result

    return wrapper


def cache_lookup(func_name: str, *args, **kwargs):
    """Manual cache read, for functions that need fine control over what
    gets cached (e.g. critic.critique, which must NOT cache a parse-failure
    fallback as if it were a genuine 'no issues found' result). Returns
    (hit: bool, key: str, value). value is None on a miss."""
    if CACHE_DISABLED:
        return False, None, None
    subdir = CACHE_DIR / func_name
    key = _hash_args(*args, **kwargs)
    cache_file = subdir / f"{key}.json"
    if cache_file.exists():
        print(f"  [cache hit] {func_name} ({key}) — skipping API call")
        with open(cache_file) as f:
            return True, key, json.load(f)["result"]
    return False, key, None


def cache_store(func_name: str, key: str, result) -> None:
    """Manual cache write, paired with cache_lookup. Caller decides whether
    a given result is safe/worth persisting."""
    if CACHE_DISABLED:
        return
    subdir = CACHE_DIR / func_name
    subdir.mkdir(parents=True, exist_ok=True)
    cache_file = subdir / f"{key}.json"
    with open(cache_file, "w") as f:
        json.dump({"result": result}, f, indent=2)
