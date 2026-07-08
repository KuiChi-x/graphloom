"""
Constants controlling the context-compaction node inserted after `history`.

All values can be overridden via environment variables (GRAPHLOOM_* prefix) to
make tuning possible without a redeploy. Keep this file free of any runtime
dependency beyond `os`.
"""
import os


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Effective model context window in tokens. Used as the denominator for the
# trigger ratio. Should roughly match the smallest model you run.
MODEL_CONTEXT_WINDOW: int = _env_int("GRAPHLOOM_MODEL_CONTEXT_WINDOW", 830_000)

# Soft trigger: when estimated context tokens reach this fraction of the
# window, the compaction node kicks in synchronously.
COMPACT_TRIGGER_RATIO: float = _env_float("GRAPHLOOM_COMPACT_TRIGGER_RATIO", 0.75)

# Target size after compaction (fraction of the window). Used as a soft
# character budget for the summarizing LLM's action_results field.
COMPACT_TARGET_RATIO: float = _env_float("GRAPHLOOM_COMPACT_TARGET_RATIO", 0.12)

# How many trailing past_steps are kept verbatim (never compacted).
COMPACT_KEEP_RECENT_STEPS: int = _env_int("GRAPHLOOM_COMPACT_KEEP_RECENT", 5)

# Anti-thrashing: hard cap on recursive compaction attempts per node call.
COMPACT_MAX_RETRY: int = _env_int("GRAPHLOOM_COMPACT_MAX_RETRY", 2)

# Emergency truncation: when retries exhaust, trim the longest action_results
# among the recent (kept) steps down to this char budget.
COMPACT_EMERGENCY_TRUNC_CHARS: int = _env_int("GRAPHLOOM_COMPACT_EMERGENCY_TRUNC_CHARS", 500)

# tiktoken encoding to use for estimation. cl100k_base is OpenAI-compatible
# and a reasonable proxy for other providers exposed via an OpenAI API shim.
COMPACT_TOKENIZER_ENCODING: str = os.environ.get("GRAPHLOOM_COMPACT_TOKENIZER", "cl100k_base")

# Sentinel key inside past_steps[0] that signals the reducer to REPLACE the
# channel instead of appending. See `add_past_steps` in state.py.
COMPACT_SENTINEL_KEY: str = "__compact_replace__"

# Concurrency ceiling for parallel subagent dispatch.
SUBAGENT_MAX_CONCURRENCY = int(os.getenv("SUBAGENT_MAX_CONCURRENCY", "10"))
