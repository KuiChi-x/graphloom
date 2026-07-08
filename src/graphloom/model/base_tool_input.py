from typing import List, Optional

from pydantic import BaseModel, Field


class PlannerSummary(BaseModel):
    """High-level planner reasoning."""
    task_analysis: str = Field(
        description="What the user wants, task type, and likely difficulty.",
    )
    progress_review: str = Field(
        description="What has already been accomplished and what is still missing.",
    )
    strategy: str = Field(
        description="What to do in the current phase only.",
    )
    concise_assessment: str = Field(
        description="Short user-facing summary of why this plan makes sense now.",
    )


# ---------------------------------------------------------------------------
# Base thought inputs
# ---------------------------------------------------------------------------

class StandardThoughtInput(BaseModel):
    """Unified chain-of-thought parameters for all agent tools."""
    model_config = {"extra": "allow"}

    last_step_review: str = Field(
        description="Concise one-sentence analysis of your last action. Clearly state success, failure, or uncertain."
    )
    working_notes: str = Field(
        description="1-3 sentences of specific notes on this step and overall progress. Put here everything that will help you track progress in future steps. Like counting pages visited, items found, etc."
    )
    next_action: str = Field(
        description="State the next immediate goal and action to achieve it, in one clear sentence."
    )

# ---------------------------------------------------------------------------
# Planner-level thought input (for main orchestrator tools)
# ---------------------------------------------------------------------------

class PlannerThoughtInput(StandardThoughtInput):
    """Extends standard chain-of-thought with planner-specific structured fields.

    Used by dispatch_subagents and request_user_interaction in the main orchestrator.
    Forces the LLM to produce structured planning outputs with every tool call.
    """
    planner_summary: PlannerSummary = Field(
        description="High-level planner reasoning for the user. "
                    "Covers only the current phase.",
    )
