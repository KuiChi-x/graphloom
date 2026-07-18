<!-- Once you have a logo, swap the emoji below for <img src="docs/logo.svg" width="120"> -->
<div align="center">

# 🧵 graphloom

**Stop rewriting the agent loop. `build_agent_graph` weaves a production-grade one in a single call.**

Three-horizon memory · forced step-by-step reflection · a self-correcting delivery gate · sub-agent orchestration · progressive skills — all dependency-injected, on top of [LangGraph](https://github.com/langchain-ai/langgraph).

[![PyPI](https://img.shields.io/pypi/v/graphloom?color=3775A9&logo=pypi&logoColor=white)](https://pypi.org/project/graphloom/)
[![Python](https://img.shields.io/pypi/pyversions/graphloom?logo=python&logoColor=white)](https://pypi.org/project/graphloom/)
[![License](https://img.shields.io/pypi/l/graphloom?color=success)](LICENSE)
![Built on LangGraph](https://img.shields.io/badge/built%20on-LangGraph-1C3C3C)
![Status](https://img.shields.io/badge/status-alpha-orange)
[![GitHub stars](https://img.shields.io/github/stars/KuiChi-x/graphloom?style=social)](https://github.com/KuiChi-x/graphloom)

[简体中文](README.zh-CN.md) · [Quick start](#quick-start-30-seconds) · [Why graphloom](#why-graphloom) · [The loop](#core-model-how-the-loop-flows)

</div>

---

## Quick start (30 seconds)

```bash
pip install graphloom
```

The model runs through [LiteLLM](https://docs.litellm.ai/docs/providers), so any provider works. Point it at your key and endpoint — either export env vars:

```bash
export OPENAI_API_KEY="sk-..."          # your key
# export OPENAI_API_BASE="https://your-gateway/v1"   # only if you use a proxy/gateway
```

…or pass them straight to `ChatLiteLLM(...)` in the snippet below (`api_key=...`, `api_base=...`). Then bring three things — an LLM, a system prompt, your tools — and `build_agent_graph` weaves them into a complete agent: it reasons, calls tools, remembers across turns, and knows when it's done. One `ainvoke` runs it:

```python
import asyncio
from langchain_litellm import ChatLiteLLM
from graphloom import build_agent_graph

# 1) Assemble — bring an LLM, a system prompt, and your tools
graph = build_agent_graph(
    custom_system_prompt="You are a helpful assistant.",
    tools=[],                                 # add your @tool functions here
    llm=ChatLiteLLM(
        model="gpt-4o-mini",                  # any LiteLLM model id, e.g. "claude-3-5-sonnet-20241022"
        # api_key="sk-...",                    # or set OPENAI_API_KEY in the env
        # api_base="https://your-gateway/v1",  # only for a proxy / self-hosted / non-OpenAI endpoint
    ),
    allow_direct_reply=True,                  # let a plain-text reply finish the run
)

# 2) Run
result = asyncio.run(graph.ainvoke(
    {"input_query": "Explain what a ReAct agent is."},
))

# 3) Read the answer — with allow_direct_reply, the agent's text lands in final_reply
print(result["final_reply"])
```

When no LangGraph config is supplied, Graphloom uses the fixed checkpoint thread `default`. `session_id` is optional business context for tools and observers that need per-session resources such as artifact directories or browser state; it does not select the checkpoint thread. A host that serves multiple independent checkpoint sessions must bind a graph with LangGraph's native `graph.with_config({"configurable": {"thread_id": session_id}})` before invoking.

Runnable as-is once your key is set. That single call gives you not "one LLM request" but a **whole loop**: memory across turns, automatic context compaction, resumability, sub-agent dispatch, on-demand skills — all woven in. Add tools and it starts acting, not just answering — see the [coding agent](#example-a-coding-agent-in-35-lines) below.

> [!NOTE]
> Model ids and per-provider auth follow [LiteLLM's conventions](https://docs.litellm.ai/docs/providers) — e.g. `gpt-4o-mini` reads `OPENAI_API_KEY`, `claude-3-5-sonnet-20241022` reads `ANTHROPIC_API_KEY`, and a local `ollama/llama3` needs no key at all.

> [!TIP]
> Pure-conversation agents finish with `allow_direct_reply=True`. To deliver files or structured output, use the builtin `deliver_artifact` tool — and optionally gate it behind [delivery review](#delivery-review-find-fault-nodes).

## Why graphloom

Calling an LLM once is easy. Keeping it on track across dozens of steps — without drifting, blowing the context window, losing state on interruption, or failing to parallelize sub-tasks — is the real work.

Those **chores around the loop** get rewritten by every serious agent: remembering progress across turns, coping when context overflows the window, resuming an interrupted task, scheduling parallel sub-tasks, reusing best-practice workflows.

graphloom distills them into one reusable loop. You bring the LLM, the system prompt and the tools; it weaves the rest. It is bound to **no** transport protocol, **no** database, **no** frontend; those are dependency-injected.

## What makes it different

Most "agent frameworks" hand you a loop and leave memory, context limits, and quality control as an exercise. graphloom's whole point is those hard parts:

### 🧠 Three-horizon memory — what actually keeps long tasks coherent

A bolt-on vector store is one kind of recall. What actually breaks long agent runs is the *span between right-now and an hour ago*. graphloom tiers its working memory by age, right in the loop state — the newest steps stay perfect, the oldest get summarized, and the middle degrades gracefully in between:

- **Short-term — recent steps kept verbatim.** The last few steps (default 5) are held **complete and untouched**, so the agent sees its immediate trajectory with full fidelity.
- **Mid-term — partial truncation.** As steps age past that window, their bulky bits (verbose tool dumps) are trimmed to a budget while the reasoning stays — detail fades, the thread doesn't.
- **Long-term — summarize as you go.** When context nears the window limit, the oldest steps are folded by the LLM into one **lossless archive** — IDs, keys, URLs, decisions and error-fixes kept **verbatim**, only redundant narration compressed. It re-counts tokens and retries tighter if needed, with an emergency floor so it can never loop forever. Amnesia never happens; the run just gets denser at the back.

And the whole stream is persisted through the checkpointer and replayed as `<agent_history>`, so this memory spans turns, sessions, and process restarts.

### 🪞 Reflection welded to every action — the agent can't act on autopilot

Every builtin tool's argument schema **requires** a three-part chain of thought before it runs — `last_step_review` (did the last action work?), `working_notes` (what I know so far), `next_action` (what's next). It's not a prompt suggestion; it's the tool's `StandardThoughtInput` base class, so the model literally **cannot call a tool without first reflecting**. That's the difference between an agent that notices it's stuck and one that repeats the same failing call ten times — and a built-in repeat-detector tags `[repeated N times]` straight back into the prompt when it starts to.

### ♻️ A self-correcting delivery gate — the agent grades its own homework

Completion isn't "the model said it's done." When the agent delivers, the work passes through up to two **find-fault** gates: your own validator (`custom_find_fault` — run tests, check a schema) and a builtin LLM reviewer (`find_fault`) that reads the actual artifacts against the original request. **Reject and the specific gaps get written straight back into the next prompt** (`fatal_gaps`, `recommended_rework`), sending the agent back to fix them. A quality loop, not a hope.

### 🛰️ An observer channel for the live world

An optional `observer` node runs before every turn and injects the **latest** external state — an environment snapshot, side-channel user guidance, a market tick. It's spliced into that turn only and **never pollutes memory**, because "the world right now" should be replaced by the next observation, not accreted as history. Most frameworks make you fake this by jamming updates into the message list.

### And the rest of the loop, assembled in one call

- **Sub-agent orchestration** — dispatch grouped sub-tasks (sequential across groups, parallel within a group); artifacts roll forward parent → child, and the parent's checkpointer threads through automatically.
- **Progressive skill loading** — a Claude Code-style skill menu: the agent sees only names + descriptions up front, and reads the full workflow on demand, keeping the system prompt lean.
- **Resumability** — inject a checkpointer; on user pause the graph suspends at the nearest checkpoint and `ainvoke(None)` resumes in place, zero state loss.
- **Structured artifact channel** — three manifests with distinct merge semantics carry real deliverables between agent, sub-agents and the outside world — not chat-log scraping.
- **Transport-agnostic, fully injected** — HITL, observability, persistence and the LLM provider are all externalized. Zero host coupling; depends only on LangGraph / LangChain.

## Install

```bash
pip install graphloom
```

From source:

```bash
git clone https://github.com/KuiChi-x/graphloom.git
cd graphloom
pip install -e ".[dev]"        # includes pytest / pytest-asyncio / ruff
```

Runtime deps: `langchain-core`, `langgraph`, `langchain-litellm`, `pydantic`, `tenacity`, `tiktoken`.

## Core model: how the loop flows

```
              ┌──────────────────── not finished, continue ────────────────────┐
              │                                                                 │
  observer? → ai → tool ──route──→ history → compaction ─────────────────────→ ai
                     │
                     └──end_tag=True──→ (find_fault?) ──→ finish ──→ END
```

- **ai** calls the LLM, merges the streamed response, records a pending step (with the three-part chain of thought).
- **tool** runs the requested tool calls; with no tool call and `allow_direct_reply=True`, a plain-text reply finishes the run.
- **route** checks `end_tag`: if true, head to completion (optionally via review), otherwise back to history.
- **history** folds the tool results into one completed step.
- **compaction** folds early steps into a lossless summary once the estimated tokens exceed the threshold.
- **finish** closes out the delivery and marks `agent_status=done`.

**Completion contract**: any tool ends the loop by putting `end_tag=True` in its return. The builtin `deliver_artifact` is the canonical way; pure-conversation agents use `allow_direct_reply=True`.

## An agent's memory

graphloom's memory isn't a bolt-on vector store — it's the **loop state itself**, tiered by age:

- **Short-term — recent steps, verbatim.** The last `COMPACT_KEEP_RECENT_STEPS` steps (default 5) are kept complete and untouched. Individual tool results are hard-capped (`_HARD_TRUNCATE_LIMIT`, 50 KB) so one giant payload can't blow the turn.
- **Mid-term — partial truncation.** Once compaction runs, the kept window's bulkiest `action_results` can be trimmed to a per-field char budget (and, in the worst case, an emergency truncation) — the reasoning survives, the raw dumps shrink.
- **Long-term — summarize as you go.** Steps older than the kept window are folded by the LLM into one lossless archive step: numbers, IDs, URLs, user constraints, and errors-plus-fixes kept **verbatim**, only redundant narration compressed. `compaction` re-estimates tokens and retries with tighter budgets until it fits.
- **The step stream is the memory** — steps accumulate into `past_steps`, persisted with the checkpointer, present across turns and sessions. Rendered back into the prompt as `<agent_history>`, so the agent always sees the path it has walked.

### Reflection is a required schema field, not a suggestion

Every builtin tool subclasses **`StandardThoughtInput`** — a Pydantic base with three required fields the LLM must fill on *every* call:

```python
from graphloom import StandardThoughtInput  # exported from the package
from pydantic import Field

class MyToolInput(StandardThoughtInput):
    # inherits required: last_step_review / working_notes / next_action
    query: str = Field(description="what to search for")
```

Subclass it for your own tools and the same forced chain of thought applies to them — `last_step_review` (judge the last action), `working_notes` (carry facts forward), `next_action` (commit to the next move). This is the framework's core defense against drift and forgetting; it's why the model reflects before it acts instead of after it fails. (`PlannerThoughtInput` extends it further for orchestration tools like `dispatch_subagents`.)

## Skills: progressive disclosure

A skill is a directory containing a `SKILL.md` whose front-matter declares a `name` and `description`. The agent initially sees only the **name + description + location** of each skill; it reads the full workflow (and any referenced scripts/resources) on demand via `read_artifact` when a task actually needs it. This gives the agent a menu of deep, reusable workflows without bloating the system prompt.

```python
graph = build_agent_graph(
    custom_system_prompt=PROMPT,
    tools=[...],
    llm=llm,
    available_skills=["pdf_extraction", "sql_report"],               # whitelist
    skills_dirs=["/path/to/builtin-skills", "/path/to/user-skills"],  # your library; not hard-coded
)
```

Where skill libraries live is up to you — inject one or more roots through `skills_dirs` in priority order. A later root overrides an earlier skill with the same name; the framework knows nothing about the skill contents.

## Sub-agent orchestration

With `subagents` configured, the framework injects a `dispatch_subagents` tool that lets the main agent break the next phase into a grouped plan: **groups run in ascending `group_id` order, steps within a group run in parallel**, upstream artifacts feed downstream, and the parent's checkpointer is threaded into every sub-graph.

```python
from graphloom import SubAgentSpec

graph = build_agent_graph(
    custom_system_prompt=PROMPT,
    tools=[...],
    llm=llm,
    subagents=[
        SubAgentSpec(agent_name="researcher", description="Research and gather evidence", factory=make_researcher),
        SubAgentSpec(agent_name="writer",     description="Write the report",            factory=make_writer),
    ],
)
```

## The observer node

`observer` is an optional custom node. When configured it becomes the graph's entry point and runs before `ai` on every turn (`compaction → observer → ai`). Its job is to inject the **latest external state** into the current turn — an environment snapshot, live side-channel guidance from the user, signals from external systems.

The `observer_message_parts` field it writes has **no accumulation semantics**: it is fully overwritten each turn, spliced only into that turn's messages to the LLM, and **never written to `past_steps` or memory**. This is deliberate — an observation reflects "the world right now"; once stale it should be replaced by a fresh one, not left to accrete as historical noise.

```python
async def observer(state):
    snapshot = await read_live_environment(state["session_id"])
    return {"observer_message_parts": [HumanMessage(content=f"[live state]\n{snapshot}")]}

graph = build_agent_graph(custom_system_prompt=PROMPT, tools=[...], llm=llm, observer=observer)
```

## Delivery review: find-fault nodes

Once the agent signals completion (`end_tag=True`), the delivery can pass through review gates before actually finishing. graphloom supports two stackable stages:

- **`custom_find_fault`** — your own review node, runs first. Plug in any external validation here (run tests, validate a schema, enforce business rules).
- **`find_fault`** — the builtin LLM self-review node. Give it a review prompt; it reads the delivered artifacts, checks them structurally against the original request, and outputs pass/fail plus a list of gaps and rework suggestions.

Routing: `end_tag → custom_find_fault? (runs first if set) → find_fault? (runs next) → finish`. A **failed** review clears the delivery manifest and sends the agent **back to history to redo the work** with feedback; only a pass proceeds to finish. A text-only delivery (no artifacts) is treated as a pass by the review nodes.

```python
graph = build_agent_graph(
    custom_system_prompt=PROMPT, tools=[...], llm=llm,
    custom_find_fault=my_test_runner_node,          # first: your validation
    find_fault="You are a strict reviewer. Verify every requirement is met.",  # then: LLM self-review
)
```

## Artifact communication

An agent's output isn't stuffed into the chat log — it flows through **structured artifact manifests**. State holds three, each with distinct merge semantics, together forming the delivery and hand-off channel:

| Manifest | Merge semantics | Meaning |
|---|---|---|
| `input_artifact_manifest` | replace | Reference artifacts passed in from outside / upstream |
| `current_delivery_manifest` | replace | Artifacts submitted this turn via `deliver_artifact`, awaiting review |
| `approved_artifact_manifest` | merge + dedup | Reviewed, deliverable artifacts; sub-agent outputs roll into here too |

The builtin artifact tools (`write / read / patch / deliver`) read and write these manifests; `deliver_artifact` writes `end_tag=True` together with `current_delivery_manifest` to trigger completion. During sub-agent orchestration, an upstream `approved_artifact_manifest` is fed in as the downstream `input_artifact_manifest` — artifacts are the hand-off language between agents.

## `build_agent_graph` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `custom_system_prompt` | `str` | required | The agent's system prompt; concatenated with the framework's common prompt. |
| `tools` | `list` | required | Your LangChain tools. Builtin artifact tools are auto-injected (yours win on name clash). |
| `llm` | `BaseChatModel` | required | Any LangChain chat model; shared by the ai and compaction nodes. |
| `find_fault` | `str \| callable` | `None` | A string assembles an artifact self-review node with that prompt; a node is used directly. |
| `custom_find_fault` | `callable` | `None` | Your own review node, runs before `find_fault`. |
| `observer` | `callable` | `None` | A node run before `ai` each turn (inject external observation / guidance). |
| `subagents` | `list[SubAgentSpec]` | `None` | When set, injects `dispatch_subagents` for multi-agent orchestration. |
| `checkpointer` | LangGraph saver | `None` | State persistence; also threaded into sub-graphs. |
| `tool_filter` | `callable` | `None` | `(state, config) -> set of hidden tool names`, to prune tools per turn. |
| `allow_direct_reply` | `bool` | `False` | Let the LLM finish with a plain-text reply instead of a tool call. |
| `available_skills` | `list[str]` | `None` | Whitelist of skills exposed to the agent. |
| `skills_dirs` | `Sequence[str]` | `None` | Skill-library roots scanned in order; a later root overrides an earlier skill with the same name. |

## Runtime context

The host injects runtime dependencies via `config["configurable"]["runtime_context"]`. The framework **passes it through verbatim** to tools and sub-graphs without interpreting it:

| Key | Purpose |
|---|---|
| `artifact_base_dir` | Workspace root for artifact tools (or the `GRAPHLOOM_ARTIFACT_BASE_DIR` env var). |
| `callbacks` | LangGraph callbacks passed to sub-graphs (token streaming and other observability). |
| `cancel_event` | An `asyncio.Event`; once set, raises `GraphInterrupt` at the nearest checkpoint to pause. |
| `user_id` | Session ownership identifier, for tools to use. |

## Inside / outside the framework

| Inside (graphloom's job) | Outside (your job) |
|---|---|
| The loop and structural nodes | Tools — incl. HITL, clarification, any business tool |
| `AgentState`, the three-part chain of thought, compaction | Transport / wire — ws event codes, streaming sentinels |
| Builtin artifact tools, optional dispatch / find_fault | Observability — via LangGraph's standard `callbacks` |
| The progressive skill-loading mechanism | Skill contents — the roots in your `skills_dirs` list |
| Injection seams: llm / checkpointer / tools / runtime_context | Persistence backend, LLM provider |

## Example: a coding agent in ~35 lines

Three tools (read, write, run) wired into `build_agent_graph` — that's a Claude Code / Codex-style coding agent:

```python
import asyncio
import subprocess
from pathlib import Path

from langchain_core.tools import tool
from langchain_litellm import ChatLiteLLM
from graphloom import build_agent_graph


@tool
def read_file(path: str) -> str:
    """Read a file's contents."""
    return Path(path).read_text(encoding="utf-8")


@tool
def write_file(path: str, content: str) -> str:
    """Write contents to a file."""
    Path(path).write_text(content, encoding="utf-8")
    return f"wrote {path}"


@tool
def run_command(cmd: str) -> str:
    """Run a shell command (only in a workspace you trust)."""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


graph = build_agent_graph(
    custom_system_prompt="You are a coding agent. Read, write and run code to satisfy the request.",
    tools=[read_file, write_file, run_command],
    llm=ChatLiteLLM(model="gpt-4o-mini"),
    allow_direct_reply=True,
)

session_id = "demo"
session_graph = graph.with_config({"configurable": {"thread_id": session_id}})
print(asyncio.run(session_graph.ainvoke(
    {"input_query": "create fizzbuzz.py and run it"},
))["final_reply"])
```

The agent reads/writes files, runs commands, verifies the result, then replies directly. Auth is the same as the [quick start](#quick-start-30-seconds) — env var or `api_key=`/`api_base=` on `ChatLiteLLM`. `run_command` executes arbitrary shell — **only run it against a workspace you trust**.

## Project layout

```
src/graphloom/
  __init__.py            public API: build_agent_graph / AgentState / SubAgentSpec / …
  graph_builder.py       entry point
  config.py              tunables (compaction thresholds, concurrency cap, …)
  model/                 state, reducers, schemas, sub-agent specs
  nodes/                 ai / tool / history / compaction / finish / find_fault / interrupt_guard
  tools/                 artifact (4 tools), dispatch (sub-agent orchestration)
  prompt/                system prompts, prompt stack, context renderer, message builder
  skills/                skill loading (SKILL.md parsing + progressive-disclosure prompt section)
  util/                  message utils, token counter, session store
```

## Status

Alpha — extracted from a production agent codebase, generalizing the loop and stripping all host coupling. API may change before 1.0. The core loop, short-term memory, context compaction, sub-agent dispatch and skill loading are all verified against real and stubbed LLMs.

## Contributing

Issues and PRs welcome. Set up with `pip install -e ".[dev]"`, and run `pytest` and `ruff check` before submitting.

## License

[Apache 2.0](LICENSE)
