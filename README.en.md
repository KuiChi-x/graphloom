<div align="center">

[中文](README.md) · **English**

# graphloom

**One loop, woven through all your agents.**

A generic agent framework on top of [LangGraph](https://github.com/langchain-ai/langgraph). A single `build_agent_graph` assembles a ReAct agent complete with the loop, short-term memory, context compaction, resumability, sub-agent orchestration and progressive skill loading — with the LLM, tools and checkpointer all dependency-injected, and no coupling to any transport or business layer.

![Python](https://img.shields.io/badge/python-3.10+-3776AB)
![Built on](https://img.shields.io/badge/built%20on-LangGraph-1C3C3C)
![License](https://img.shields.io/badge/license-MIT-success)
![Status](https://img.shields.io/badge/status-alpha-orange)

</div>

---

## Table of contents

- [Why graphloom](#why-graphloom)
- [Highlights](#highlights)
- [Install](#install)
- [Quick start](#quick-start)
- [Core model: how the loop flows](#core-model-how-the-loop-flows)
- [An agent's memory](#an-agents-memory)
- [Skills: progressive disclosure](#skills-progressive-disclosure)
- [Sub-agent orchestration](#sub-agent-orchestration)
- [The observer node](#the-observer-node)
- [Delivery review: find-fault nodes](#delivery-review-find-fault-nodes)
- [Artifact communication](#artifact-communication)
- [`build_agent_graph` parameters](#build_agent_graph-parameters)
- [Runtime context](#runtime-context)
- [Inside / outside the framework](#inside--outside-the-framework)
- [Example: a coding agent](#example-a-coding-agent)
- [Project layout](#project-layout)
- [Status](#status)

---

## Why graphloom

The hard part of building a real agent was never "call the LLM once" — it's the **chores around the loop**: remembering progress across turns, coping when context overflows the window, resuming an interrupted task, scheduling parallel sub-tasks, reusing best-practice workflows.

graphloom distills these — the things every serious agent ends up rewriting — into one reusable loop. You bring the LLM, the system prompt and the tools; it weaves the rest. It is bound to **no** transport protocol, **no** database, **no** frontend; those are dependency-injected.

## Highlights

- **Generic ReAct loop** — `observer → ai → tool → history → compaction`, exits on completion. Assembled in one call.
- **Short-term memory** — every step records a three-part chain of thought (`last_step_review / working_notes / next_action`) that accumulates into a replayable step stream across the whole session.
- **Context compaction** — as the estimated token count nears the window limit, early steps are **losslessly folded** into one archival summary (concrete facts kept verbatim: IDs, URLs, constraints, errors, decisions). Long tasks never blow the context.
- **Resumability** — inject a checkpointer and get persistence; on user pause the graph suspends at the nearest checkpoint and `ainvoke(None)` resumes in place with zero state loss.
- **Sub-agent orchestration** — dispatch grouped sub-tasks (sequential across groups, parallel within a group); artifact manifests roll forward between parent and children, and the parent checkpointer is threaded through automatically.
- **Progressive skill loading** — a Claude Code-style skill mechanism: the agent first sees only a menu of skills, and reads the full workflow on demand when a task needs it, keeping the system prompt small.
- **The observer node** — an optional entry node that runs before `ai` every turn, injecting the latest external state into that turn only. It never enters the step stream or memory — for feeding live signals (environment changes, side-channel user guidance) without polluting history.
- **Artifacts + three-stage review** — builtin artifact tools handle output and completion; at completion you can chain `custom_find_fault` (your own review) and `find_fault` (builtin QA) — either can send the work back for rework, and only approved work is delivered.
- **Artifact communication** — three artifact manifests flow between the agent, sub-agents and the outside world with distinct merge semantics — a structured delivery and hand-off channel.
- **Transport-agnostic, fully injected** — HITL, observability, persistence and the LLM provider are all externalized. Zero host coupling; depends only on LangGraph / LangChain.

## Install

```bash
pip install graphloom
```

From source:

```bash
pip install -e ".[dev]"        # includes pytest / pytest-asyncio / ruff
```

Runtime deps: `langchain-core`, `langgraph`, `langchain-openai`, `pydantic`, `tenacity`, `tiktoken`.

## Quick start

Give it an LLM, a system prompt and a set of tools; `build_agent_graph` returns a compiled LangGraph you `ainvoke`:

```python
import asyncio
from langchain_openai import ChatOpenAI
from graphloom import build_agent_graph, build_initial_agent_state

graph = build_agent_graph(
    custom_system_prompt="You are a helpful agent.",
    tools=[...],                              # your tools
    llm=ChatOpenAI(model="gpt-4o-mini"),
    allow_direct_reply=True,                  # let a plain-text reply finish the run
)

state = build_initial_agent_state(input_query="Summarize this repo", session_id="s1")
result = asyncio.run(graph.ainvoke(state))
print(result["final_reply"])
```

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

graphloom's memory isn't a bolt-on vector store — it's the **loop state itself**:

- **A three-part chain of thought per step** — each step produces `last_step_review` (did the last action work), `working_notes` (progress and key facts), and `next_action` (what to do next). This forces the model to reflect and to keep notes explicitly — the core defense against drift and forgetting.
- **The step stream is the memory** — steps accumulate into `past_steps`, persisted with the checkpointer, present across turns and sessions. Rendered back into the prompt as `<agent_history>`, so the agent always sees the path it has walked.
- **Compaction, not truncation** — nearing the limit, the `compaction` node hands the earliest steps to the LLM to fold into a lossless archive summary — numbers, IDs, URLs, user constraints, errors and their fixes kept verbatim, only redundant narration compressed. Recent steps stay intact. Long tasks keep running instead of dropping old history.

## Skills: progressive disclosure

A skill is a directory containing a `SKILL.md` whose front-matter declares a `name` and `description`. The agent initially sees only the **name + description + location** of each skill; it reads the full workflow (and any referenced scripts/resources) on demand via `read_artifact` when a task actually needs it. This gives the agent a menu of deep, reusable workflows without bloating the system prompt.

```python
graph = build_agent_graph(
    custom_system_prompt=PROMPT,
    tools=[...],
    llm=llm,
    available_skills=["pdf_extraction", "sql_report"],   # whitelist
    skills_dirs=["/path/to/builtin-skills", "/path/to/user-skills"],                   # your library; not hard-coded
)
```

Where skill libraries live is up to you—inject one or more roots through `skills_dirs` in priority order. A later root overrides an earlier skill with the same name; the framework knows nothing about the skill contents.

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

## Example: a coding agent

[`examples/coding_agent/`](examples/coding_agent/) builds a Claude Code / Codex-style coding agent in ~60 lines: `read_file`, `write_file`, `run_command` wired into `build_agent_graph`, reading an OpenAI-compatible gateway from `.env`.

```bash
pip install -e ".[dev]" python-dotenv
cp .env.example .env        # fill in BASE_URL / OPENAI_API_KEY / MODEL
python -m examples.coding_agent.agent "create fizzbuzz.py and run it"
```

The agent reads/writes files and runs commands in a sandboxed workspace, verifies the result, then replies directly. `run_command` executes arbitrary shell — only run it against a workspace you trust.

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

## License

MIT
