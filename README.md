<div align="center">

# graphloom

**一个循环，织进你所有的 agent。**

一个构建在 [LangGraph](https://github.com/langchain-ai/langgraph) 之上的通用 agent 循环框架：`build_agent_graph` 把标准 ReAct 循环装配成图，llm、工具、检查点、运行时上下文全部依赖注入，transport 与业务无关。

![Python](https://img.shields.io/badge/python-3.10+-3776AB)
![Built on](https://img.shields.io/badge/built%20on-LangGraph-1C3C3C)
![License](https://img.shields.io/badge/license-MIT-success)
![Status](https://img.shields.io/badge/status-alpha-orange)

</div>

---

## 目录

- [是什么](#是什么)
- [设计原则](#设计原则)
- [安装](#安装)
- [快速上手](#快速上手)
- [核心模型：循环如何流转](#核心模型循环如何流转)
- [`build_agent_graph` 参数](#build_agent_graph-参数)
- [运行时上下文](#运行时上下文)
- [内置组件](#内置组件)
- [框架内 / 框架外](#框架内--框架外)
- [示例：编码 agent](#示例编码-agent)
- [项目布局](#项目布局)
- [项目状态](#项目状态)

---

## 是什么

graphloom 把"一个 agent 反复调用 LLM、执行工具、积累历史直到收尾"这件每个 agent 都要做的事，抽成一个可复用的图。你带来 LLM、系统提示词和一组工具，graphloom 织好循环：

```
observer? → ai → tool → route → history → compaction → ai   (+ finish)
```

它**不认识** HITL、不认识你的 websocket 协议、不认识你的业务数据库。这些都是你通过 `tools=[...]` 和 `runtime_context` 传进来的东西——框架只负责循环本身。

## 设计原则

- **循环即框架** —— 结构性节点（ai / tool / history / compaction / finish）构成循环机制，其余一切都注入。
- **transport 无关** —— 可观测性走 LangGraph 标准 `callbacks`，框架不定义任何自有事件、不碰任何 wire 协议。
- **工具即扩展** —— HITL、子 agent 派发、业务能力，全是你传入的工具；框架对它们一无所知。
- **单向依赖** —— 框架只依赖 LangGraph / LangChain，绝不反向抓取宿主。零 `import` 指向你的应用。
- **状态自带压缩** —— 上下文逼近窗口上限时，历史步骤自动折叠成一条无损归档摘要。

## 安装

```bash
pip install graphloom
```

从源码开发：

```bash
pip install -e ".[dev]"        # 含 pytest / pytest-asyncio / ruff
```

运行时依赖：`langchain-core`、`langgraph`、`langchain-openai`、`pydantic`、`tenacity`、`tiktoken`。

## 快速上手

给一个 LLM、一段系统提示词、一组工具，`build_agent_graph` 就还你一张编译好的 LangGraph，`ainvoke` 即跑：

```python
import asyncio
from langchain_openai import ChatOpenAI
from graphloom import build_agent_graph, build_initial_agent_state

graph = build_agent_graph(
    custom_system_prompt="You are a helpful agent.",
    tools=[...],                              # 你的工具
    llm=ChatOpenAI(model="gpt-4o-mini"),
    allow_direct_reply=True,                  # 允许纯文本回复直接收尾
)

state = build_initial_agent_state(input_query="总结这个仓库", session_id="s1")
result = asyncio.run(graph.ainvoke(state))
print(result["final_reply"])
```

带检查点持久化 + 运行时上下文：

```python
graph = build_agent_graph(
    custom_system_prompt=PROMPT,
    tools=[...],
    llm=llm,
    checkpointer=checkpointer,                # 任意 LangGraph checkpointer
)

result = await graph.ainvoke(
    build_initial_agent_state(input_query=task, session_id="s1"),
    config={
        "configurable": {
            "thread_id": "s1",
            "runtime_context": {              # 框架透传给工具与子图，自身不解读
                "artifact_base_dir": "/workspace",
                "callbacks": [my_stream_handler],
            },
        },
    },
)
```

## 核心模型：循环如何流转

```
              ┌─────────────────────── 未收尾，继续 ───────────────────────┐
              │                                                            │
  observer? → ai → tool ──route──→ history → compaction ────────────────→ ai
                     │
                     └──end_tag=True──→ (find_fault?) ──→ finish ──→ END
```

- **ai** 调用 LLM，流式合并出响应，记录一个 pending step。
- **tool** 执行 LLM 请求的工具调用；若无工具调用且 `allow_direct_reply=True`，则以纯文本回复收尾。
- **route** 看 `end_tag`：为真则走向收尾（可选先经审阅），否则回到 history 继续。
- **history** 把工具结果折进一条完成态 step。
- **compaction** 当估算 token 超过阈值时，把早期步骤无损折叠成一条摘要。
- **finish** 收束交付物，标记 `agent_status=done`。

**收尾契约**：任何工具把 `end_tag=True` 写进返回即可结束循环。内置的 `deliver_artifact` 是规范做法；纯对话型 agent 用 `allow_direct_reply=True` 走直接回复。

## `build_agent_graph` 参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `custom_system_prompt` | `str` | 必填 | agent 的系统提示词；与框架通用提示词拼接。 |
| `tools` | `list` | 必填 | 你的 LangChain 工具。内置 artifact 工具会自动注入（重名以你的为准）。 |
| `llm` | `BaseChatModel` | 必填 | 任意 LangChain 聊天模型；ai 与 compaction 节点共用。 |
| `find_fault` | `str \| callable` | `None` | 传字符串则装配一个用该提示词做交付物自审的审阅节点；传节点则直接用。 |
| `custom_find_fault` | `callable` | `None` | 你自己的审阅节点，先于 `find_fault` 运行。 |
| `observer` | `callable` | `None` | 每轮 ai 之前运行的节点（注入外部观测/引导）。 |
| `subagents` | `list[SubAgentSpec]` | `None` | 配置后自动注入 `dispatch_subagents` 工具做多 agent 编排。 |
| `checkpointer` | LangGraph saver | `None` | 状态持久化；同时透传给子图。 |
| `tool_filter` | `callable` | `None` | `(state, config) -> 隐藏的工具名集合`，按轮动态裁剪工具。 |
| `allow_direct_reply` | `bool` | `False` | 允许 LLM 不调用工具、直接以文本回复收尾。 |

## 运行时上下文

宿主通过 `config["configurable"]["runtime_context"]` 注入运行期依赖。框架**原样透传**给工具与子图，自身不解读其中任何键。常见键：

| 键 | 用途 |
|---|---|
| `artifact_base_dir` | artifact 工具的工作区根目录（也可用 `GRAPHLOOM_ARTIFACT_BASE_DIR` 环境变量）。 |
| `callbacks` | 传给子图的 LangGraph 回调（token 流式等观测）。 |
| `cancel_event` | `asyncio.Event`；置位后在最近检查点抛出 `GraphInterrupt` 暂停。 |
| `user_id` | 会话归属标识，供工具使用。 |

## 内置组件

| 组件 | 角色 | 是否默认启用 |
|---|---|---|
| `write / read / patch / deliver_artifact` | agent 的产出与收尾原语（沙箱在工作区内） | 是，自动注入 |
| `dispatch_subagents` | 分组并行的多 agent 编排；透传父图 checkpointer | 仅当传入 `subagents` |
| `find_fault` | 对交付物做结构化自审，不合格打回重做 | 仅当传入 `find_fault` |
| 上下文压缩 | token 超限时无损折叠历史 | 是，始终在循环内 |

## 框架内 / 框架外

| 框架内（graphloom 负责） | 框架外（你负责） |
|---|---|
| 循环与结构性节点 | 工具——含 HITL、澄清、任何业务工具 |
| `AgentState` 与 reducers | transport / wire——ws 事件码、流式哨兵 |
| 内置 artifact 工具、可选 dispatch / find_fault | 可观测性——走 LangGraph 标准 `callbacks` |
| 上下文压缩 | 持久化——业务库、记忆库（框架只认注入的 checkpointer） |
| 注入接缝：llm / checkpointer / tools / runtime_context | LLM 供应商——注入任意 `BaseChatModel` |

## 示例：编码 agent

[`examples/coding_agent/`](examples/coding_agent/) 用约 60 行搭了一个 Claude Code / Codex 风格的编码 agent：`read_file`、`write_file`、`run_command` 三个工具接进 `build_agent_graph`，从 `.env` 读一个 OpenAI 兼容网关。

```bash
pip install -e ".[dev]" python-dotenv
cp .env.example .env        # 填入 BASE_URL / OPENAI_API_KEY / MODEL
python -m examples.coding_agent.agent "创建 fizzbuzz.py 并运行它"
```

agent 在沙箱工作区里读写文件、执行命令，验证结果后直接回复。`run_command` 会执行任意 shell——只在你信任的工作区里运行。

## 项目布局

```
src/graphloom/
  __init__.py            公开 API：build_agent_graph / AgentState / SubAgentSpec / …
  graph_builder.py       入口
  config.py              可调项（压缩阈值、并发上限等）
  model/                 state、reducers、schema、子 agent 规格
  nodes/                 ai / tool / history / compaction / finish / find_fault / interrupt_guard
  tools/                 artifact（4 个工具）、dispatch（子 agent 编排）
  prompt/                系统提示词、提示词栈、上下文渲染、消息装配
  util/                  消息工具、token 计数、会话存储
```

## 项目状态

Alpha —— 从一套生产 agent 代码中抽取而来，正在泛化循环、剥离全部宿主耦合。1.0 之前 API 可能变动。核心循环、内置工具、子 agent 派发、上下文压缩均已用真实与桩 LLM 验证。

## License

MIT
