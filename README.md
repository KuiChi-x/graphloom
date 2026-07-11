<div align="center">

**中文** · [English](README.en.md)

# graphloom

**一个循环，织进你所有的 agent。**

构建在 [LangGraph](https://github.com/langchain-ai/langgraph) 之上的通用智能体框架。一个 `build_agent_graph` 就把一个具备完整循环、短期记忆、上下文压缩、断点续跑、子 agent 编排和技能渐进加载的 ReAct agent 装配出来——LLM、工具、检查点全部依赖注入，与传输层和业务无关。

![Python](https://img.shields.io/badge/python-3.10+-3776AB)
![Built on](https://img.shields.io/badge/built%20on-LangGraph-1C3C3C)
![License](https://img.shields.io/badge/license-MIT-success)
![Status](https://img.shields.io/badge/status-alpha-orange)

</div>

---

## 目录

- [为什么是 graphloom](#为什么是-graphloom)
- [亮点](#亮点)
- [安装](#安装)
- [快速上手](#快速上手)
- [核心模型：循环如何流转](#核心模型循环如何流转)
- [智能体的记忆](#智能体的记忆)
- [技能：渐进式加载](#技能渐进式加载)
- [子 agent 编排](#子-agent-编排)
- [观察者节点](#观察者节点)
- [交付审阅：找茬节点](#交付审阅找茬节点)
- [产物通信](#产物通信)
- [`build_agent_graph` 参数](#build_agent_graph-参数)
- [运行时上下文](#运行时上下文)
- [框架内 / 框架外](#框架内--框架外)
- [示例：编码 agent](#示例编码-agent)
- [项目布局](#项目布局)
- [项目状态](#项目状态)

---

## 为什么是 graphloom

搭一个能用的 agent，真正的难点从来不是"调一次 LLM"，而是那一圈**循环里的琐事**：如何跨轮记住进度、上下文撑爆窗口怎么办、任务中断了怎么续、要拆子任务并行怎么调度、复杂工作流的最佳实践怎么复用。

graphloom 把这些"每个正经 agent 都要重写一遍"的东西沉淀成一个可复用的循环。你只带来 LLM、系统提示词和工具——其余的循环机制它替你织好。它**不绑定**任何传输协议、任何数据库、任何前端；这些通过依赖注入接入。

## 亮点

- **通用 ReAct 循环** —— `observer → ai → tool → history → compaction` 一圈到底，收尾即停。一次调用装配完成。
- **短期记忆** —— 每一步都沉淀 `last_step_review / working_notes / next_action` 三段结构化思维链，累积成可回溯的步骤流，贯穿整个会话。
- **上下文压缩(memory compaction)** —— 估算 token 逼近窗口上限时，自动把早期步骤**无损折叠**成一条归档摘要（关键事实逐字保留：ID、URL、约束、报错、决策），长任务永不爆上下文。
- **断点续跑** —— 注入 checkpointer 即获得持久化；用户主动暂停时在最近检查点挂起，`ainvoke(None)` 原地续跑，状态零丢失。
- **子 agent 编排** —— 分组并行派发子任务，组间顺序、组内并发，产物清单在父子图之间滚动合并，父图检查点自动透传。
- **技能渐进加载** —— Claude Code 式的 skill 机制：先只给 agent 一份技能清单，任务需要时才按需读入完整工作流，系统提示词保持精简。
- **观察者节点(observer)** —— 可选的入口节点，每轮在 ai 之前运行，把最新外部状态注入当轮上下文。它只影响**当前这一轮**，不写进步骤流、不进记忆——用来喂实时信号（环境变化、用户旁路引导），而不污染历史。
- **交付物系统 + 三级审阅** —— 内置 artifact 工具做产出与收尾；收尾时可串联 `custom_find_fault`（你的自定义审阅）与 `find_fault`（内置质检）两道关卡，任一不合格就打回重做，通过才交付。
- **产物通信(artifact manifest)** —— 三条产物清单以不同合并语义在 agent、子 agent 与外界之间流转，是结构化的交付与交接通道。
- **传输无关 · 全注入** —— HITL、可观测性、持久化、LLM 供应商全部外置。框架零宿主耦合，只依赖 LangGraph / LangChain。

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
from langchain_litellm import ChatLiteLLM
from graphloom import build_agent_graph, build_initial_agent_state

graph = build_agent_graph(
    custom_system_prompt="You are a helpful agent.",
    tools=[...],                              # 你的工具
    llm=ChatLiteLLM(model="gpt-4o-mini"),
    allow_direct_reply=True,                  # 允许纯文本回复直接收尾
)

state = build_initial_agent_state(input_query="总结这个仓库", session_id="s1")
result = asyncio.run(graph.ainvoke(state))
print(result["final_reply"])
```

## 核心模型：循环如何流转

```
              ┌─────────────────────── 未收尾，继续 ───────────────────────┐
              │                                                            │
  observer? → ai → tool ──route──→ history → compaction ────────────────→ ai
                     │
                     └──end_tag=True──→ (find_fault?) ──→ finish ──→ END
```

- **ai** 调用 LLM，流式合并响应，记录一个 pending step（含三段思维链）。
- **tool** 执行 LLM 请求的工具调用；若无工具调用且 `allow_direct_reply=True`，以纯文本回复收尾。
- **route** 看 `end_tag`：为真走向收尾（可选先经审阅），否则回 history 继续。
- **history** 把工具结果折进一条完成态 step。
- **compaction** 当估算 token 超阈值时，把早期步骤无损折叠成摘要。
- **finish** 收束交付物，标记 `agent_status=done`。

**收尾契约**：任何工具把 `end_tag=True` 写进返回即可结束循环。内置 `deliver_artifact` 是规范做法；纯对话型 agent 用 `allow_direct_reply=True` 走直接回复。

## 智能体的记忆

graphloom 的记忆不是外挂的向量库，而是**循环状态本身**：

- **每步三段式思维链** —— agent 每一步都产出 `last_step_review`（复盘上一步成败）、`working_notes`（记录进度与关键事实）、`next_action`（下一步动作）。这逼着模型显式反思、显式记账，是抗跑偏、抗遗忘的核心。
- **步骤流即记忆** —— 这些步骤累积成 `past_steps`，随 checkpointer 持久化，跨轮、跨会话都在。渲染回提示词时以 `<agent_history>` 呈现，agent 始终看得见自己走过的路。
- **压缩而非截断** —— 上下文逼近上限时，`compaction` 节点把最早的步骤交给 LLM 折成一条"无损归档"摘要——数字、ID、URL、用户约束、报错与其解决方式逐字保留，只压缩冗余叙述。近几步始终保持原样。于是长任务能一直跑下去，而不是简单丢掉旧历史。

## 技能：渐进式加载

技能（skill）是一个含 `SKILL.md` 的目录，front-matter 声明 `name` 与 `description`。agent 一开始只看到技能的**名字+描述+位置**清单；真正需要时才用 `read_artifact` 读入完整工作流，及其引用的脚本/参考。这样既给了 agent 一菜单可复用的深度流程，又不让系统提示词膨胀。

```python
graph = build_agent_graph(
    custom_system_prompt=PROMPT,
    tools=[...],
    llm=llm,
    available_skills=["pdf_extraction", "sql_report"],   # 白名单
    skills_dir="/path/to/your/skills",                   # 你的技能库，框架不写死
)
```

技能库放哪由你决定——`skills_dir` 注入即可，框架对内容一无所知。

## 子 agent 编排

配置 `subagents` 后，框架自动注入一个 `dispatch_subagents` 工具，让主 agent 把下一阶段拆成分组计划：**组间按 `group_id` 升序串行、组内并行**，上游产物滚动喂给下游，父图的 checkpointer 自动透传给每个子图。

```python
from graphloom import SubAgentSpec

graph = build_agent_graph(
    custom_system_prompt=PROMPT,
    tools=[...],
    llm=llm,
    subagents=[
        SubAgentSpec(agent_name="researcher", description="调研并取证", factory=make_researcher),
        SubAgentSpec(agent_name="writer",     description="撰写报告",   factory=make_writer),
    ],
)
```

## 观察者节点

`observer` 是一个可选的自定义节点。配置后它成为图的入口，且每一轮都在 `ai` 之前运行（`compaction → observer → ai`）。它的职责是把**最新的外部状态**注入当轮——环境快照、用户在旁路发来的实时引导、外部系统的信号等。

它写入的 `observer_message_parts` 字段**没有累积语义**：每轮整体覆盖，只拼进当轮发给 LLM 的消息，**不写进 `past_steps`、不进记忆**。这是刻意的——观察者反映"此刻的世界"，一旦过时就该被新观察取代，而不是沉淀成历史噪声。

```python
async def observer(state):
    snapshot = await read_live_environment(state["session_id"])
    return {"observer_message_parts": [HumanMessage(content=f"[实时状态]\n{snapshot}")]}

graph = build_agent_graph(custom_system_prompt=PROMPT, tools=[...], llm=llm, observer=observer)
```

## 交付审阅：找茬节点

agent 标记收尾（`end_tag=True`）后，交付物在真正 finish 之前可以先过审阅关卡。graphloom 支持两级、可叠加：

- **`custom_find_fault`** —— 你自己的审阅节点，先运行。想接入任何外部校验（跑测试、schema 校验、业务规则）就放这里。
- **`find_fault`** —— 内置的 LLM 自审节点。传入一段审阅提示词，它会读交付物内容、对照原始请求做结构化质检，输出是否合格 + 缺陷清单 + 返工建议。

路由：`end_tag → custom_find_fault?（有则先跑）→ find_fault?（再跑）→ finish`。审阅**不合格**则清空交付清单、带着反馈**回到 history 让 agent 重做**；合格才放行到 finish。纯文本交付（无产物）被审阅节点视为放行。

```python
graph = build_agent_graph(
    custom_system_prompt=PROMPT, tools=[...], llm=llm,
    custom_find_fault=my_test_runner_node,          # 先跑：你的校验
    find_fault="You are a strict reviewer. Verify every requirement is met.",  # 后跑：LLM 自审
)
```

## 产物通信

agent 的产出不是塞进聊天记录，而是走**结构化产物清单（artifact manifest）**。state 里有三条清单，各有不同的合并语义，共同构成交付与交接通道：

| 清单 | 合并语义 | 含义 |
|---|---|---|
| `input_artifact_manifest` | 覆盖 | 外部/上游传入的参考产物 |
| `current_delivery_manifest` | 覆盖 | 本轮 `deliver_artifact` 提交、待审阅的产物 |
| `approved_artifact_manifest` | 合并去重 | 已通过审阅、可交付的产物；子 agent 的产物也滚动并入这里 |

内置 artifact 工具（`write / read / patch / deliver`）读写这些清单，`deliver_artifact` 把 `end_tag=True` 与 `current_delivery_manifest` 一起写入触发收尾。子 agent 编排时，上游的 `approved_artifact_manifest` 会作为下游的 `input_artifact_manifest` 喂进去——产物就是 agent 之间的交接语言。

## `build_agent_graph` 参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `custom_system_prompt` | `str` | 必填 | agent 的系统提示词；与框架通用提示词拼接。 |
| `tools` | `list` | 必填 | 你的 LangChain 工具。内置 artifact 工具自动注入（重名以你的为准）。 |
| `llm` | `BaseChatModel` | 必填 | 任意 LangChain 聊天模型；ai 与 compaction 节点共用。 |
| `find_fault` | `str \| callable` | `None` | 传字符串则用该提示词装配交付物自审节点；传节点则直接用。 |
| `custom_find_fault` | `callable` | `None` | 你自己的审阅节点，先于 `find_fault` 运行。 |
| `observer` | `callable` | `None` | 每轮 ai 之前运行的节点（注入外部观测/引导）。 |
| `subagents` | `list[SubAgentSpec]` | `None` | 配置后自动注入 `dispatch_subagents` 做多 agent 编排。 |
| `checkpointer` | LangGraph saver | `None` | 状态持久化；同时透传给子图。 |
| `tool_filter` | `callable` | `None` | `(state, config) -> 隐藏工具名集合`，按轮动态裁剪工具。 |
| `allow_direct_reply` | `bool` | `False` | 允许 LLM 不调用工具、直接以文本回复收尾。 |
| `available_skills` | `list[str]` | `None` | 暴露给 agent 的技能白名单。 |
| `skills_dir` | `str` | `None` | 扫描 `*/SKILL.md` 的技能库目录。 |

## 运行时上下文

宿主通过 `config["configurable"]["runtime_context"]` 注入运行期依赖。框架**原样透传**给工具与子图，自身不解读：

| 键 | 用途 |
|---|---|
| `artifact_base_dir` | artifact 工具的工作区根目录（亦可用 `GRAPHLOOM_ARTIFACT_BASE_DIR` 环境变量）。 |
| `callbacks` | 传给子图的 LangGraph 回调（token 流式等观测）。 |
| `cancel_event` | `asyncio.Event`；置位后在最近检查点抛 `GraphInterrupt` 暂停。 |
| `user_id` | 会话归属标识，供工具使用。 |

## 框架内 / 框架外

| 框架内（graphloom 负责） | 框架外（你负责） |
|---|---|
| 循环与结构性节点 | 工具——含 HITL、澄清、任何业务工具 |
| `AgentState`、三段式思维链、上下文压缩 | 传输 / wire——ws 事件码、流式哨兵 |
| 内置 artifact 工具、可选 dispatch / find_fault | 可观测性——走 LangGraph 标准 `callbacks` |
| 技能渐进加载机制 | 技能内容——你的 `skills_dir` |
| 注入接缝：llm / checkpointer / tools / runtime_context | 持久化后端、LLM 供应商 |

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
  skills/                技能加载（SKILL.md 解析 + 渐进加载提示段）
  util/                  消息工具、token 计数、会话存储
```

## 项目状态

Alpha —— 从一套生产 agent 代码中抽取而来，正在泛化循环、剥离全部宿主耦合。1.0 之前 API 可能变动。核心循环、短期记忆、上下文压缩、子 agent 派发、技能加载均已用真实与桩 LLM 验证。

## License

MIT
