<!-- 有 logo 后，把下面的 emoji 换成 <img src="docs/logo.svg" width="120"> 之类 -->
<div align="center">

# 🧵 graphloom

**别再重写 agent 的那圈循环。一次 `build_agent_graph`，织出一个生产级 agent loop。**

三层记忆 · 强制的分步反思 · 会自我纠错的交付关卡 · 子 agent 编排 · 技能渐进加载 —— 全部依赖注入，构建在 [LangGraph](https://github.com/langchain-ai/langgraph) 之上。

[![PyPI](https://img.shields.io/pypi/v/graphloom?color=3775A9&logo=pypi&logoColor=white)](https://pypi.org/project/graphloom/)
[![Python](https://img.shields.io/pypi/pyversions/graphloom?logo=python&logoColor=white)](https://pypi.org/project/graphloom/)
[![License](https://img.shields.io/pypi/l/graphloom?color=success)](LICENSE)
![Built on LangGraph](https://img.shields.io/badge/built%20on-LangGraph-1C3C3C)
![Status](https://img.shields.io/badge/status-alpha-orange)
[![GitHub stars](https://img.shields.io/github/stars/KuiChi-x/graphloom?style=social)](https://github.com/KuiChi-x/graphloom)

[English](README.md) · [快速上手](#快速上手30-秒) · [为什么是 graphloom](#为什么是-graphloom) · [核心循环](#核心模型循环如何流转)

</div>

---

## 快速上手（30 秒）

```bash
pip install graphloom
```

模型走 [LiteLLM](https://docs.litellm.ai/docs/providers)，所以任意供应商都行。把它指向你的 key 和 endpoint——可以导出环境变量：

```bash
export OPENAI_API_KEY="sk-..."          # 你的 key
# export OPENAI_API_BASE="https://your-gateway/v1"   # 仅当你走代理/网关时才需要
```

……也可以直接传给下面代码里的 `ChatLiteLLM(...)`（`api_key=...`、`api_base=...`）。然后带上三样东西——一个 LLM、一段系统提示词、你的工具——`build_agent_graph` 就把它们织成一个完整的 agent：它会推理、调工具、跨轮记忆，还知道自己什么时候该收尾。一次 `ainvoke` 就跑起来：

```python
import asyncio
from langchain_litellm import ChatLiteLLM
from graphloom import build_agent_graph, build_initial_agent_state

# 1) 装配 —— 带上 LLM、系统提示词、你的工具
graph = build_agent_graph(
    custom_system_prompt="You are a helpful assistant.",
    tools=[],                                 # 在这里加入你的 @tool 函数
    llm=ChatLiteLLM(
        model="gpt-4o-mini",                  # 任意 LiteLLM 模型 id，如 "claude-3-5-sonnet-20241022"
        # api_key="sk-...",                    # 或在环境变量里设 OPENAI_API_KEY
        # api_base="https://your-gateway/v1",  # 仅代理 / 自建 / 非 OpenAI 端点才需要
    ),
    allow_direct_reply=True,                  # 允许纯文本回复直接收尾
)

# 2) 运行 —— 一个 ReAct 循环就此转起来
state = build_initial_agent_state(input_query="解释一下什么是 ReAct agent", session_id="s1")
result = asyncio.run(graph.ainvoke(state))

# 3) 拿结果 —— 开了 allow_direct_reply，agent 的文本回复就落在 final_reply
print(result["final_reply"])
```

配好 key 后原样即可跑。就这一个调用，你拿到的不是"一次 LLM 请求"，而是一个**完整的循环**：跨轮记忆、上下文自动压缩、断点续跑、子 agent 派发、技能按需加载——全都织好了。加上工具，它就不只是回答、而是开始行动——见下方的[编码 agent 示例](#示例约-35-行搭一个编码-agent)。

> [!NOTE]
> 模型 id 和各供应商的鉴权都遵循 [LiteLLM 的约定](https://docs.litellm.ai/docs/providers)——例如 `gpt-4o-mini` 读 `OPENAI_API_KEY`，`claude-3-5-sonnet-20241022` 读 `ANTHROPIC_API_KEY`，而本地的 `ollama/llama3` 完全不需要 key。

> [!TIP]
> 纯对话 agent 用 `allow_direct_reply=True` 直接收尾即可；要交付文件或结构化产物，就用内置的 `deliver_artifact` 工具，还能顺带接上[交付审阅](#交付审阅找茬节点)。

## 为什么是 graphloom

让 LLM 调用一次很容易；让它跨几十步不跑偏、不爆上下文、断了能续、还能拆并行子任务——才是真正的工作量。

这些**循环里的琐事**，每个正经 agent 都要重写一遍：如何跨轮记住进度、上下文撑爆窗口怎么办、任务中断了怎么续、要拆子任务并行怎么调度、复杂工作流的最佳实践怎么复用。

graphloom 把它们沉淀成一个可复用的循环。你只带来 LLM、系统提示词和工具——其余的循环机制它替你织好。它**不绑定**任何传输协议、任何数据库、任何前端；这些通过依赖注入接入。

## 凭什么不一样

大多数"agent 框架"给你一个循环，然后把记忆、上下文上限、质量管控全留给你自己搞。graphloom 存在的意义，恰恰就是这些最难啃的部分：

### 🧠 三层记忆——真正让长任务保持连贯的东西

外挂向量库只是其中一种召回。真正让长任务崩掉的，是**"此刻"到"一小时前"之间那段跨度**。graphloom 把工作记忆按新旧分层，直接放进循环状态里——最新的步骤完整保留，最旧的被总结，中间的优雅降级：

- **短期——最近步骤，完整保留。** 最近几步（`COMPACT_KEEP_RECENT_STEPS`，默认 5）原封不动地完整保留，agent 以全保真度看见自己刚走过的轨迹。单个工具结果还有 50 KB 硬上限（`_HARD_TRUNCATE_LIMIT`），一个巨型返回撑不爆当轮。
- **中期——部分截断。** 步骤老过保留窗口后，压缩时会把最臃肿的 `action_results` 按字符预算裁剪（最坏情况还有应急截断）——推理留下，原始 dump 缩水。细节淡去，线索不断。
- **长期——边执行边总结。** 比保留窗口更老的步骤，被 LLM 折成一条**无损归档**：数字、ID、URL、用户约束、报错与其修复方式**逐字保留**，只压缩冗余叙述。`compaction` 折完再估一遍 token，若仍超就收紧预算重试直到装下。永不失忆，只是越往前越密。

而整条步骤流都随 checkpointer 持久化、以 `<agent_history>` 回放——这份记忆跨轮、跨会话、跨进程重启都在。

### 🪞 反思焊死在每个动作上——agent 无法"自动驾驶"

每个内置工具的参数 schema 都**强制要求**先给出三段思维链，才能执行——`last_step_review`（上一步成了吗）、`working_notes`（目前已知什么）、`next_action`（下一步做什么）。这不是提示词里的建议，而是工具的 `StandardThoughtInput` 基类，所以模型**不先反思，就调不了工具**。这就是"会发现自己卡住了"的 agent 和"把同一个失败调用重试十遍"的 agent 之间的差别——而且内置的重复检测器会在它开始打转时，把 `[repeated N 次]` 直接标回提示词里。

### ♻️ 会自我纠错的交付关卡——agent 自己批改自己的作业

收尾不是"模型说做完了就算完"。agent 交付时，产物要过最多两道 **find-fault** 关卡：你自己的校验器（`custom_find_fault`——跑测试、校 schema）和一个内置 LLM 审阅器（`find_fault`，它会拿真实产物内容对照原始请求逐条审）。**一旦驳回，具体缺陷会被直接写回下一轮提示词**（`fatal_gaps`、`recommended_rework`），把 agent 打回去修。是一个质量闭环，而不是一句祈祷。

### 🛰️ 面向实时世界的观察者通道

可选的 `observer` 节点在每一轮之前运行，注入**最新的**外部状态——环境快照、用户旁路引导、一个行情跳动。它只拼进当轮，**绝不污染记忆**，因为"此刻的世界"应该被下一次观察取代，而不是沉淀成历史。大多数框架逼你把这类更新硬塞进消息列表来凑合。

### 循环的其余部分，一次调用全部装配

- **子 agent 编排** —— 分组派发子任务，组间顺序、组内并发；产物父→子滚动前递，父图 checkpointer 自动透传。
- **技能渐进加载** —— Claude Code 式的技能菜单：agent 一开始只看到名字+描述，需要时才读入完整工作流，系统提示词保持精简。
- **断点续跑** —— 注入 checkpointer；用户暂停时在最近检查点挂起，`ainvoke(None)` 原地续跑，状态零丢失。
- **结构化产物通道** —— 三条不同合并语义的清单，在 agent、子 agent 与外界之间搬运真正的交付物——不是从聊天记录里刮。
- **传输无关 · 全注入** —— HITL、可观测性、持久化、LLM 供应商全部外置。框架零宿主耦合，只依赖 LangGraph / LangChain。

## 安装

```bash
pip install graphloom
```

从源码开发：

```bash
git clone https://github.com/KuiChi-x/graphloom.git
cd graphloom
pip install -e ".[dev]"        # 含 pytest / pytest-asyncio / ruff
```

运行时依赖：`langchain-core`、`langgraph`、`langchain-litellm`、`pydantic`、`tenacity`、`tiktoken`。

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

graphloom 的记忆不是外挂的向量库，而是**循环状态本身**，按新旧分层：

- **短期——最近步骤，逐字保留。** 最近 `COMPACT_KEEP_RECENT_STEPS` 步（默认 5）完整原样保留。单个工具结果有 50 KB 硬上限（`_HARD_TRUNCATE_LIMIT`），一个巨型返回撑不爆当轮。
- **中期——部分截断。** 压缩一旦触发，保留窗口里最臃肿的 `action_results` 会被裁到按字段的字符预算（最坏情况还有应急截断）——推理留下，原始 dump 缩水。
- **长期——边执行边总结。** 比保留窗口更老的步骤，被 LLM 折成一条无损归档 step：数字、ID、URL、用户约束、报错与其解决方式**逐字保留**，只压缩冗余叙述。`compaction` 会重新估算 token，并以更紧的预算重试直到装下。
- **步骤流即记忆** —— 这些步骤累积成 `past_steps`，随 checkpointer 持久化，跨轮、跨会话都在。渲染回提示词时以 `<agent_history>` 呈现，agent 始终看得见自己走过的路。

### 反思是必填的 schema 字段，不是建议

每个内置工具都继承 **`StandardThoughtInput`**——一个 Pydantic 基类，带三个必填字段，LLM 每次调用都必须填：

```python
from graphloom import StandardThoughtInput  # 从包顶层导出
from pydantic import Field

class MyToolInput(StandardThoughtInput):
    # 继承必填：last_step_review / working_notes / next_action
    query: str = Field(description="要搜索什么")
```

给你自己的工具继承它，同一套强制思维链就对它们生效——`last_step_review`（复盘上一步）、`working_notes`（把关键事实带到下一步）、`next_action`（明确下一步动作）。这是框架抗跑偏、抗遗忘的核心；也是模型"先反思再动手"、而不是"出错后才反思"的原因。（编排类工具如 `dispatch_subagents` 用的是在此之上再扩展的 `PlannerThoughtInput`。）

## 技能：渐进式加载

技能（skill）是一个含 `SKILL.md` 的目录，front-matter 声明 `name` 与 `description`。agent 一开始只看到技能的**名字+描述+位置**清单；真正需要时才用 `read_artifact` 读入完整工作流，及其引用的脚本/参考。这样既给了 agent 一菜单可复用的深度流程，又不让系统提示词膨胀。

```python
graph = build_agent_graph(
    custom_system_prompt=PROMPT,
    tools=[...],
    llm=llm,
    available_skills=["pdf_extraction", "sql_report"],              # 白名单
    skills_dirs=["/path/to/builtin-skills", "/path/to/user-skills"], # 你的技能库，框架不写死
)
```

技能库放哪由你决定——通过 `skills_dirs` 按优先级注入一个或多个目录即可；同名 Skill 由后面的目录覆盖。框架对具体技能内容一无所知。

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
| `skills_dirs` | `Sequence[str]` | `None` | 按顺序扫描一个或多个技能库目录；后面的同名 Skill 覆盖前面的。 |

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
| 技能渐进加载机制 | 技能内容——你的 `skills_dirs` 目录列表 |
| 注入接缝：llm / checkpointer / tools / runtime_context | 持久化后端、LLM 供应商 |

## 示例：约 35 行搭一个编码 agent

三个工具（读、写、跑命令）接进 `build_agent_graph`，就是一个 Claude Code / Codex 风格的编码 agent：

```python
import asyncio
import subprocess
from pathlib import Path

from langchain_core.tools import tool
from langchain_litellm import ChatLiteLLM
from graphloom import build_agent_graph, build_initial_agent_state


@tool
def read_file(path: str) -> str:
    """读取文件内容。"""
    return Path(path).read_text(encoding="utf-8")


@tool
def write_file(path: str, content: str) -> str:
    """写入文件内容。"""
    Path(path).write_text(content, encoding="utf-8")
    return f"wrote {path}"


@tool
def run_command(cmd: str) -> str:
    """执行 shell 命令（只在你信任的工作区运行）。"""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


graph = build_agent_graph(
    custom_system_prompt="You are a coding agent. Read, write and run code to satisfy the request.",
    tools=[read_file, write_file, run_command],
    llm=ChatLiteLLM(model="gpt-4o-mini"),
    allow_direct_reply=True,
)

state = build_initial_agent_state(input_query="创建 fizzbuzz.py 并运行它", session_id="demo")
print(asyncio.run(graph.ainvoke(state))["final_reply"])
```

agent 会在工作区里读写文件、执行命令、验证结果，然后直接回复。鉴权方式和[快速上手](#快速上手30-秒)一样——环境变量，或在 `ChatLiteLLM` 上传 `api_key=`/`api_base=`。`run_command` 执行任意 shell——**只在你信任的工作区里运行**。

## 项目布局

```
src/graphloom/
  __init__.py            公开 API：build_agent_graph / build_initial_agent_state / AgentState / SubAgentSpec / …
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

## 贡献

欢迎 issue 与 PR。开发环境 `pip install -e ".[dev]"`，提交前跑一遍 `pytest` 和 `ruff check`。

## License

[Apache 2.0](LICENSE)
