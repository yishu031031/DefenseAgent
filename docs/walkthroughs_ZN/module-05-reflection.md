# 模块 5 代码导览 —— 反思

> 配套[设计规范](../superpowers/specs_ZN/2026-04-22-module-05-reflection-design.md)。逐行解释代码，并追踪 `scripts/reflection_demo.py` 的执行流程。

---

## 核心类：`Reflector`

从这里开始。整个模块就是一个文件里的一个类：

```python
from DefenseAgent.reflection import Reflector

reflector = Reflector(memory, llm, num_insights=3, reflection_importance=8.0)

# Rate how poignant an observation is (Park §3.2.1)
score = await reflector.score_importance("Got stuck on problem 3 for an hour.")

# Synthesize higher-level insights (Park §3.2.2)
insights = await reflector.maybe_reflect()  # no-op if below threshold
forced   = await reflector.reflect_now()        # force it, returns records
```

`Reflector` 从 `Memory` 中读取，并将 `kind="reflection"` 记录写回。它还暴露了一个基于 LLM 的重要性评分器，使调用方可以写 `await memory.remember(content, importance=await reflector.score_importance(content))`。

这是**第一个同时接触模块 1（LLM）和模块 4（Memory）的模块**。它构成了反馈回路：记忆进行存储，反思器在其之上思考，思考结果再回到记忆中。

---

## 1. 本模块解决的问题

模块 4 是一个被动的记忆层。它记录与检索，但从不*思考*。一个只记录观察的智能体，使用一周后，将积累一份冗长而平铺的事件列表，毫无模式感——"我喝了咖啡"、"我又喝了咖啡"、"我和 Chloe 喝了咖啡"、"我一个人喝了咖啡"。

Park 等人 2023 年的洞察是：**当智能体周期性地对近期经验进行反思并将这些反思写回时，记忆质量会产生复利效应**。反思具有更高的重要性，它们会作为未来决策的上下文被召回，并且让智能体注意到任何单一观察都无法揭示的自身特征。

**模块 5 为框架提供：**

1. **基于 LLM 的重要性评分** —— 在 1–10 的量表上评估一条观察有多深刻。调用方在调用 `memory.remember(importance=score)` 时使用该分值。
2. **反思综合** —— 读取最近的非反思记忆，提示 LLM 产出 `N` 条高层洞察，将每条作为新的 `kind="reflection"` 记录存储，重要性可配置。
3. **基于计数的触发逻辑** —— `maybe_reflect()` 在未反思的观察数达到 `profile.cognitive.reflection_threshold` 之前都是空操作。

---

## 2. 目录地图

```
DefenseAgent/reflection/
├── __init__.py              # CORE CLASS header + Reflector re-export
└── reflection.py            # everything: Reflector + prompts + parsers (~180 lines)

tests/DefenseAgent/reflection/
├── __init__.py
└── test_reflection.py       # 33 tests, 5 behavior groups

scripts/
└── reflection_demo.py       # Maya observes 6 things, scores them, reflects, retrieves
```

一个文件。用户的指示是"力求简洁：争取做到单一核心类文件，仅当需要实现额外功能且代码量较大时才新建文件"——大约 180 行，整个模块完全容得下。

---

## 3. 一个反思周期的解剖

```
                    stream of observations
                           │
                           ▼
               ┌───────────────────────────┐
               │  reflector.check_and_     │
               │  reflect()                │
               └───────────────────────────┘
                           │
          unreflected_count < threshold?
               ┌───────────┴───────────┐
           yes │                       │ no
               ▼                       ▼
          return []             reflect_now():
                                ┌──────────────────────────┐
                                │ 1. gather recent non-    │
                                │    reflection records    │
                                │ 2. format prompt         │
                                │ 3. call LLM              │
                                │ 4. parse N insights      │
                                │ 5. memory.remember(      │
                                │      insight,            │
                                │      kind="reflection",  │
                                │      importance=8.0)     │
                                │ 6. advance cutoff        │
                                └──────────────────────────┘
                                           │
                                           ▼
                                    new memories in stream,
                                    retrievable like any other kind
```

`Reflector` 构造函数上的两个旋钮控制行为：

- `num_insights` —— 每个周期向 LLM 请求多少条反思。默认 3（Park 论文的取值）。
- `reflection_importance` —— 分配给每条新反思记录的重要性值。默认 8.0（按设计，反思的重要性高于原始观察）。

---

## 4. 代码逐段解读：`reflection.py`

### 4.1 Prompt 模板

两个模块级常量，通过从类中拆出来以保持可读：

```python
_IMPORTANCE_PROMPT = """\
On a scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth,
making the bed) and 10 is extremely poignant (e.g., a breakup, a college
acceptance), rate the likely poignancy of the following memory.

Memory: {content}

Respond with a single integer between 1 and 10. No explanation."""
```

这几乎是 Park 论文的原文，我们补了显式的 "integer between 1 and 10, no explanation" 后缀，以便让解析变得轻而易举。

```python
_REFLECTION_PROMPT = """\
Recent memories (chronological):
{memory_list}

Given the memories above, produce exactly {n} high-level insights about
patterns, lessons, or deeper observations. Each insight should be a
single clear sentence. Return one per line. No numbering, no bullets,
no empty lines."""
```

单步式（而不是 Park 论文的先提问后回答的两步）。更简单，一次 API 调用代替两次。如果在真实使用中反思质量受影响，我们可以之后升级为两步式。

### 4.2 解析器 —— 刻意宽容

两个解析器都对 LLM 的输出采取防御性处理。LLM 并不总是严格按指令执行。

```python
_INT_RE = re.compile(r"\d+")
_BULLET_PREFIX_RE = re.compile(r"^\s*(?:\d+[\.)]\s*|[-*•]\s*)")


def _parse_importance_response(text: str) -> float:
    m = _INT_RE.search(text or "")
    if m is None:
        return 5.0                  # default on unparseable
    value = int(m.group())
    return float(max(1, min(10, value)))    # clip to [1, 10]
```

**三点需要注意：**
- 在响应中任意位置找到*第一个*整数（`"I'd rate this an 8."` → 8）。
- 裁剪到 `[1, 10]` —— 如果模型返回 `42`，它会变成 `10`，而非违反值域。
- 解析失败时返回 `5.0`（中间值）—— 永不抛出。在处理数百条内容的记录管线中，一次失败的重要性评分不应让整个管线崩溃。

```python
def _parse_reflection_response(text: str, n: int) -> list[str]:
    if not text:
        return []
    insights = []
    for raw_line in text.splitlines():
        line = _BULLET_PREFIX_RE.sub("", raw_line).strip()
        if line:
            insights.append(line)
    return insights[:n]
```

处理 LLM 在被要求返回纯行时仍会产生的三种常见格式：
- `"1. First insight.\n2. Second insight."` —— 带编号
- `"- alpha\n* beta\n• gamma"` —— 带项目符号（三种符号风格）
- `"Insight one.\n\nInsight two."` —— 干净的行之间夹着意外空行

都被归约为 `["clean text 1", "clean text 2", ...]`。取前 `n` 条，其余丢弃。

### 4.3 `Reflector` —— 核心类

#### 构造函数

```python
def __init__(
    self,
    memory: Memory,
    llm: LLM,
    *,
    num_insights: int = 3,
    reflection_importance: float = 8.0,
    clock: Callable[[], datetime] | None = None,
) -> None:
    self.memory = memory
    self.llm = llm
    self._num_insights = num_insights
    self._reflection_importance = reflection_importance
    self._clock = clock or _default_clock
    self._last_reflection_time: datetime | None = None
```

`Reflector` 拥有的唯一状态是 `_last_reflection_time`。其余一切都是对 Memory 或 LLM 的透传。**自带 Memory、自带 LLM** —— Reflector 不关心二者是如何构建的。

#### `score_importance()` —— Park §3.2.1

```python
async def score_importance(self, content: str) -> float:
    resp = await self.llm.chat(
        [Message(role="user", content=_IMPORTANCE_PROMPT.format(content=content))],
        temperature=0.0,        # deterministic — we want the model's best guess
        max_tokens=16,          # one integer doesn't need more
    )
    return _parse_importance_response(resp.content)
```

**为何使用 `temperature=0.0`：** 这里一致性比创造性更重要。如果同一内容被评分两次，我们希望得到相同答案。

**为何使用 `max_tokens=16`：** 提示词要求的是一个整数。16 个 token 已经很宽裕——如果模型要偏离去解释，它会浪费 token 而无增益。把预算卡紧。

#### `unreflected_count` —— 触发器的状态

```python
@property
def unreflected_count(self) -> int:
    return len(self._recent_unreflected())

def _recent_unreflected(self) -> list[MemoryRecord]:
    cutoff = self._last_reflection_time
    return [
        r for r in self.memory.stream.get_all()
        if r.kind != "reflection"
        and (cutoff is None or r.timestamp > cutoff)
    ]
```

两层过滤：

1. **排除反思。** 防止无限递归——存入一条反思不应将计数器抬高到足以触发另一次反思。
2. **只计算截止时间之后的记录。** `_last_reflection_time` 初始为 `None`（尚未反思），因此一切都计入；一次反思之后，截止时间推进到"现在"；只有时间戳严格更晚的记录才计入。

#### `maybe_reflect()` —— 软触发

```python
async def maybe_reflect(self) -> list[MemoryRecord]:
    threshold = self.memory.profile.cognitive.reflection_threshold
    if self.unreflected_count < threshold:
        return []
    return await self.reflect_now()
```

三行。从档案中读取阈值（`CognitiveConfig.reflection_threshold`，默认 5）。没事可做时静默返回 `[]` —— 未来的 `Agent` 类可以在每次 `observe()` 后调用它而无需顾虑。

#### `reflect_now()` —— Park §3.2.2

```python
async def reflect_now(self) -> list[MemoryRecord]:
    recent = self._recent_unreflected()
    if not recent:
        return []

    prompt = _REFLECTION_PROMPT.format(
        memory_list=self._format_memories_for_prompt(recent),
        n=self._num_insights,
    )
    resp = await self.llm.chat(
        [Message(role="user", content=prompt)],
        temperature=0.5,        # some variability is good for synthesis
        max_tokens=512,
    )
    insights = _parse_reflection_response(resp.content, self._num_insights)

    stored: list[MemoryRecord] = []
    for insight in insights:
        record = await self.memory.remember(
            insight,
            kind="reflection",
            importance=self._reflection_importance,
        )
        stored.append(record)

    self._last_reflection_time = self._clock()     # advance cutoff
    return stored
```

**两处微妙的设计点：**

1. **`self._last_reflection_time` 是在 `remember()` 调用之后推进的，而不是之前。** 新的反思记录在截止时间推进之前获取时间戳，这意味着截止时间的更新会正确地把它们排除在**下一个**周期之外（它们的时间戳将 `<= cutoff`，而过滤器要求 `> cutoff`）。

2. **即使输出为空，截止时间也会推进。** 如果 LLM 返回了无效内容，解析产出 0 条洞察，`stored` 为 `[]`，但 `_last_reflection_time` 仍然推进。否则，每一次后续的 `maybe_reflect()` 调用都会重新尝试同一批次、重新调用 LLM、重新得到无效结果，永无止境。静默失败可以接受；重试风暴不可接受。

#### `_format_memories_for_prompt()`

```python
@staticmethod
def _format_memories_for_prompt(records: list[MemoryRecord]) -> str:
    chronological = sorted(records, key=lambda r: r.timestamp)
    return "\n".join(
        f"- [{r.kind}, imp={r.importance:.1f}] {r.content}"
        for r in chronological
    )
```

按时间顺序排列，让 LLM 看到时间流动。每条记忆都标注 `kind` 和 `importance`，让模型获得超越原始文本的上下文。

---

## 5. 温度 + token 预算 —— 两次调用为何不同

| Call | Temperature | Max tokens | Rationale |
|---|---|---|---|
| `score_importance` | 0.0 | 16 | 确定性整数；不需要创造力 |
| `reflect_now` (synthesis) | 0.5 | 512 | 一些变化有益；洞察需要发挥空间 |

反思调用使用**中等温度，而非零** —— 在相似观察上的不同运行应当能浮现不同的模式。但也不用高温度：我们要的是忠实的反思，而不是创意写作。

---

## 6. 执行流程：`scripts/reflection_demo.py`

Maya 的一天：

```
$ python scripts/reflection_demo.py

┌─ main() (async)
│
├─ Step 1: AgentProfile.from_yaml("agents/maya_rodriguez/profile.yaml")
│          profile.cognitive.reflection_threshold = 4
│
├─ Step 2: AgentLogger.from_profile(...)   # records every step
│
├─ Step 3: LLM.from_env()                  # real DeepSeek adapter
│          Memory.from_env(profile)        # real Qwen embedder
│
├─ Step 4: Reflector(memory, llm, num_insights=3, reflection_importance=8.5)
│
├─ Step 5: for each of 6 observations:
│            score     = await reflector.score_importance(obs)   ← LLM call
│            memory.remember(obs, kind="observation", importance=score,
│                            timestamp=back-dated)
│            logger.info("memory.observed", importance=score, ...)
│
│          Example output:
│            imp= 6.0  (−9h)  Attended the 9 AM data structures lecture...
│            imp= 3.0  (−10h) Grabbed coffee with my roommate before class.
│            imp= 7.0  (−4h)  Spent two hours in the library ...
│            imp= 8.0  (−2h)  Got stuck on problem 3 for an hour...
│            imp= 7.0  (−1h)  Met the study group at 5 PM and walked them through problem 3.
│            imp= 9.0  (−0h)  Realized I learn faster when I struggle alone first.
│
├─ Step 6: unreflected_count = 6 >= threshold (4), so:
│          new = await reflector.reflect_now()    ← one more LLM call
│
│          Prints each reflection:
│            • Maya learns algorithms deepest by first struggling independently
│              before seeking help from peers or TAs.
│            • The library environment, for Maya, operates as a productivity
│              ritual that pairs with fellow students into study accountability.
│            • Teaching peers (problem 3 walkthrough) consolidates Maya's own
│              understanding — she benefits twice from one struggle.
│
├─ Step 7: query = "what patterns are emerging in how Maya studies?"
│          results = await memory.recall(query, top_k=5)
│          Prints results with a ★ marker on reflection-kind hits.
│          Reflections now surface at the TOP of retrieval because:
│            • importance 8.5 → importance_score 0.85
│            • recency ~ 1.0 (just-added)
│            • relevance high (semantically aligned with "patterns" query)
│
└─ Report log file line count.
```

### 哪些地方可能出错、在哪里显现

| Situation | Surface |
|---|---|
| `EMBEDDING_API_KEY` 为空 | 脚本以退出码 2 退出并给出明确提示 |
| `score_importance` 期间 DeepSeek 限流 | `LLMProviderError` 向上传播；脚本以 1 退出 |
| DeepSeek 返回格式错误的整数 | `_parse_importance_response` 静默返回 5.0 |
| DeepSeek 返回少于 3 条洞察 | `_parse_reflection_response` 按实际返回；`reflect_now` 只存储那么多；截止时间仍然推进 |
| DeepSeek 返回零条洞察 | `reflect_now` 返回 `[]`；截止时间推进；没有无限重试 |

---

## 7. 测试覆盖地图

| Section | Tests | Covers |
|---|---|---|
| `_parse_importance_response` | 7 | 纯整数、句中整数、上下界裁剪、不可解析、空、返回 float |
| `_parse_reflection_response` | 7 | 干净行、带编号前缀、带项目符号前缀（3 种）、空行丢弃、`n` 上限、空输入、纯空白 |
| `score_importance` | 4 | LLM 集成、无效输入时用默认值、`LLMProviderError` 传播、使用 temperature=0 |
| `unreflected_count` | 4 | 空流、忽略反思、反思后重置、遵循时间戳截止 |
| `reflect_now` | 6 | 解析并存储 N 条洞察、容忍项目符号、无近期记录时返回空、空响应时仍推进截止时间、不会重复计数自身反思、尊重配置的重要性 |
| `maybe_reflect` | 4 | 低于阈值空操作、达到阈值触发、高于阈值触发、直到有新观察才不再空操作 |
| Integration | 1 | 反思可通过 `memory.recall()` 与观察一起被检索到 |

**合计 33 项测试。** 全部离线。桩 `_StubLLMAdapter` 从队列中返回预设响应；桩 `_StubEmbedder` 按内容分配确定性向量。

---

## 8. 值得留意的细节

- **触发状态只有一个 `datetime`。** 没有计数器，没有标志位，没有"脏"标记。只有 `_last_reflection_time`。未反思计数按需从 `stream.get_all()` 计算——始终正确，没有过时状态。

- **反思的存储与其他记忆完全一致。** `kind="reflection"`、`importance=8.5`，通过同一个 `memory.remember()`。这意味着它们会被嵌入、去重检查、BM25 索引，并由同一条管线进行检索排序，与观察一视同仁。检索器对它们的处理完全相同（与观察使用相同的时近衰减规则——不同于事实/偏好）。

- **没有新的错误类。** 在生产中，反思唯一可能"失败"的方式是 `LLMProviderError` 从 LLM 调用传播出来。解析失败是静默且可恢复的。增加 `ReflectionError` 层次结构只会是多余的抽象。

- **空响应时仍推进截止时间的规则可防止重试风暴。** 这是唯一不那么直观的不变量：产出零条洞察的反思*依然*会让时钟前进。若无此规则，每次后续的 `maybe_reflect()` 都会发现同一批未反思记录、调用 LLM、得到无效结果，并不断重复——在生产中就是一台账单机器。测试 `test_reflect_now_advances_cutoff_even_on_empty_response` 锁定了这一点。

- **自带 Memory、自带 LLM。** `Reflector.__init__` 接受 `memory: Memory` 和 `llm: LLM`，对二者的构造方式不作任何假设。这让该类既易于测试，也易于组合——未来的 `Agent` 类只需构建一个并注入。

- **单文件，180 行。** 这是有意为之的范围控制——用户说"争取做到单一核心类文件"。反思模块所持有的一切都放得下。日后若增加两步式先问后答提示、交叉编码器重排序，或后台调度，它们会在同一个文件里待到装不下为止。
