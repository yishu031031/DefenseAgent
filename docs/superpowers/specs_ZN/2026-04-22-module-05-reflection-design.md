#  模块 5 —— 反思设计

**日期：** 2026-04-22
**状态：** 草稿，可以实现
**模块位置：** N 个模块中的第 5 个。这是第一个将模块 1（LLM）与模块 4（记忆）联合起来的模块 —— 它读取记忆，请 LLM 在其之上进行思考，再将新的记忆写回。

## 目的

模块 4 为该 harness 提供了一个被动的记忆层 —— 它会记录、检索，但从不*思考*。模块 5 补上了 Park 等人 2023 年《Generative Agents》论文中所缺的反馈回路：

1. 随着智能体不断积累观察，重要性在未被审视的情况下持续堆积。
2. 当累积的重要性足够高时，用这些记忆向 LLM 提问：*"从中能浮现出哪些高层洞察？"*
3. 将每条洞察作为一条新的 `reflection` 类别记忆存回同一条流中，其重要性高于原始观察。

反思在检索中的参与方式与其他任何类别的记录一样，因此未来的查询能够在原始事件旁边呈现综合后的洞察。

**此处还存在一个次要能力**：基于 LLM 的**重要性评分** —— 这是对 `LLM.chat()` 的一个轻量包装，请模型在 1–10 的尺度上给出一条观察的深刻度。我们在模块 4 中推迟了这一能力，因为它需要 LLM；本模块正是两个涉及记忆的 LLM 包装器应当归属的地方。

## 范围

### 在范围内
- 一个核心类：`Reflector`。
- 两个依赖 LLM 的能力：
  - `score_importance(content) -> float` —— 对单条观察以 1–10 打分。
  - `reflect_now()` / `check_and_reflect()` —— 对近期记忆综合出洞察；每条以 `kind="reflection"` 存为一条新记录。
- 触发逻辑：基于计数。当自上次反思之后新增的非反思记录数跨过 `profile.cognitive.reflection_threshold` 时，`check_and_reflect()` 会执行一次反思；否则为 no-op。
- 两次 LLM 调用的响应解析（容错，不会崩溃）。
- 完整的测试覆盖（离线，桩 LLM + 桩 embedder）。
- 一个会命中真实 LLM 的演示脚本。

### 不在范围内
- 在每次 `memory.remember()` 时自动触发 —— 未来的 `Agent` 类会调用 `check_and_reflect()`；Reflector 本身是手动的。
- 多步反思（Park 的"先生成问题，再回答它们"） —— v1 采用单步；如果质量不佳再升级。
- 后台 / 异步调度 —— 由调用方决定节奏。
- 反思的再反思回路 —— 反思不计入阈值；避免无限递归。
- 持久化 —— 与模块 4 相同（仅内存）。

## 设计

### 核心类

```python
# DefenseAgent/reflection/reflection.py

class Reflector:
    """Reads from a Memory, asks an LLM to think about it, writes back.

    Bring-your-own-Memory and bring-your-own-LLM: Reflector has no opinion
    about how either was constructed. It just needs the two front-door
    objects.
    """

    def __init__(
        self,
        memory: Memory,
        llm: LLM,
        *,
        num_insights: int = 3,
        reflection_importance: float = 8.0,
    ) -> None: ...

    # ---- LLM-based importance scoring (Park §3.2.1) ----

    async def score_importance(self, content: str) -> float:
        """Rate how poignant `content` is on a 1–10 scale (via LLM).

        Returns 5.0 (the middle) if parsing fails — never raises.
        """

    # ---- Reflection (Park §3.2.2) ----

    async def check_and_reflect(self) -> list[MemoryRecord]:
        """If the unreflected count >= threshold, reflect now. Else no-op.
        Returns the new reflection records ([] if nothing happened).
        """

    async def reflect_now(self) -> list[MemoryRecord]:
        """Force a reflection regardless of threshold."""

    @property
    def unreflected_count(self) -> int:
        """Non-reflection records added since the last reflection."""
```

### 触发器：基于计数，从共享档案中读取

`profile.cognitive.reflection_threshold`（已在模块 2 的 schema 中，默认为 `5`，注释为：*"trigger reflection after N new memories"*）就是这个旋钮。这与 schema 对外声明的一致，并让 API 保持简单 —— 不引入新的配置字段。

Reflector 只跟踪一份状态：`self._last_reflection_time: datetime | None`。`unreflected_count` 返回 `len([r for r in memory.stream.get_all() if r.timestamp > self._last_reflection_time and r.kind != "reflection"])`。在一次成功反思之后，设置 `_last_reflection_time = self._clock()`。

**为什么不用重要性累加触发（如 Park 论文所述）？** 我们的档案 schema 指定的是计数，而非累计重要性。计数更容易推理，也与文档保持一致。如果用户之后需要重要性累加，可以给观察打高分并把阈值保持在低位，或者我们将其作为可选项加入。

**为什么反思本身不计入计数？** 为了避免无限递归 —— 反思自身会以 kind=`reflection` 存为一条新记忆；如果它被计入，反思这一动作本身就会再次触发阈值。

### 重要性评分提示词

Park 风格，适配我们的 `LLM.chat()`：

```
On a scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth,
making the bed) and 10 is extremely poignant (e.g., a breakup, a college
acceptance), rate the likely poignancy of the following memory.

Memory: {content}

Respond with a single integer between 1 and 10. No explanation.
```

解析：在响应中找到第一个整数；裁剪到 `[1, 10]`；若无法解析，返回 `5.0`。从不抛异常。

### 反思提示词

```
Recent memories (chronological):
{chronological_list_of_recent_memories}

Given the memories above, produce exactly {N} high-level insights about
patterns, lessons, or deeper observations. Each insight should be a
single clear sentence. Return one per line. No numbering, no bullets,
no empty lines.
```

解析：按 `\n` 拆分，去掉每一行的空白以及前置的项目符号/序号（`-`、`*`、`1.`、`1)`），丢弃空行，取前 `N` 个非空行。若解析后不足 N 条，则返回现有的若干条；若模型什么有用的内容都没返回，则返回空列表（不会创建反思记录；但触发器状态仍会推进，这样我们就不会在每次调用时都对同一批次反复重试）。

### 提示词中"近期记忆"是什么意思

即 `memory.stream.get_all()` 中所有时间戳 `> self._last_reflection_time` 的记录。这恰是自上次反思以来新出现的观察集合 —— 也正是要被综合的范围。

每条记忆按时间顺序以 `"- [kind, imp={importance}] {content}"` 的格式渲染到提示词里。

### 集成表

| 模块 | 反思如何使用它 | 对该模块的改动 |
|---|---|---|
| 模块 1（LLM） | 使用 `LLM.chat()` 完成重要性评分和反思综合。只读。 | 无 |
| 模块 2（配置） | 通过 `memory.profile` 读取 `profile.cognitive.reflection_threshold` | 无 |
| 模块 3（ops） | 无耦合 —— 调用方可用 `logger.info()` 包装 `check_and_reflect()` | 无 |
| 模块 4（记忆） | 读取 `memory.stream.get_all()`；通过 `memory.remember(kind="reflection", …)` 写入 | 无 |

零改造。`Reflector` 严格做加法。

### 错误

本模块自己不抛出任何异常。两个隐含契约：

- 来自 `llm.chat()` 的 `LLMProviderError` 会向上传播给调用方（网络故障、限速等）。
- 格式不良的 LLM 响应（重要性的整数无法解析；反思内容为空或乱码）会被吞掉并返回合理默认值 —— 绝不让智能体崩溃。

没有 `ReflectionError` 类。如果 `LLMProviderError` 是唯一的错误，我们不需要新的层级。

### 文件布局 —— 按用户指示，一文件一职

```
DefenseAgent/reflection/
├── __init__.py           # re-exports Reflector
└── reflection.py         # everything: Reflector class, prompts, parsers (~150 lines)

tests/DefenseAgent/reflection/
├── __init__.py
└── test_reflection.py

scripts/
└── reflection_demo.py    # seed Maya with observations, force a reflection, retrieve
```

### 依赖

无新增依赖。使用 `LLM`（模块 1）、`Memory` 与 `MemoryRecord`（模块 4）、`AgentProfile`（模块 2） —— 这些在 harness 中都已存在。

## 测试策略

全程离线。桩 `LLM` 与桩 `EmbeddingAdapter` 让我们能固定响应并断言完整流程。

测试大纲：

1. **重要性评分 —— 解析变体**（6 项测试）：
   - 纯整数（`"7"`） → 7.0。
   - 句子中的整数（`"I'd rate this a 8."`） → 8.0。
   - 越界时裁剪到 [1, 10]（`"42"` → 10，`"-3"` → 1…… 实际上正则只匹配非负整数，所以 `-3` → 3；请核对）。
   - 无法解析（`"hmm"`） → 5.0（默认值）。
   - 返回的是 float，而不是 int。
   - LLM 异常向上传播（不吞掉）。

2. **未反思计数**（4 项测试）：
   - 空流 → 0。
   - 计入观察，忽略反思。
   - 反思运行之后复位。
   - 遵守时间戳截断（早于上次反思的记录不计入）。

3. **`reflect_now`**（6 项测试）：
   - 从干净的响应中恰好解析出 N 条洞察。
   - 容忍项目符号 / 编号。
   - 反思以 `kind="reflection"` 与配置的重要性存储。
   - 反思会出现在后续的 `memory.recall()` 中。
   - LLM 响应近似为空 → 返回 `[]`，但仍然推进 `_last_reflection_time`。
   - 下一轮不会把自己之前的反思重复计入。

4. **`check_and_reflect` 触发逻辑**（3 项测试）：
   - 低于阈值 → no-op，返回 `[]`，不调用 LLM。
   - 达到阈值 → 触发，返回记录。
   - 高于阈值，且在手动复位后 → 在有新记录到达之前保持 no-op。

5. **集成**（1 项测试）：端到端，使用 `AgentProfile.from_yaml(maya_rodriguez.yaml)` + 桩 LLM + 桩 embedder + 真实 `Memory`；验证反思能够通过查询被检索到。

## 执行流程（`reflect_now()` 时发生了什么）

```
reflector.reflect_now()
│
├─ 1. recent = [r for r in memory.stream.get_all()
│               if r.timestamp > self._last_reflection_time
│               and r.kind != "reflection"]
│
├─ 2. if not recent:           # nothing new to reflect on
│         return []
│
├─ 3. prompt = build_reflection_prompt(recent, self._num_insights)
│
├─ 4. resp = await self._llm.chat([Message(role="user", content=prompt)])
│
├─ 5. insights = _parse_reflection_response(resp.content, self._num_insights)
│
├─ 6. records: list[MemoryRecord] = []
│     for insight in insights:
│         r = await memory.remember(
│                 insight,
│                 kind="reflection",
│                 importance=self._reflection_importance,
│             )
│         records.append(r)
│
├─ 7. self._last_reflection_time = self._clock()
│
└─ return records
```

`check_and_reflect()` 在其之前追加一个早检查：如果 `unreflected_count < threshold`，在步骤 3 之前就返回 `[]`。

## 开放问题

无。设计决策在上文中已逐一确认：
1. **基于计数的触发** 与现有的 `profile.cognitive.reflection_threshold` schema 相匹配。
2. **单步提示词**（而非 Park 的先提问后回答两步） —— 如果质量不佳之后再升级。
3. **手动触发** —— 调用方控制节奏；Reflector 不会自动挂到 `memory.remember()` 上。
4. **一个文件**（`reflection.py`） —— 约 150 行之内能轻松容纳。
5. **不新增错误类** —— 模块 1 的 `LLMProviderError` 足矣。
