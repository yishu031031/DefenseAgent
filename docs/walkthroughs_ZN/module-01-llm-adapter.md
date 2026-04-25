# 模块 1 代码导览 —— LLM Adapter

> 本文档与[设计规范](../superpowers/specs_ZN/2026-04-22-module-01-llm-adapter-design.md)配套。设计规范记录了我们**决定了什么**以及**为什么**这么决定；本代码导览则解释代码**如何**实现这些决策，以及运行脚本时**发生了什么**。

---

## 核心类：`LLM`

从这里开始。[`DefenseAgent/llm/llm.py`](../../DefenseAgent/llm/llm.py) 中的 `LLM` 类是规范的入口：

```python
from DefenseAgent.llm import LLM, Message

llm  = LLM.from_env()                              # reads AGENT_LAB_LLM_PROVIDER from .env
resp = await llm.chat([Message(role="user", content="hi")])
```

`LLM` 封装了一个 `LLMAdapter`（下文讨论的具体提供方适配器之一），并将其以 `llm.adapter` 的形式暴露出来。本模块中的其他一切要么是：
- 门面类内部使用的机制（适配器、工厂、规范类型），
- 要么是门面类可能抛出的错误类。

下面的代码导览沿着数据流从 `.env` 走到适配器栈，以便你理解 `LLM.from_env()` 和 `llm.chat(...)` 实际做了什么。

---

## 1. 本模块解决什么问题

如果下游模块（认知循环、记忆、上下文管理器）直接调用各个厂商的 SDK，那么切换提供方就会在代码库中引发连锁反应。harness 还不得不去了解五种不同的消息格式、五种不同的工具调用结构、五种不同的响应结构。

**模块 1 为 harness 的其余部分提供了唯一的接口：**

```python
resp = await adapter.chat(messages, tools=..., system=...)
```

无论适配器路由到 Claude、OpenAI、DeepSeek、Qwen、Google 还是本地的 vLLM 服务器，这对调用方都是透明的。调用方始终发送规范的 `Message` 对象，并始终拿到规范的 `LLMResponse`。

---

## 2. 目录地图

```
DefenseAgent/llm/
├── __init__.py                     # public API re-exports
├── types.py                        # canonical data shapes (Message, ToolCall, LLMResponse, TokenUsage)
├── errors.py                       # exception hierarchy (LLMError + 3 subclasses)
├── llm_adapter.py                  # abstract base class (LLMAdapter)
├── anthropic_adapter.py            # concrete: Claude (Anthropic SDK)
├── openai_compatible_adapter.py    # concrete: OpenAI / DeepSeek / Qwen / Google / vLLM (OpenAI SDK)
└── factory.py                      # make_adapter_from_env() — reads .env, returns the right adapter

scripts/
└── smoke_chat.py                   # runnable demo, hits whichever provider .env configures
```

每个文件都不超过 200 行。除了 `anthropic`、`openai` 和 `python-dotenv` 之外，本模块没有任何外部运行时依赖。

---

## 3. 规范类型（`types.py`）

这些 dataclass 是 **harness 与 LLM 世界之间的契约**。每个提供方都会被翻译成这些结构，或从这些结构翻译回去。

### `Message`
```python
@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)  # assistant-only
    tool_call_id: str | None = None                            # tool-result-only
    name: str | None = None                                    # tool name on tool-result
```

- `role` 使用与 OpenAI 相同的四值词汇表。Anthropic 的 Claude 使用 `user/assistant/system`，但我们统一归一化到这个列表。
- `tool_calls` 会在请求了工具使用的 assistant 消息上被填充。对于纯文本的 assistant 则为空。
- `tool_call_id` + `name` 会在 `role="tool"` 消息上被填充（即我们在下一轮发回的工具结果）。

### `ToolCall`
```python
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]   # parsed dict, NOT a JSON string
```

**关键决策：**`arguments` 始终是一个 `dict`。OpenAI 返回的 `arguments` 是 JSON 字符串；适配器会在交回给调用方之前先解析好。调用方永远不需要记住"这个到底是字符串还是 dict"——它始终是 dict。

### `TokenUsage`
```python
@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
```

每次响应都会被填充。如果提供方不返回 `total_tokens`，适配器会计算为 `prompt + completion`。

### `LLMResponse`
```python
@dataclass
class LLMResponse:
    content: str                # assistant text; "" if only tool calls
    tool_calls: list[ToolCall]  # [] if no tools requested
    usage: TokenUsage
    stop_reason: str | None     # normalized vocabulary
    raw: dict[str, Any]         # original provider dict, for debugging
```

**归一化的 `stop_reason` 词汇表**（无论哪个提供方都使用相同的字符串）：
- `"end_turn"` —— 模型自行结束。
- `"tool_use"` —— 模型希望调用某个工具。
- `"max_tokens"` —— 到达 token 上限。
- `"stop_sequence"` —— 命中了用户提供的停止字符串（目前只有 Anthropic 有）。
- `"other"` —— 其他任何情况（内容过滤、遗留的 function_call 等）。

---

## 4. 抽象接口（`llm_adapter.py`）

```python
class LLMAdapter(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system: str | None = None,
    ) -> LLMResponse: ...
```

有三点值得注意：

1. **`ABC` + `@abstractmethod`。**你不能直接实例化 `LLMAdapter`——Python 会抛出 `TypeError`。你也不能在没有实现 `chat()` 的情况下继承它。这就是为什么 `test_llm_adapter.py` 中有 `test_cannot_instantiate_abstract_adapter` 和 `test_subclass_missing_chat_cannot_instantiate`。

2. **`async def`。**每次调用 LLM 提供方都要跨越网络。异步让 harness 的其余部分（事件总线、工具执行器、认知循环）可以在等待时做别的事情。

3. **`system` 作为关键字参数。**某些提供方（Anthropic）有专门的 `system` 参数；其他提供方（OpenAI 兼容）则只是把 system 消息放在列表顶部。抽象层两种方式都接受；具体适配器负责翻译。

---

## 5. `OpenAICompatibleAdapter` —— 逐步剖析

这一个处理器服务于 OpenAI、DeepSeek、Qwen、Google（通过 OpenRouter）以及 vLLM。它们都使用 OpenAI 的 `chat/completions` 线协议。

### 构造函数（客户端注入）

```python
def __init__(self, *, api_key, base_url, model, client=None):
    self._model = model
    self._client = client or AsyncOpenAI(api_key=api_key or None, base_url=base_url or None)
```

可选的 `client` 参数是一个测试接缝。生产代码不传递任何内容，会创建一个 `AsyncOpenAI` 实例。测试则传入 `MagicMock`，这样就不会发生网络调用。没有这个接缝的话，测试将需要复杂的模块级 patch。

### `chat()` —— 五个阶段

**阶段 1：System 消息冲突检查。**
```python
has_system_in_messages = any(m.role == "system" for m in messages)
if system is not None and has_system_in_messages:
    raise LLMAdapterError(...)
```
不允许歧义：调用方必须只选一种方式。

**阶段 2：将规范格式翻译为 OpenAI 线上消息。**由 `_message_to_wire(m)` 完成：
- 普通的 user/assistant/system → `{"role": ..., "content": ...}`。
- 带 `tool_calls` 的 assistant → 包含一个 `tool_calls` 数组，每个条目为 `{"id", "type": "function", "function": {"name", "arguments"}}`，其中 `arguments` 被重新序列化为 JSON 字符串（OpenAI 在线上需要字符串形式）。
- `role="tool"` → `{"role": "tool", "tool_call_id", "name", "content"}`。

**阶段 3：翻译工具 schema。**调用方传入 JSON-Schema 字典；OpenAI 要求将每个包装为 `{"type": "function", "function": {...}}`。

**阶段 4：带错误包装的 API 调用。**
```python
try:
    response = await self._client.chat.completions.create(**kwargs)
except Exception as e:
    raise LLMProviderError(
        provider="openai-compatible",
        status_code=getattr(e, "status_code", None),
        message=str(e),
    ) from e
```
`raise ... from e` 将原始异常保留为 `__cause__`，使调用方仍然可以检查它。

**阶段 5：解析响应。**由 `_parse_response(response)` 完成：
- 取出 `choices[0].message.content` → `content`（如果为 `None` 则为空字符串）。
- 对消息上的每个 `tool_call`：将 `arguments` JSON 解析回 dict，构造 `ToolCall`。
- 通过 `_FINISH_REASON_MAP` 映射 `finish_reason`：`"stop"→"end_turn"`、`"tool_calls"→"tool_use"`、`"length"→"max_tokens"`、其余一律 → `"other"`。
- 构造 `TokenUsage`，如果缺失则计算 `total_tokens`。
- 通过 `_to_dict_safe` 将提供方对象序列化给 `raw` 字段（尽力而为——先尝试 `model_dump()`，然后 `to_dict()`，最后回退到 `{"repr": repr(obj)}`）。

---

## 6. `AnthropicAdapter` —— 有哪些不同

与上面相同的五阶段结构，但线协议不同：

| 关注点 | OpenAI 兼容 | Anthropic |
|---|---|---|
| System prompt | 顶部 `role="system"` 消息 | API 调用上单独的 `system=` 关键字参数 |
| 多条 system 消息 | 不合法 | 我们的适配器用 `\n` 连接它们并作为一个 `system` 字符串传递 |
| 带工具的 assistant | `content` 字符串 + `tool_calls` 数组 | `content` 是一个 **block 列表**，混合了 `text` 与 `tool_use` block |
| 工具结果 | `role="tool"` 消息 | 带 `tool_result` 内容 block 的 `role="user"` 消息 |
| 工具 schema 字段 | `parameters` | `input_schema` |
| 响应内容 | `choices[0].message.content` 字符串 | `response.content` 是类型化 block 的列表；我们收集 `text` block（拼接）+ `tool_use` block |
| 停止原因 | `finish_reason`（stop/tool_calls/length/…） | `stop_reason`（end_turn/tool_use/max_tokens/stop_sequence）—— 基本原样透传 |

**为什么一个适配器无法同时处理两者：**内容结构从根本上不同（字符串 vs. block 列表），而且 Claude 的工具流程依赖于在 block 结构的 user 消息之间匹配 `tool_use` ID。强行让两者走一条代码路径的成本会高于保留两个约 150 行的适配器。

---

## 7. 工厂（`factory.py`）

工厂读取环境变量并返回正确的具体适配器。它从不发起网络调用——构造 SDK 客户端是惰性的。

### 环境变量结构（按设计规范）

- **选择器：**`AGENT_LAB_LLM_PROVIDER` —— `openai | anthropic | google | deepseek | qwen | vllm` 之一。
- **每提供方块：**`{PROVIDER}_API_KEY`、`{PROVIDER}_BASE_URL`、`{PROVIDER}_MODEL`。
- **覆盖层（每字段）：**`LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL_ID`。每一项只要非空，就会独立地覆盖提供方特定的值。

### `make_adapter_from_env()` 内部流程

```
1. (optional) load_dotenv(dotenv_path)                       ← reads .env into os.environ
2. _resolve_provider()                                       ← read AGENT_LAB_LLM_PROVIDER, validate
3. _resolve_fields(provider)                                 ← per-field override resolution
4. _validate(provider, api_key, base_url, model)             ← raise LLMConfigError if anything missing
5. if provider == "vllm" and not api_key: api_key = "token-not-needed"
6. if provider == "anthropic":
        return AnthropicAdapter(...)
   else:
        return OpenAICompatibleAdapter(...)
```

`_pick(override, fallback)` 是覆盖层核心的三行辅助函数：

```python
def _pick(override, fallback):
    if override:
        return override
    return fallback or ""
```

将空字符串视为"未设置"，这样 `LLM_API_KEY=""` 就能正确地回退到 provider 的配置块。

### 测试控制

测试传入 `load_env=False` 以跳过 `load_dotenv()`，否则它会重新填充测试故意清除掉的环境变量。`tests/DefenseAgent/llm/test_factory.py` 中的 `clear_llm_env` fixture 还会 monkeypatch `load_dotenv` 为 no-op，以做纵深防御。

---

## 8. 错误（`errors.py`）

```
LLMError (base)
├── LLMConfigError        — .env / configuration problem, raised by the factory
├── LLMAdapterError       — caller misused the adapter (e.g. both system sources supplied)
└── LLMProviderError      — provider API returned an error; wraps original via __cause__
```

调用方可以宽泛地捕获 `LLMError`，也可以按子类分支处理：
```python
try:
    resp = await adapter.chat(...)
except LLMConfigError:
    ... # fix env
except LLMProviderError as e:
    ... # maybe retry; inspect e.status_code
```

---

## 9. 执行流程：`scripts/smoke_chat.py`

当你运行 `python scripts/smoke_chat.py` 时：

```
┌─ main() (async)
│
├─ 1. make_adapter_from_env()
│     • python-dotenv loads .env into os.environ
│     • factory reads AGENT_LAB_LLM_PROVIDER
│     • factory resolves {PROVIDER}_API_KEY / BASE_URL / MODEL
│                (with LLM_* overrides applied per-field)
│     • factory instantiates AsyncOpenAI (or AsyncAnthropic) lazily
│     • factory returns the configured adapter
│
├─ 2. Print diagnostic info (provider, model, adapter class)
│
├─ 3. adapter.chat([Message(role="user", content="Say hello in 5 words or fewer.")])
│     │
│     ├─ OpenAICompatibleAdapter._chat
│     │   ├─ no system conflict
│     │   ├─ wire_messages = [{"role": "user", "content": "Say hello ..."}]
│     │   ├─ kwargs = {model, messages, temperature=0.2, max_tokens=64}
│     │   ├─ await self._client.chat.completions.create(**kwargs)
│     │   │    └─ HTTPS POST to https://api.deepseek.com/v1/chat/completions
│     │   └─ _parse_response → LLMResponse
│     │         • content extracted from choices[0].message.content
│     │         • finish_reason="stop" normalized to "end_turn"
│     │         • usage filled, total_tokens computed if missing
│     │
│     └─ returns LLMResponse
│
└─ 4. Print content, stop_reason, usage
```

**哪些地方会失败、在哪个阶段：**

| 错误 | 阶段 | 表现 |
|---|---|---|
| `AGENT_LAB_LLM_PROVIDER` 未设置 | 阶段 1 | `LLMConfigError` |
| 未知的 provider 值 | 阶段 1 | `LLMConfigError` |
| 缺少 model / base_url / api_key | 阶段 1 | `LLMConfigError` |
| 网络故障、鉴权错误、限流 | 阶段 3 | `LLMProviderError`（原始异常作为 `__cause__`） |

脚本在两个阶段都会捕获 `LLMError` 并返回非零退出码：配置错误返回 `2`，运行时错误返回 `1`。

---

## 10. 值得留意的点

- **与 harness 零耦合。**`llm` 模块对智能体档案、记忆、工具类或认知循环一无所知。其他模块导入它；它不导入它们中的任何一个。
- **通过客户端注入实现的测试接缝。**适配器构造函数中的 `client=fake` 意味着测试可以精确断言构造了什么请求，而不需要任何模块级 patch。在发生任何真实的网络调用之前，60/60 个测试就已经通过。
- **停止原因词汇表的集中化。**`openai_compatible_adapter.py` 中的 `_FINISH_REASON_MAP` 字典和 `anthropic_adapter.py` 中的 `_PASSTHROUGH_STOP_REASONS` 集合是 harness 其余部分唯一需要理解的地方。新增一个停止原因，在每个适配器中都只是一行的改动。
- **`_to_dict_safe` 在两个适配器中都出现。**这是有意的重复——适配器之间互不导入。如果重复扩展到三个适配器，就把它提升到 `_common.py`；在此之前，YAGNI。
