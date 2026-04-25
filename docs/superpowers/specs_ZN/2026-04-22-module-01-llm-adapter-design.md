# 模块 1 —— LLM 适配器设计

**日期：** 2026-04-22
**状态：** 已批准，可进入实施计划阶段
**模块位置：** N 个模块中的第 1 个。智能体 harness 的首个模块。认知循环与记忆模块将依赖于此。

## 目的

为 harness 提供一个稳定的、与服务商无关的 LLM 聊天接口。下游模块（认知循环、上下文管理器、记忆检索器）永远不需要知道是哪家厂商在处理请求。切换服务商应当只是配置变更，而非代码变更。

## 范围

### 范围之内（本模块）
- 抽象 `LLMAdapter` 基类，仅有一个方法：`async chat(...)`。
- 两个具体的适配器：
  - `AnthropicAdapter` —— Anthropic messages API（原生或经由代理）。
  - `OpenAICompatibleAdapter` —— OpenAI 兼容 API，覆盖 OpenAI、Google/Gemini（经代理）、Qwen、DeepSeek 和 vLLM（本地）。运行时通过 `base_url`、`api_key`、`model` 区分。
- 规范的内部类型（`Message`、`ToolCall`、`LLMResponse`、`TokenUsage`），harness 的其余部分都将使用这些类型。
- 通过 `python-dotenv` 加载 `.env`，并提供一个 `make_adapter_from_env()` 工厂，支持按提供方划分的配置块和按字段覆盖的层级。
- 使用模拟 HTTP 的单元测试 —— 测试套件中不进行真实的 API 调用。

### 范围之外（推迟到后续模块）
- `embed()` —— 目前还没有消费者；当记忆模块落地时，会将其加入抽象接口。推迟每个提供方的嵌入模型选型。
- `score_importance()` —— 它是 `chat()` 的一个薄封装；当记忆模块构建时，它将存在于该模块中。
- 流式响应。目前仅非流式；当 UI 或事件总线需要时再添加流式传输。
- 重试逻辑、速率限制、熔断器。目前原始错误直接向上传递；执行器/操作员层之后会补充策略。
- 在持久化存储中进行成本/token 计量。`LLMResponse.usage` 会返回，但尚未聚合（那是 `ops/metrics.py` 模块的职责）。

## 设计

### 提供方形态

| 提供方    | 线路协议               | 我们如何访问                                        |
|-----------|------------------------|----------------------------------------------------|
| OpenAI    | OpenAI chat/completions| `openai` SDK（base_url 默认为 `api.openai.com`，或用户自定义的代理如 OpenRouter） |
| Anthropic | Anthropic messages     | `anthropic` SDK（base_url 为默认值；对于兼容代理可覆盖） |
| Google    | OpenAI chat/completions| 通过 OpenAI 兼容代理（如 OpenRouter）使用 `openai` SDK；必须提供 base_url |
| Qwen      | OpenAI chat/completions| `openai` SDK，DashScope 或 OpenAI 兼容代理；必须提供 base_url |
| DeepSeek  | OpenAI chat/completions| `openai` SDK，`base_url=https://api.deepseek.com`  |
| vLLM      | OpenAI chat/completions| `openai` SDK，用户自定义的本地 `base_url`          |

六个提供方中有五个共用 OpenAI 线路协议。这促使我们采用一个共享的适配器类。

### 规范的内部类型

定义在 `DefenseAgent/llm/types.py` 中。这些是 harness 其余部分**唯一**能看到的形态。

```python
from dataclasses import dataclass, field
from typing import Literal, Any

Role = Literal["system", "user", "assistant", "tool"]

@dataclass
class Message:
    role: Role
    content: str
    # Present on assistant messages that requested tools:
    tool_calls: list["ToolCall"] = field(default_factory=list)
    # Present on tool-result messages (role="tool"):
    tool_call_id: str | None = None
    name: str | None = None  # tool name, when role="tool"

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]  # parsed JSON, not a string

@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int  # = prompt + completion; adapter fills if provider omits

@dataclass
class LLMResponse:
    content: str               # assistant text (may be empty if only tool calls)
    tool_calls: list[ToolCall] # empty if no tools requested
    usage: TokenUsage
    stop_reason: str | None    # "end_turn", "tool_use", "max_tokens", etc. — normalized
    raw: dict[str, Any]        # raw provider response for debugging
```

**归一化决策：**
- `arguments` 始终是已解析的 dict，绝不是 JSON 字符串。OpenAI 适配器会解析 API 返回的 JSON 字符串；Anthropic 适配器直接透传其 `input` dict。
- `stop_reason` 被归一化为一小组词汇：`end_turn`、`tool_use`、`max_tokens`、`stop_sequence`、`other`。两个适配器都会将各自原生值映射到这套词汇。
- 无论提供方是单独返回 `total_tokens` 还是完全不返回，都会被填充。由适配器完成算术运算。

### 抽象接口

```python
# DefenseAgent/llm/llm_adapter.py
from abc import ABC, abstractmethod

class LLMAdapter(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,   # JSON Schema list; adapter translates to vendor format
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system: str | None = None,         # optional; some providers accept a dedicated system param
    ) -> LLMResponse:
        """Send a chat request and return a canonical LLMResponse."""
```

**签名说明：**
- `messages` 是规范形式。如果调用方在列表顶部包含一条 `Message(role="system", ...)`，同时又传入了 `system=...`，适配器会抛出 `LLMAdapterError` —— 这是模糊的。调用方应当二选一。
- `tools` 使用 JSON Schema（OpenAI 形态）。Anthropic 适配器会将其翻译为 Anthropic 的 `tools` 格式。

### `AnthropicAdapter`

```python
# DefenseAgent/llm/anthropic_adapter.py
class AnthropicAdapter(LLMAdapter):
    def __init__(self, api_key: str, model: str, base_url: str | None = None): ...
    async def chat(self, messages, *, tools=None, ...): ...
```

**职责：**
1. 从列表中剥离所有 `role="system"` 消息，并将其合并为一个 system 字符串（以换行符拼接）。使用该字符串或者 `system=` kwarg —— 两者同时存在则报错。
2. 将 `messages` 翻译为 Anthropic 的消息格式：
   - 助手消息上的 `tool_calls` → 类型为 `tool_use` 的 `content` 块。
   - `role="tool"` 的消息 → 放置在 `user` 消息内部、类型为 `tool_result` 的 `content` 块。
3. 将 `tools`（JSON Schema）翻译为 Anthropic 的 `tools` 列表（结构相同，键略有不同）。
4. 调用 `anthropic.AsyncAnthropic(api_key=...).messages.create(...)`。
5. 将响应翻译为 `LLMResponse`：
   - 拼接所有 `text` 块 → `content`。
   - 收集 `tool_use` 块 → `tool_calls`（将 `input` dict 直接传入 `arguments`）。
   - 映射 `stop_reason`：`end_turn`→`end_turn`、`tool_use`→`tool_use`、`max_tokens`→`max_tokens`、`stop_sequence`→`stop_sequence`，其他一切 → `other`。
   - 从 `response.usage` 构建 `TokenUsage`。

### `OpenAICompatibleAdapter`

```python
# DefenseAgent/llm/openai_compatible_adapter.py
class OpenAICompatibleAdapter(LLMAdapter):
    def __init__(self, api_key: str, base_url: str, model: str): ...
    async def chat(self, messages, *, tools=None, ...): ...
```

**职责：**
1. 如果提供了 `system=` kwarg，在前面添加一条 `{"role": "system", "content": system}` 消息；若 `messages` 中已经存在 system 角色则报错。
2. 将规范的 `Message` 列表翻译为 OpenAI 消息 dict：
   - 带有 `tool_calls` 的助手消息 → 包含 `tool_calls` 字段（将已解析的 dict 转为 JSON 字符串作为 `arguments`）。
   - `role="tool"` → `{"role": "tool", "tool_call_id": ..., "name": ..., "content": ...}`。
3. 将 `tools`（JSON Schema）翻译为 OpenAI 的 `[{"type": "function", "function": {...}}]` 形态。
4. 调用 `openai.AsyncOpenAI(api_key=..., base_url=...).chat.completions.create(...)`。
5. 将响应翻译为 `LLMResponse`：
   - `choices[0].message.content` → `content`（只调用工具时可能为 `None`/空）。
   - `choices[0].message.tool_calls` → `tool_calls`（将每个 `function.arguments` 的 JSON 字符串解析为 dict）。
   - 映射 `finish_reason`：`stop`→`end_turn`、`tool_calls`→`tool_use`、`length`→`max_tokens`，其他 → `other`。
   - 从 `response.usage` 构建 `TokenUsage`。

### 工厂

```python
# DefenseAgent/llm/factory.py
def make_adapter_from_env(
    dotenv_path: str | None = None,
    *,
    load_env: bool = True,
) -> LLMAdapter:
    """Read env (optionally loading .env first) and return the configured adapter."""
```

**环境变量结构：**

- **选择器：** `AGENT_LAB_LLM_PROVIDER` —— 取 `openai | anthropic | google | deepseek | qwen | vllm` 之一（不区分大小写，去除空白）。
- **按提供方划分的块：** `{PROVIDER}_API_KEY`、`{PROVIDER}_BASE_URL`、`{PROVIDER}_MODEL`，其中 `{PROVIDER}` 是选择器的值转大写（例如 `OPENAI_API_KEY`、`ANTHROPIC_MODEL`、`DEEPSEEK_BASE_URL`）。
- **覆盖层（按字段）：** `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL_ID`。每一项只要非空，就会覆盖对应提供方的值。各项独立求值 —— 用户可以只覆盖 model 而不触碰 key 或 base_url。

**逻辑：**
1. 如果 `load_env=True`，调用 `dotenv.load_dotenv(dotenv_path, override=False)`。测试会传入 `load_env=False`，以在受控的干净环境中工作。
2. 从 `AGENT_LAB_LLM_PROVIDER` 解析 `provider`（去空白、转小写）。空值或未知值 → `LLMConfigError`。
3. 对每个字段（api_key、base_url、model）：如果 `LLM_*` 覆盖项非空则取它，否则取 `{PROVIDER}_*` 值。空字符串视为"未设置"。
4. 校验：
   - `model` 为空 → `LLMConfigError`。
   - `base_url` 为空且 `provider in {google, qwen, deepseek, vllm}` → `LLMConfigError`。
   - `api_key` 为空且 `provider != vllm` → `LLMConfigError`。（vLLM 容忍空 key —— 默认为 `"token-not-needed"`，OpenAI 兼容服务端会接受。）
5. 构建：
   - `provider == anthropic` → `AnthropicAdapter(api_key=..., model=..., base_url=base_url or None)`。
   - 否则 → `OpenAICompatibleAdapter(api_key=..., base_url=..., model=...)`。对于 `openai`，`base_url` 仍可为空（使用 SDK 默认）。

**存在但尚未接入适配器的额外环境变量**（为后续模块保留）：
`AGENT_LAB_LLM_TEMPERATURE`、`LLM_MAX_TOKENS`、`LLM_TIMEOUT`、`LLM_MAX_RETRIES`、`TAVILY_API_KEY`、`AGENT_LAB_DISABLE_BACKGROUND_RUNTIME`、`AGENT_LAB_RUNTIME_AUTONOMOUS_A2A`、`AGENT_LAB_RUNTIME_LLM_DECISIONS`。它们存在于 `.env.example` 中以便用户知晓结构，但本模块并不消费它们。

### 错误

定义在 `DefenseAgent/llm/errors.py` 中：

```python
class LLMError(Exception): ...                  # base
class LLMConfigError(LLMError): ...             # bad/missing .env config
class LLMAdapterError(LLMError): ...            # adapter-layer bug or misuse
class LLMProviderError(LLMError):               # provider API returned an error
    def __init__(self, provider: str, status_code: int | None, message: str): ...
```

适配器将提供方异常封装为 `LLMProviderError`，通过 `raise ... from e` 保留原始原因。

### `.env` 与 `.env.example`

完整模板参见已提交的 [`.env.example`](../../../.env.example)。各部分概要如下：

- **选择器：** `AGENT_LAB_LLM_PROVIDER`。
- **覆盖层：** `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL_ID`。
- **提供方块：** 每个支持的提供方各一个（`OPENAI_*`、`ANTHROPIC_*`、`GOOGLE_*`、`DEEPSEEK_*`、`QWEN_*`、`VLLM_*`）。
- **模型参数默认值（尚未接入）：** `AGENT_LAB_LLM_TEMPERATURE`、`LLM_MAX_TOKENS`、`LLM_TIMEOUT`、`LLM_MAX_RETRIES`。
- **未来模块占位符：** `TAVILY_API_KEY`、`AGENT_LAB_DISABLE_BACKGROUND_RUNTIME`、`AGENT_LAB_RUNTIME_AUTONOMOUS_A2A`、`AGENT_LAB_RUNTIME_LLM_DECISIONS`。

`.env` 结构相同，但被 git 忽略，并由用户的密钥填充。

### 文件布局

```
agent_lab/
├── .env                           # gitignored, blank initially
├── .env.example                   # committed template
├── .gitignore                     # .env, __pycache__, .pytest_cache
├── requirements.txt               # anthropic, openai, python-dotenv, pytest, pytest-asyncio
├── docs/superpowers/specs/
│   └── 2026-04-22-module-01-llm-adapter-design.md   # this document
├── DefenseAgent/                 # all harness code lives here
│   ├── __init__.py
│   └── llm/
│       ├── __init__.py            # re-exports LLMAdapter, types, factory
│       ├── types.py
│       ├── errors.py
│       ├── llm_adapter.py         # abstract base
│       ├── anthropic_adapter.py
│       ├── openai_compatible_adapter.py
│       └── factory.py
└── tests/
    └── DefenseAgent/
        ├── __init__.py
        └── llm/
            ├── __init__.py
            ├── conftest.py        # shared fixtures
            ├── test_types.py
            ├── test_errors.py
            ├── test_llm_adapter.py
            ├── test_anthropic_adapter.py
            ├── test_openai_compatible_adapter.py
            └── test_factory.py
```

## 测试策略

测试中不进行真实的 API 调用。策略：

- **`AnthropicAdapter` 测试**：对 `anthropic.AsyncAnthropic` 打补丁，使其返回一个桩对象，其 `messages.create` 是一个返回手工构造响应对象的 `AsyncMock`。断言适配器构造出的请求（通过 mock 的调用参数捕获）以及翻译后的 `LLMResponse`。
- **`OpenAICompatibleAdapter` 测试**：对 `openai.AsyncOpenAI` 采用相同做法。
- **工厂测试**：使用 `monkeypatch.setenv` 设置 `PROVIDER`/`API_KEY`/`BASE_URL`/`MODEL`，断言返回的适配器类型正确且其配置已连线。还要测试每一条错误路径。
- **类型测试**：对 dataclass 的往返和默认值进行廉价的合理性检查。

本模块的覆盖率目标：
- 每个适配器的每个公共方法至少有一个正常路径（happy-path）测试。
- 消息格式翻译在双向（输入翻译与响应翻译）上都要测试。
- 两个适配器的工具调用路径都要做端到端测试。
- 工厂中每一条 `LLMConfigError` 分支都要被测试。

## 本模块新增的依赖

加入 `requirements.txt`：
```
anthropic>=0.40.0
openai>=1.50.0
python-dotenv>=1.0.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

## 未来扩展（不在本模块中）

当未来的模块需要它们时，此接口会扩展：
- `async embed(text: str | list[str]) -> list[list[float]]` —— 随记忆模块一起加入。
- `async chat_stream(...) -> AsyncIterator[Chunk]` —— 当 UI 或基于事件的消费者出现时加入。
- `async score_importance(observation: str) -> float` —— 作为 `chat()` 的封装存在于记忆模块中；不是适配器方法。

规范类型的设计使得这些新增不会破坏调用方。

## 遗留问题

在规范获批之时没有遗留问题。设计选择已于 2026-04-22 与用户确认：
1. 针对 OpenAI/Google/Qwen/DeepSeek/vLLM 共用一个 `OpenAICompatibleAdapter` 类 —— 已确认。
2. `embed()` 和 `score_importance()` 推迟 —— 已确认。
3. 环境变量结构：按提供方划分的块加上按字段的 `LLM_*` 覆盖层（而非全有或全无）—— 在用户提供其偏好的 `.env` 格式后采用。
4. 支持的提供方：`openai, anthropic, google, deepseek, qwen, vllm`。`claude` 重命名为 `anthropic` 以匹配 SDK 名称。

## 修订历史

- **2026-04-22**：将工厂从扁平环境变量（`PROVIDER/API_KEY/BASE_URL/MODEL`）切换为带命名空间的按提供方块加覆盖层（`AGENT_LAB_LLM_PROVIDER`、`{PROVIDER}_*`、`LLM_*`）。新增 `google` 提供方。将 `claude` 重命名为 `anthropic`。适配器类与规范类型没有变化。
