# 模块 2 代码导览 — 智能体档案加载器

> 配套[设计规范](../superpowers/specs_ZN/2026-04-22-module-02-agent-profile-design.md)阅读。逐行讲解代码，并追踪 `show_profile.py` 与 `profile_chat_demo.py` 的执行过程。

---

## 核心类：`AgentProfile`

从这里开始。规范的入口是 `AgentProfile.from_yaml(path)`：

```python
from DefenseAgent.config import AgentProfile

profile = AgentProfile.from_yaml("agents/maya_rodriguez/profile.yaml")
print(profile.name, profile.age)
print(profile.tools.skills)    # declared skill paths (resolved relative to the profile)
```

自由函数 `load_profile(path)` 仍然存在，功能完全一致——它为了向后兼容而保留，但新代码应当使用 `AgentProfile.from_yaml(...)`。

本代码导览的剩余部分涵盖 Pydantic 模式、加载器的五阶段流程以及错误层级——也就是 `AgentProfile.from_yaml()` 在底层真正做的事情。

---

## 1. 本模块解决的问题

每个智能体都有身份信息（id、name、traits、backstory）和调参旋钮（认知阈值、记忆权重）。这些值存放在 YAML 文件中，这样无需触碰代码就能编辑。但 YAML 本身没有类型、过于宽容——像把 `traits:` 写成 `traists:` 这样的拼写错误会悄无声息地通过，而框架会在三个模块之后才出现异常行为。

**模块 2 做三件事：**
1. 定义一个**带类型的 Pydantic 模型**（`AgentProfile`），为每个字段命名并进行校验。
2. 提供一个**严格加载器**（`AgentProfile.from_yaml`），将 YAML 解析为该模型，并在任何不匹配时显式失败。
3. 在 `agents/<id>/` 下提供**参考包**，让演示和测试有具体可加载的内容——并让每个智能体的工具集（技能 + MCP 服务器）独立存放在自己的目录中。

---

## 2. 目录结构 *(2026-04-24 修订 — 智能体包)*

```
DefenseAgent/config/                # loader CODE (no data)
├── __init__.py                      # public API re-exports
└── profile.py                       # models + ConfigError hierarchy + from_yaml

agents/                              # user-editable DATA — one bundle per agent
├── alice_chen/
│   └── profile.yaml                 # reference profile — a data scientist
└── maya_rodriguez/
    ├── profile.yaml                 # identity + cognitive + memory + tools
    └── skills/                      # private skills (loaded by ToolRegistry.from_profile)
        └── tabular-report/
            ├── SKILL.md
            ├── scripts/
            └── templates/

scripts/
├── show_profile.py                  # load a profile + pretty-print it
├── profile_chat_demo.py             # load a profile + use it with the LLM
└── tools_demo.py                    # load a profile + build its ToolRegistry
```

**关键布局决策：**

- **加载器代码与智能体数据存放在不同目录。** `DefenseAgent/config/` 中没有 YAML；`agents/` 中没有 Python。
- **每个智能体都是一个包。** 身份（`profile.yaml`）、私有技能以及 MCP 声明都存放在 `agents/<agent_id>/` 内部。创建一个新的智能体只需 `cp -r agents/maya_rodriguez agents/new_agent` 然后编辑档案——仓库中其它地方无需关心。
- **工具路径从档案所在目录解析。** 在 `agents/maya/profile.yaml` 中声明 `skills: [skills/foo]` 意味着 `agents/maya/skills/foo`，*而不是*一个共享的顶层 `skills/`。智能体之间不会意外看到彼此的工具。

---

## 3. Pydantic 模型（`profile.py`）

三个嵌套的 BaseModel 子类。每个都设置了 `model_config = ConfigDict(extra="forbid")`——未知键会变成校验错误。

### `CognitiveConfig`
```python
class CognitiveConfig(BaseModel):
    model_config = _STRICT

    max_steps_per_cycle: int   = Field(ge=1, default=10)
    reflection_threshold: int  = Field(ge=1, default=5)
    importance_threshold: float = Field(ge=1, le=10, default=7)
    planning_horizon: str      = Field(min_length=1, default="1 day")
```

| 字段 | 含义 | 约束原因 |
|---|---|---|
| `max_steps_per_cycle` | 一次清醒周期内的最大行动数 | ≥1 —— 零步周期没有意义 |
| `reflection_threshold` | 积累 N 条新记忆后触发反思 | ≥1 —— 对零条记忆反思是空操作 |
| `importance_threshold` | 高于此值（1–10）的记忆被视为"重要" | 1–10 与打分刻度一致 |
| `planning_horizon` | 类似 "1 day" 这样的自由文本字符串 | 自由形式，因为解析属于未来的事情 |

所有字段都有默认值，因此在 YAML 中可以省略整个 `cognitive:` 块。

### `MemoryConfig`
```python
class MemoryConfig(BaseModel):
    model_config = _STRICT

    max_working_memory_tokens: int = Field(ge=1, default=4000)
    retrieval_top_k: int           = Field(ge=1, default=10)
    recency_weight: float          = Field(ge=0, default=1.0)
    importance_weight: float       = Field(ge=0, default=1.0)
    relevance_weight: float        = Field(ge=0, default=1.0)
```

权重使用 `ge=0`（而不是 `gt=0`），这样用户可以通过把某项权重设为 0 来禁用它。例如 `recency_weight: 0.0` 意味着"在为检索打分记忆时完全忽略新近度"。

### `AgentProfile`
```python
class AgentProfile(BaseModel):
    model_config = _STRICT

    id: str            = Field(min_length=1)
    name: str          = Field(min_length=1)
    age: int           = Field(ge=0)
    traits: str        = Field(min_length=1)
    backstory: str     = Field(min_length=1)
    initial_plan: str  = Field(min_length=1)
    cognitive: CognitiveConfig = Field(default_factory=CognitiveConfig)
    memory: MemoryConfig       = Field(default_factory=MemoryConfig)
```

**为什么嵌套模型使用 `default_factory`**：使用 `= CognitiveConfig()` 作为默认值会在所有 AgentProfile 实例之间创建一个共享实例——对其进行修改会产生泄漏。`default_factory=CognitiveConfig` 会为每个档案构建一个新的实例。

**所有身份字段都是必填的**，其它都不是。一个最小的 YAML 只需要这六个字段加上 `agent:` 包装。

---

## 4. 加载器（`profile.py`）

```python
def load_profile(path: str | Path) -> AgentProfile:
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigFileNotFoundError(f"profile file not found: {file_path}")

    raw_text = file_path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as e:
        raise ConfigParseError(f"invalid YAML in {file_path}: {e}") from e

    if not isinstance(data, dict):
        raise ConfigParseError(f"expected top-level mapping ..., got {type(data).__name__}")
    if "agent" not in data:
        raise ConfigParseError(f"missing top-level 'agent:' key in {file_path}")
    agent_data = data["agent"]
    if not isinstance(agent_data, dict):
        raise ConfigParseError(f"'agent:' value must be a mapping ...")

    try:
        return AgentProfile.model_validate(agent_data)
    except ValidationError as e:
        raise ConfigValidationError(f"profile at {file_path} failed schema validation:\n{e}") from e
```

从上往下读：

1. **读取前先 `is_file()`。** 缺失路径和目录都会快速失败为 `ConfigFileNotFoundError`。（注意：对目录调用 `open()` 会抛出 `IsADirectoryError`；我们提前捕获它，这样调用方只会看到 `ConfigError` 子类。）

2. **`yaml.safe_load`，而不是 `yaml.load`。** 这很重要。`yaml.load` 会处理 `!!python/object:os.system` 和其它能在解析时执行任意代码的标签。`safe_load` 会拒绝它们——它只加载简单类型（字符串、数字、列表、字典、布尔值）。`test_safe_load_rejects_python_object_tag` 会守护这一点。

3. **在 schema 之前进行形状检查。** 空 YAML 文件解析为 `None`；像 `"hi"` 这样的标量解析为字符串；顶层列表解析为列表。这些都不是字典，所以尝试校验它们时会得到奇怪的 pydantic 错误。我们改为抛出清晰的 `ConfigParseError` 消息，明确指出顶层实际上是什么。

4. **`"agent" not in data` 检查。** 没有它，用户会得到一个令人困惑的 pydantic 错误，提示缺少 `id`、`name` 等。我们直接告诉他们：缺少顶层 `agent:` 键。

5. **`model_validate` + 包装后的 ValidationError。** Pydantic 自己的 `ValidationError` 很冗长。我们用自己的简短消息将其包装成 `ConfigValidationError`，但通过 `from e` 保留 pydantic 错误，这样调用方仍能通过 `e.__cause__` 查看字段级细节。

---

## 5. 错误（`errors.py`）

```
ConfigError (base)
├── ConfigFileNotFoundError   — path does not exist (or is a directory)
├── ConfigParseError          — file exists but isn't a valid YAML mapping with an 'agent:' key
└── ConfigValidationError     — YAML parses but schema fails; __cause__ is pydantic.ValidationError
```

四种不同的失败模式，每种都有自己的类。调用方可以精准处理：
```python
try:
    profile = load_profile(path)
except ConfigFileNotFoundError:
    ...  # bad path
except ConfigParseError:
    ...  # fix your YAML
except ConfigValidationError as e:
    print(e.__cause__.errors())   # per-field pydantic detail
```

---

## 6. 示例 YAML —— `agents/maya_rodriguez/profile.yaml`

```yaml
agent:
  id: "student_maya_001"
  name: "Maya Rodriguez"
  age: 20
  traits: "curious, persistent, collaborative"
  backstory: >
    A second-year Computer Science student at a state university.
    Grew up bilingual (Spanish/English)...
  initial_plan: >
    Wake up at 7:30, review yesterday's lecture notes over coffee,
    attend the 9 AM data structures lecture...
  cognitive:
    max_steps_per_cycle: 8
    reflection_threshold: 4
    importance_threshold: 6
    planning_horizon: "1 day"
  memory:
    max_working_memory_tokens: 3000
    retrieval_top_k: 8
    recency_weight: 1.0
    importance_weight: 1.2
    relevance_weight: 1.5
```

**YAML `>` 折叠标量。** `>` 之后的行会用空格连接，并在末尾保留一个换行。这让我们可以书写多行背景故事而无需字面的 `\n` 字符，同时仍能干净地序列化。加载器在构建系统提示词时会对结果调用 `.strip()`，这样尾部的 `\n` 就不会泄漏到 LLM 输入中。

---

## 7. 执行流程：`scripts/show_profile.py`

快速诊断工具。接受一个可选路径；默认为 `agents/alice_chen/profile.yaml`。

```
$ python scripts/show_profile.py

┌─ main()
│
├─ 1. path = DEFAULT_PATH (= agents/alice_chen/profile.yaml)
│        or argv[1] if provided
│
├─ 2. profile = load_profile(path)
│     │
│     ├─ ConfigFileNotFoundError? print + return 1
│     ├─ ConfigParseError? print + return 1
│     └─ ConfigValidationError? print + return 1
│     (all three subclasses are caught by `except ConfigError`)
│
├─ 3. print("[show_profile] loaded <path>")
│
└─ 4. print(json.dumps(profile.model_dump(), indent=2))
       • model_dump() returns a pure-dict representation
       • json.dumps pretty-prints it (including nested cognitive/memory blocks)
```

示例输出：
```
[show_profile] loaded /Users/.../agent_lab/agents/alice_chen/profile.yaml
[show_profile] model: AgentProfile
---
{
  "id": "agent_001",
  "name": "Alice Chen",
  ...
  "cognitive": { ... },
  "memory": { ... }
}
```

这是回答"我编辑的 YAML 有没有破坏什么"最快的方式——如果 `load_profile` 失败，你会看到具体是哪个错误类 + 消息。

---

## 8. 执行流程：`scripts/profile_chat_demo.py` —— 模块 1 ⊗ 模块 2

组合演示。加载一个档案（模块 2）+ 从 .env 加载一个适配器（模块 1）+ 发送一轮对话。

```
$ python scripts/profile_chat_demo.py

┌─ main() (async)
│
├─ Step 1: load_profile(agents/maya_rodriguez/profile.yaml)
│          → AgentProfile(name="Maya Rodriguez", age=20, ...)
│          → fails fast with ConfigError if YAML is broken
│
├─ Step 2: make_adapter_from_env()
│          → reads AGENT_LAB_LLM_PROVIDER from .env
│          → reads {PROVIDER}_API_KEY / BASE_URL / MODEL (with LLM_* overrides)
│          → returns OpenAICompatibleAdapter or AnthropicAdapter
│          → fails fast with LLMConfigError if .env is incomplete
│
├─ Step 3: build_system_prompt(profile)
│          → returns a multi-line str built from profile.name, age, traits,
│            backstory, initial_plan + the "stay in character" instruction
│
├─ Step 4: adapter.chat(
│              messages=[Message(role="user", content=USER_QUESTION)],
│              system=system_prompt,
│              temperature=0.7,
│              max_tokens=256,
│          )
│          │
│          ├─ OpenAICompatibleAdapter._chat (for DeepSeek)
│          │   • merges system_prompt as first {"role":"system","content":...} on the wire
│          │   • serializes messages to OpenAI shape
│          │   • HTTPS POST to api.deepseek.com/v1/chat/completions
│          │   • parses choices[0].message.content → LLMResponse
│          │
│          └─ returns LLMResponse
│
└─ Step 5: print profile.name + resp.content + usage
          e.g.: "[demo] Maya Rodriguez: Morning was solid! ..."
```

**为什么这作为一个例子很重要：** 它展示了未来每个框架模块都会使用的**精确组合模式**。认知循环、上下文管理器和记忆都将被传入一个 `AgentProfile` 和一个 `LLMAdapter`，并且会做大致相同的编排：*用档案塑造提示词 → 调用适配器 → 解释响应*。这个演示是手写版本的流程，而它很快会被 `core/` 自动化。

**执行可能在何处失败？**

| 失败情形 | 阶段 | 异常类 | 退出码 |
|---|---|---|---|
| 给定路径下 YAML 文件缺失 | Step 1 | `ConfigFileNotFoundError` | 2 |
| YAML 文件无法解析 | Step 1 | `ConfigParseError` | 2 |
| YAML 文件有效但模式不匹配 | Step 1 | `ConfigValidationError` | 2 |
| `.env` 缺失 provider / key / model | Step 2 | `LLMConfigError` | 2 |
| 网络失败、认证错误、限流 | Step 4 | `LLMProviderError` | 1 |

所有五条错误路径都被演示捕获；退出码用于区分"请修复配置"（2）与"运行时失败，或许可重试"（1）。

---

## 9. 测试覆盖映射

| 文件 | 测试数 | 覆盖内容 |
|---|---|---|
| `tests/DefenseAgent/config/test_errors.py` | 3 | 异常层级 + `__cause__` 保留 |
| `tests/DefenseAgent/config/test_profile_models.py` | 34 | 每个字段的默认值，每个校验器（参数化），每个嵌套模型对额外键的拒绝，空字符串拒绝，`age=0` 边界情形 |
| `tests/DefenseAgent/config/test_loader.py` | 17 | 顺利路径（最小 + 完整 YAML），每个 `ConfigError` 子类分支，YAML 安全守卫，校验错误上的链式 `__cause__` |
| `tests/DefenseAgent/integration/test_profile_llm_integration.py` | 4 | 附带的学生档案能够往返；身份字段能到达适配器边界；最小档案配合默认值正常组合；内联档案能够组合而不依赖随包发布的 YAML |

所有测试都是完全离线的——YAML 文件写入 `tmp_path`，LLM 使用 `StubAdapter`。

---

## 10. 值得注意的事项

- **数据与代码在磁盘上是分开的。** `DefenseAgent/config/` 中没有 YAML。`agents/` 中没有 Python。你可以删除整个 `agents/`，加载器代码仍然能运行（当你尝试使用它时会抛出 `ConfigFileNotFoundError`）。

- **每个智能体都是自包含的。** `agents/maya_rodriguez/` 存放着 Maya 的档案、她的技能和她的 MCP 声明——属于 Maya 的一切都不存放在其它地方。复制粘贴整个文件夹即可创建一个新智能体；档案中声明的所有路径都是相对路径，因此它们随包一起迁移。

- **`profile.source_dir` 是相对路径的锚点。** `AgentProfile.from_yaml` 会在实例上记录已解析的 YAML 位置。`ToolRegistry.from_profile` 读取 `profile.tools.skills`，并将每一项与 `profile.source_dir` 拼接得到绝对路径，再调用 `add_skill`。两个智能体永远不可能意外看到彼此的技能，因为每个智能体的路径只会解析到自己的包之下。
- **严格 Pydantic 模式让拼写错误大声报错。** 每个模型上的 `extra="forbid"` 意味着 YAML 中的 `traists:` 会在档案到达任何消费者之前就成为 ValidationError。这在教学上很重要——悄无声息地忽略会让学习模式变得更困难。
- **嵌套模型上的 `default_factory`。** 使用 `Field(default_factory=CognitiveConfig)`（而不是 `= CognitiveConfig()`）是正确的 pydantic 模式，当测试开始跨用例修改档案对象时这一点就变得重要。
- **加载器不知道 LLM。** `profile.py` 导入 `yaml` 和 `pydantic`。它**不**从 `DefenseAgent.llm` 导入任何东西。组合只发生在 `scripts/profile_chat_demo.py` 和集成测试中——而不在两个模块内部。未来的跨模块逻辑（从档案构建系统提示词、将档案的记忆权重喂给检索器）会存放在未来的 `DefenseAgent/core/` 模块中，而不是在这里。
- **集成测试附带一个 `StubAdapter`。** 它直接子类化 `LLMAdapter`。这是 `DefenseAgent/llm/` 之外第一个验证抽象基类契约可行的地方——任何实现了 `chat()` 的 `LLMAdapter` 子类都可以直接嵌入。这就是适配器层当初费劲做成抽象的原因。
