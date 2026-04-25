# 模块 2 —— 智能体档案加载器设计规范

**日期：** 2026-04-22
**状态：** 已批准，可进入实现阶段
**模块位置：** N 个模块中的第 2 个。后续模块（认知循环、上下文管理器、记忆检索器）会消费本模块，用于获取智能体身份和调优参数。与模块 1（LLM）之间没有运行时耦合。

> **修订（2026-04-24）：**
> - `AgentProfile` 现在携带一个 `tools: ToolsConfig` 字段（`skills: list[str]`、`mcp: list[MCPServerConfig]`），使每个智能体都能声明自己的工具集。默认值为空 —— 早于该字段的档案仍可原样加载。
> - `AgentProfile` 实例会记住自己从哪里加载而来：`profile.source_path` / `profile.source_dir` 由 `from_yaml` 填充，以便下游代码（尤其是 `ToolRegistry.from_profile`）能基于档案所在目录解析相对工具路径。
> - 档案 YAML 从 `profiles/<name>.yaml` 迁移到了 **智能体束（agent bundles）** `agents/<id>/profile.yaml`，每个智能体的 `skills/` 目录与其档案同级。下方"文件布局"块反映了当前结构；模型表和加载器章节内联呈现修订。

## 目的

把一个 YAML 文件变成一个有类型、经过验证的 `AgentProfile` 对象。整个 harness 的其他部分都不应该直接读取 YAML —— 所有智能体配置旋钮都要流经本模块的模型。YAML 中的拼写错误必须在加载时就大声失败，而不是到三个模块之后的某次认知步骤中才暴露。

## 范围

### 范围之内
- 三个 Pydantic v2 模型：`CognitiveConfig`、`MemoryConfig`、`AgentProfile`。
- `load_profile(path) -> AgentProfile` 加载器（读取文件、解析 YAML、验证、返回模型）。
- 严格验证（`extra = "forbid"`）：未知键会被拒绝。
- 自定义错误层级（`ConfigError` 加三个子类），让调用方可以按失败模式分支处理。
- 位于 `profiles/alice_chen.yaml` 的默认/参考 YAML。
- 覆盖每个模型验证器和加载器每个错误分支的单元测试。
- 示例脚本 `scripts/show_profile.py`，加载默认文件并漂亮打印解析后的模型。

### 范围之外（延后）
- `settings.yaml`（如日志级别等运行时设置）—— 这是另一个关注点，等到有消费方出现时再加。
- 把 `planning_horizon` 解析为结构化的时长（例如 `timedelta`）—— 在认知循环真正用到它之前，保持自由格式字符串。
- 热重载 / 文件监听。
- 多档案加载 / 档案继承。
- 与 LLM 模块的集成（目前还没有耦合）。

## 设计

### 模型

所有模型都设置 `model_config = ConfigDict(extra="forbid")` —— 未知的 YAML 键会触发验证错误。

#### `CognitiveConfig`
认知循环的调优旋钮。所有字段都有合理的默认值，所以最小化档案可以整体省略这个块。

| Field                  | Type  | Constraint | Default |
|------------------------|-------|------------|---------|
| `max_steps_per_cycle`  | int   | ≥ 1        | 10      |
| `reflection_threshold` | int   | ≥ 1        | 5       |
| `importance_threshold` | float | 1 ≤ x ≤ 10 | 7       |
| `planning_horizon`     | str   | non-empty  | "1 day" |

#### `MemoryConfig`
记忆系统的权重与预算。所有字段都有合理的默认值。

| Field                        | Type  | Constraint | Default |
|------------------------------|-------|------------|---------|
| `max_working_memory_tokens`  | int   | ≥ 1        | 4000    |
| `retrieval_top_k`            | int   | ≥ 1        | 10      |
| `recency_weight`             | float | ≥ 0        | 1.0     |
| `importance_weight`          | float | ≥ 0        | 1.0     |
| `relevance_weight`           | float | ≥ 0        | 1.0     |

#### `MCPServerConfig` *（2026-04-24 新增）*
单个 MCP 服务器的 stdio 启动参数。

| Field     | Type              | Constraint | Default |
|-----------|-------------------|------------|---------|
| `command` | str               | non-empty  | (required) |
| `args`    | list[str]         | —          | `[]`    |
| `env`     | dict[str,str] \| None | —      | `None`  |
| `cwd`     | str \| None       | —          | `None`  |

#### `ToolsConfig` *（2026-04-24 新增）*
按智能体声明的工具。所有字段默认为空列表，这样现有档案仍能通过验证。

| Field    | Type                        | Default |
|----------|-----------------------------|---------|
| `skills` | list[str] (paths)           | `[]`    |
| `mcp`    | list[`MCPServerConfig`]     | `[]`    |

#### `AgentProfile`
身份 + 嵌套配置。身份字段是必填的；每个嵌套块都有默认工厂，所以最小化档案可以省略它们。

| Field          | Type               | Constraint        | Default                |
|----------------|--------------------|-------------------|------------------------|
| `id`           | str                | non-empty         | (required)             |
| `name`         | str                | non-empty         | (required)             |
| `age`          | int                | ≥ 0               | (required)             |
| `traits`       | str                | non-empty         | (required)             |
| `backstory`    | str                | non-empty         | (required)             |
| `initial_plan` | str                | non-empty         | (required)             |
| `cognitive`    | `CognitiveConfig`  | —                 | `CognitiveConfig()`    |
| `memory`       | `MemoryConfig`     | —                 | `MemoryConfig()`       |
| `tools`        | `ToolsConfig`      | —                 | `ToolsConfig()`        |

**来源追踪。** `from_yaml` 会在返回的实例上填充两个"私有但可读"的属性：

- `profile.source_path: Path | None` —— YAML 加载时解析出的路径。
- `profile.source_dir: Path | None`  —— 它的父目录，`ToolRegistry.from_profile` 以此作为锚点来解析相对的 skill 路径。

纯内存中的档案（通过 `AgentProfile(...)` 或 `model_validate` 构建）两个字段都会被设为 `None`。希望为这类档案加载工具的调用方，必须向 `ToolRegistry.from_profile` 显式传入 `base_dir`。

### YAML 形态

加载器期望顶层有且仅有一个 `agent:` 键，其值匹配 `AgentProfile`。理由：YAML 文件经常会演进成携带多个顶层小节（例如 `agent:`、`environment:`、`world:`）；从一开始就在 `agent:` 下命名空间化，为以后留门。

```yaml
agent:
  id: "agent_001"
  name: "Alice Chen"
  age: 28
  traits: "curious, methodical, empathetic"
  backstory: "A data scientist who recently moved to Lakeside..."
  initial_plan: "Wake up, review emails, work on the data analysis project"
  cognitive:
    max_steps_per_cycle: 10
    reflection_threshold: 5
    importance_threshold: 7
    planning_horizon: "1 day"
  memory:
    max_working_memory_tokens: 4000
    retrieval_top_k: 10
    recency_weight: 1.0
    importance_weight: 1.0
    relevance_weight: 1.0
```

### 加载器

```python
def load_profile(path: str | Path) -> AgentProfile:
    """Read YAML at `path`, validate, return an AgentProfile.

    Raises:
        ConfigFileNotFoundError: path does not exist.
        ConfigParseError:        file exists but isn't valid YAML, or top-level isn't a mapping.
        ConfigValidationError:   YAML parses but doesn't match the AgentProfile schema
                                 (missing required field, wrong type, out-of-range value,
                                 unknown key, etc.).
    """
```

**加载器内部的解析顺序：**
1. 打开文件 → 如果不存在则抛 `ConfigFileNotFoundError`。
2. `yaml.safe_load` → 遇到 `yaml.YAMLError` 抛 `ConfigParseError`，如果结果不是 dict 也抛。
3. 要求顶层有 `agent:` 键 → 缺失则抛 `ConfigParseError`。
4. `AgentProfile.model_validate(data["agent"])` → 把 `pydantic.ValidationError` 包装进 `ConfigValidationError`，原始的 pydantic 错误作为 `__cause__`。

使用 `yaml.safe_load`（而不是 `yaml.load`）来拒绝 `!!python/object` 等代码执行路径。

### 错误类型（`DefenseAgent/config/errors.py`）

```python
class ConfigError(Exception): ...
class ConfigFileNotFoundError(ConfigError): ...
class ConfigParseError(ConfigError): ...
class ConfigValidationError(ConfigError):
    """YAML parsed but failed schema validation.

    The original pydantic ValidationError is attached via `raise ... from e`.
    """
```

### 文件布局 *（2026-04-24 修订）*

```
DefenseAgent/config/                # loader CODE only
├── __init__.py                      # re-exports AgentProfile + Tools/MCP models + errors
└── profile.py                       # all models + ConfigError hierarchy + from_yaml

agents/                              # user-editable DATA — one bundle per agent
├── alice_chen/
│   └── profile.yaml
└── maya_rodriguez/
    ├── profile.yaml                 # identity + cognitive + memory + tools
    └── skills/                      # private skills (resolved by ToolRegistry.from_profile)
        └── tabular-report/
            ├── SKILL.md
            ├── scripts/
            │   └── generate.py
            └── templates/
                └── header.md

tests/DefenseAgent/config/
├── __init__.py
├── test_profile.py                  # model + loader validation
└── test_tools_config.py             # ToolsConfig / MCPServerConfig / source_path

scripts/
├── show_profile.py                  # load + pretty-print the default profile
├── profile_chat_demo.py             # load a profile + chat via the LLM
└── tools_demo.py                    # load profile + build ToolRegistry from it
```

原先的布局是一个扁平的 `profiles/` 目录加一个单一的 `errors.py`。两者都在早先的迭代中被整合；按智能体束拆分是 2026-04-24 的改动。

### 新增的依赖

加入 `requirements.txt`：
```
pyyaml>=6.0
pydantic>=2.0
```

（`pydantic` 之前是通过 `anthropic` 间接引入的传递依赖；现在显式声明，由我们自己掌控下界。）

## 测试策略

不需要 mock I/O —— pytest 的 `tmp_path` fixture 会为每个测试提供全新的临时目录。测试把小型 YAML 文件写入 `tmp_path`，然后调用 `load_profile(tmp_path / "x.yaml")`。

**模型测试**（直接构造，不走 YAML）：
- 只填必填字段 → 档案有效，嵌套块被默认填充。
- 每个范围验证器：下界/边界/上界 → 测试边界行为。
- 任意模型上的未知字段 → `ValidationError`（pydantic）。
- `id` / `name` / `traits` / `backstory` / `initial_plan` 上的空字符串 → 被拒绝。

**加载器测试**（基于文件，使用 `tmp_path`）：
- 合法 YAML → 等于期望的 `AgentProfile`。
- 文件缺失 → `ConfigFileNotFoundError`。
- 格式错误的 YAML → `ConfigParseError`。
- 顶层是列表或标量而不是 mapping → `ConfigParseError`。
- 缺失顶层 `agent:` → `ConfigParseError`。
- YAML 形态合法但 `age: "twenty-eight"` → `ConfigValidationError`（pydantic 尝试了强制转换并失败）。
- `ConfigValidationError.__cause__` 是 `pydantic.ValidationError`（这样调用方可以检查细节）。

## 与后续模块的集成

`AgentProfile` 被以下模块消费：
- **认知循环：** `profile.cognitive.max_steps_per_cycle`、`reflection_threshold`、`importance_threshold`、`planning_horizon`。
- **上下文管理器：** `profile.memory.max_working_memory_tokens`。
- **记忆检索器：** `profile.memory.retrieval_top_k`、`recency_weight`、`importance_weight`、`relevance_weight`。
- **智能体编排器：** `profile.id`、`name`、`traits`、`backstory`，用于系统提示。

这些模块目前都还不存在。本模块对下游唯一的契约是：_"我们会交给你一个经过验证的 `AgentProfile`，带有这些字段和这些类型。"_

## 遗留问题

设计规范批准时没有遗留问题。设计选择已于 2026-04-22 与用户确认：
1. Pydantic v2 优先于普通 dataclass —— 已确认。
2. 严格验证（`extra="forbid"`）—— 选择它是为了在拼写错误时得到清晰反馈。
3. YAML 放在 `DefenseAgent/config/` 下（随仓库一起发布），加载器接受任意路径。
4. 单一的 `agent:` 顶层键 —— 为将来的顶层小节预留命名空间。
