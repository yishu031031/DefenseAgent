# 模块 3 代码导览 —— 结构化日志记录器

> 配套阅读[设计规范](../superpowers/specs_ZN/2026-04-22-module-03-logger-design.md)。本文逐行讲解代码，并追踪 `scripts/logger_demo.py` 的执行过程。

---

## 核心类：`AgentLogger`

从这里开始。整个模块就是一个文件里的一个类：

```python
from DefenseAgent.ops import AgentLogger

logger = AgentLogger.from_profile(profile, log_file="logs/maya.log")
logger.info("llm.request", "Calling model", model="deepseek-chat")
```

不需要门面类 —— `AgentLogger` 本身就是唯一统一的接口。下面的代码导览会剖析这一个类：记录的结构、输出端（sink）、级别过滤，以及 I/O 失败时的容错约定。

---

## 1. 本模块解决什么问题

智能体每秒会做出许多小决策 —— 调用 LLM、检索记忆、调用工具、处理错误。如果框架只用 `print()` 语句，调试就意味着重新翻阅终端滚动历史，期望能发现问题。如果使用 Python 默认的 `logging`，每位团队成员都会配置得略有不同，日志聚合器（Splunk、Datadog、磁盘上的 `jq`）将接收到非结构化文本。

**模块 3 给框架提供一个日志记录器，它：**
- 每行输出恰好一个 **JSON 对象**（即 "JSON-lines" 格式）。聚合器无需解析器就能索引每一个事件。
- 给每一行都附上 `agent_id`，所以单个日志文件可以无歧义地容纳多个智能体的事件。
- **永远不会因为 I/O 失败而抛出异常** —— 磁盘写满不应该让智能体崩溃。
- 一行代码就能接入：`logger = AgentLogger.from_profile(profile)`。

---

## 2. 目录结构

```
DefenseAgent/ops/                      # all "operational" concerns
├── __init__.py                         # re-exports AgentLogger
└── logger.py                           # the whole module — one class, one file

tests/DefenseAgent/ops/
├── __init__.py
└── test_logger.py                      # 36 tests covering every behavior

tests/DefenseAgent/integration/
└── test_logger_integration.py          # 2 tests: profile + LLM + logger compose

scripts/
└── logger_demo.py                      # comprehensive demonstration
```

本模块有意保持为一个小文件。没有 `errors.py` —— 日志记录器恰好只有一个面向用户的异常（保留键被误用时抛出 `ValueError`），而标准库已经提供了这个类。

---

## 3. 日志记录的结构剖析

每一次调用都会产生严格如下形状的输出：一个 JSON 对象，一行，以换行符结尾：

```json
{
  "timestamp":  "2026-04-22T10:15:30.480Z",
  "agent_id":   "student_maya_001",
  "level":      "INFO",
  "event_type": "llm.request",
  "message":    "Calling DeepSeek",
  "data":       {"model": "deepseek-chat", "max_tokens": 200}
}
```

| Key | 形状 | 由谁设置 |
|---|---|---|
| `timestamp` | ISO-8601 UTC，毫秒精度，`Z` 后缀 | 日志记录器通过其时钟设置 |
| `agent_id` | 字符串 | 日志记录器（来自 `__init__` 或 `from_profile`） |
| `level` | `DEBUG/INFO/WARNING/ERROR/CRITICAL` 之一 | 被调用的级别方法 |
| `event_type` | 调用方提供的点分标识符（`llm.request`、`tool.timeout` 等） | 调用方 |
| `message` | 调用方提供的可读摘要 | 调用方 |
| `data` | 包含调用方 kwargs 的对象；若无则为 `{}` | 调用方 |

**为什么选择毫秒精度而不是微秒？** 大多数日志聚合器索引到毫秒；在时间戳里塞进完整微秒每条记录会浪费 3 个字节。Python 的 `datetime.isoformat()` 默认给出微秒，因此日志记录器自带一个 `_format_timestamp` 来截断。

---

## 4. 代码逐行讲解：`DefenseAgent/ops/logger.py`

### 模块级常量

```python
_RESERVED_KEYS = frozenset(
    {"agent_id", "level", "timestamp", "event_type", "message", "data"}
)
```

六个顶层记录键。调用方不能传入与这些名字相同的 kwargs —— 否则我们会悄悄覆盖真实值。使用 `frozenset` 意味着成员查询是 O(1)，并且集合不可变（在导入时无法被修改）。

```python
def _default_clock() -> datetime:
    return datetime.now(timezone.utc)
```

默认时间源。测试会注入一个固定时间的 lambda，这样对 `timestamp` 的断言就是确定性的。

```python
def _format_timestamp(dt: datetime) -> str:
    ms = dt.microsecond // 1000
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{ms:03d}Z"
```

毫秒精度加显式的 `Z` 后缀。这个格式正是 `datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ")` 所期望的，所以测试可以往返转换。

### 构造函数

```python
def __init__(
    self,
    agent_id: str,
    *,
    log_file: str | Path | None = None,
    stream: TextIO | None = sys.stdout,
    level: int = logging.INFO,
    clock: Callable[[], datetime] | None = None,
) -> None:
    self.agent_id = agent_id
    self._level = level
    self._stream = stream
    self._file = Path(log_file) if log_file is not None else None
    self._clock = clock or _default_clock
    self._lock = threading.Lock()
    if self._file is not None:
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
```

有四个参数是仅关键字参数（`*`）。这意味着你不会不小心写出 `AgentLogger("a", "path/x.log")` —— 路径必须用 `log_file=`，级别必须用 `level=`，等等。调用点更明确，歧义更少。

**父目录创建**会提前尝试，但包裹在 `try/except OSError` 中。如果父目录创建不了（权限问题、磁盘满），我们吞掉错误 —— 实际的写操作会在之后失败，同样会在那里被吞掉。日志记录器的约定是"永远不让智能体崩溃"，这一点从这里就开始体现。

### `from_profile` —— 与模块 2 的握手

```python
@classmethod
def from_profile(cls, profile: "AgentProfile", **kwargs) -> "AgentLogger":
    return cls(agent_id=profile.id, **kwargs)
```

三行。其余 kwargs 原样透传。`AgentProfile` 类型只在 `TYPE_CHECKING` 下导入，因此运行时导入图是 `ops → nothing`，没有循环依赖风险。

### 级别方法

```python
def debug   (self, event_type, message, /, **data): self.log(logging.DEBUG,    event_type, message, **data)
def info    (self, event_type, message, /, **data): self.log(logging.INFO,     event_type, message, **data)
def warning (self, event_type, message, /, **data): self.log(logging.WARNING,  event_type, message, **data)
def error   (self, event_type, message, /, **data): self.log(logging.ERROR,    event_type, message, **data)
def critical(self, event_type, message, /, **data): self.log(logging.CRITICAL, event_type, message, **data)
```

**`/` 很关键。** 它让 `event_type` 和 `message` 成为仅位置参数。如果没有它，调用
```python
logger.info("e", "m", event_type="oops")
```
会抛出一个原始的 `TypeError: got multiple values for argument 'event_type'`。有了 `/`，kwarg `event_type="oops"` 会被路由进 `**data`，而我们显式的 `_RESERVED_KEYS` 检查会抛出一个清晰且信息丰富的 `ValueError`。五个方法名的行为都一致。

### 核心分发：`log()`

```python
def log(self, level, event_type, message, /, **data):
    if level < self._level:
        return                                            # (1) fast early-out

    collision = _RESERVED_KEYS.intersection(data.keys())
    if collision:
        raise ValueError(
            f"data kwargs may not reuse reserved record keys: {sorted(collision)}."
        )                                                 # (2) hard-fail on programmer error

    record = {
        "timestamp":  _format_timestamp(self._clock()),
        "agent_id":   self.agent_id,
        "level":      logging.getLevelName(level),
        "event_type": event_type,
        "message":    message,
        "data":       dict(data),
    }                                                     # (3) build the record
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"

    with self._lock:                                      # (4) atomic write
        self._write_stream(line)
        self._write_file(line)
```

有四个值得注意的时刻：

**(1) 先做级别过滤，其他一切放后面。** 如果你在代码库中到处撒 `logger.debug(...)` 调用，它们在生产环境下几乎零成本 —— 一次整数比较，然后返回。不构建字典，不生成 JSON，不加锁，也不做 I/O。

**(2) 保留键检查会抛出异常。** 这是日志记录器*唯一*抛异常的地方（除了 I/O 路径，那里是吞掉的）。错误是确定性的、可操作的，并且直接指向修复方法。

**(3) 记录是一个普通的 dict。** `json.dumps(..., default=str)` 通过对非 JSON 安全类型调用 `str()` 来处理它们。传入一个 `Path`、一个 `datetime`，甚至一个自定义对象都不会崩溃 —— 你只会在日志中看到它的 `str()` 形式。这通常是你想从诊断日志中得到的东西。

**(4) 一把锁覆盖两个输出端。** 另一种做法（两把锁，或不加锁）在两个线程并发记录日志时可能会使字节交织。一把锁意味着要么流的那一行在下一行开始前被完整写入，要么文件的那一行被完整写入 —— 任一输出端都不会出现两行半拉子内容。

### 输出端辅助函数

```python
def _write_stream(self, line):
    if self._stream is None:
        return
    try:
        self._stream.write(line)
        self._stream.flush()
    except Exception:
        pass

def _write_file(self, line):
    if self._file is None:
        return
    try:
        with open(self._file, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass
```

两个形状相同：`if sink is None, bail; else try/except: pass`。这个 `try/except: pass` 是故意的，"永远不让智能体崩溃"的约定在实践中就是在这里落地的。如果你的日志磁盘在运行中途被写满，你会丢失日志行；但智能体会继续运行。

文件是**每次调用都打开并关闭**的，而不是一直保持打开。三个后果：
- 每次 `logger.x()` 调用后，行会立即刷新到磁盘（进程崩溃不会丢失缓冲中的行）。
- 外部工具（`logrotate`）的文件轮转能正常工作：我们不会持有一个过期的文件句柄。
- 每次调用多一次 `open()` 的微小性能损耗 —— 在人可读的日志量级下不是问题。

---

## 5. 错误与容错矩阵

| 情景 | 发生什么 |
|---|---|
| 保留 kwarg 冲突 | `ValueError`（程序员 bug，要大声失败） |
| 流写入抛异常（管道断开、stdout 被关闭） | 静默吞掉 |
| 文件父目录无法创建 | 静默吞掉；后续写入也会静默失败 |
| 文件打开/写入失败（磁盘满、无权限） | 静默吞掉 |
| JSON 序列化碰到奇怪的类型（datetime、Path、对象） | 通过 `default=str` 转成 `str()` |
| 日志级别低于阈值 | 提前返回，不构建记录 |

从日志记录器获取异常的唯一方式是保留键误用时的 ValueError。其他所有情况要么是静默的，要么是一行日志。

---

## 6. 模块 3 如何与模块 1 和 2 集成

**模块 2（配置）：** 显式，一行。
```python
logger = AgentLogger.from_profile(profile)
```
读取 `profile.id`。这是唯一的耦合。

**模块 1（LLM）：没有直接耦合。** 适配器保持纯净 —— 它们不导入日志记录器。相反，**调用方包装适配器调用**：
```python
logger.info("llm.request", "Sending", model=model, max_tokens=200)
try:
    resp = await adapter.chat(messages)
except LLMProviderError as e:
    logger.error("llm.error", "Provider failed", provider=e.provider, status_code=e.status_code)
    raise
logger.info("llm.response", "OK", stop_reason=resp.stop_reason, total_tokens=resp.usage.total_tokens)
```

为什么不在适配器内部记录日志？因为：
1. 那样适配器就需要注入一个 logger（多一个参数、多一个生命周期问题）。
2. 不同的调用方对同一次调用想要不同的详细程度（认知循环步骤记录的上下文比冒烟测试多）。
3. 这会让 `OpenAICompatibleAdapter` 知道 `AgentLogger`，仅仅为了可观测性就把模块 1 和模块 3 紧耦合起来。

"在调用方处包装"的模式正是未来认知循环模块将系统性地采用的方式。演示脚本手工展示了这一点。

---

## 7. 执行流程：`scripts/logger_demo.py`

该演示运行五个带标签的小节，覆盖日志记录器的每一项能力：

```
$ python scripts/logger_demo.py

┌─ main() (async)
│
├─ Step 1: load_profile(agents/maya_rodriguez/profile.yaml)     [Module 2]
├─ Step 2: log_file = <repo>/logs/<profile.id>.log
├─ Step 3: logger = AgentLogger.from_profile(profile, log_file=log_file)
│
├─ [section 1] demo_level_filtering(logger)
│     • debug()    ← dropped (below INFO threshold)
│     • info()     ← emitted
│     • warning()  ← emitted
│     • error()    ← emitted
│     • critical() ← emitted
│     4 lines appear on stdout AND in the log file
│
├─ [section 2] demo_structured_data(logger)
│     • info("demo.data", "...", request_id="...", retry_count=0, nested={"k":"v","list":[1,2,3]})
│     • `data` object in the emitted line carries those kwargs verbatim
│
├─ [section 3] demo_reserved_key_rejection(logger)
│     • info("...", "...", agent_id="other")  ← raises ValueError
│     • caught; printed to stdout
│     • no log line emitted (ValueError precedes record construction)
│
├─ [section 4] await demo_successful_llm_call(logger, question, system)    [Module 1]
│     • adapter = make_adapter_from_env()
│     • logger.info("llm.request", ..., model="deepseek-chat", max_tokens=160)
│     • await adapter.chat(...)   ← real HTTPS call to DeepSeek
│     • logger.info("llm.response", ..., stop_reason="end_turn", total_tokens=127)
│     • prints Maya's in-character reply
│
├─ [section 5] demo_error_path_logging(logger)
│     • raise LLMProviderError(provider="demo-stub", status_code=503, ...)
│     • caught; logger.error("llm.error", ..., provider="demo-stub", status_code=503)
│     • nothing propagates
│
└─ Tail summary
      • count total JSON lines in the file
      • print the last 3 lines so the user sees what's on disk
```

### 真实运行输出（节选）

```
[section 4] Wrapping a real LLM call with log events
------------------------------------------------------------
{"timestamp": "...", "agent_id": "student_maya_001", "level": "INFO", "event_type": "llm.request", "message": "Sending chat request", "data": {"adapter": "OpenAICompatibleAdapter", "model": "deepseek-chat", "messages_count": 1, "max_tokens": 160}}
{"timestamp": "...", "agent_id": "student_maya_001", "level": "INFO", "event_type": "llm.response", "message": "Received chat response", "data": {"stop_reason": "end_turn", "prompt_tokens": 101, "completion_tokens": 26, "total_tokens": 127, ...}}
[demo] assistant reply: I've got Data Structures and Algorithms in 20 minutes, and I'm actually pretty excited because we're covering graph traversal today.
```

这就是**三个模块协同工作**，只用了七行：
1. 模块 2 加载了 Maya 的档案。
2. 模块 3 把日志记录器绑定到她的 id，并打开了日志文件。
3. 模块 1 以她的口吻与 DeepSeek 对话。
4. 模块 3 记录了调用前后。

---

## 8. 测试覆盖图

| 文件 | 测试数 | 覆盖内容 |
|---|---|---|
| `tests/DefenseAgent/ops/test_logger.py` | 36 | 每个字段、每个级别、两个输出端、时钟注入、保留键拒绝、JSON 安全性、I/O 失败容错、`from_profile` |
| `tests/DefenseAgent/integration/test_logger_integration.py` | 2 | 带档案和日志记录器的顺利路径包装（StubAdapter）+ 错误路径包装（StubErrorAdapter） |

值得一读的亮点：
- `test_reserved_kwargs_raise_value_error[level|event_type|message]` —— 这些测试之所以存在，正是因为这些 kwargs 会与方法的仅位置参数冲突，它们证明了 `/` 标记确实起到了作用。
- `test_default_clock_produces_iso_utc_ms_with_z` —— 验证时间戳可以按设计规范文档里列出的那串精确格式解析回 `datetime`。
- `test_file_write_failure_is_swallowed` —— 使用 `monkeypatch` 把 `builtins.open` 替换成一个会抛异常的函数，然后断言日志调用正常返回。
- `test_logger_records_provider_error_without_crashing` —— 在集成测试文件里，这是最重要的一个测试：它证明即使适配器失败，logger + 适配器异常包装依然能正常工作。

所有测试完全离线运行（`io.StringIO`、`tmp_path`、`StubAdapter`）。没有 sleep，没有 flakiness，总耗时 0.07 秒。

---

## 9. 值得关注的几点

- **一个文件，没有 errors 模块。** 日志记录器总共只需要一个自定义异常：来自标准库的 `ValueError`。再加一个带 `LoggerError(Exception)` 的 `errors.py` 属于范围蔓延。
- **构造函数使用仅关键字参数。** `__init__` 里的 `*` 强制除 `agent_id` 外的所有参数都必须显式用关键字传递。永久避免了一类"哦我以为第二个参数是路径"的 bug。
- **方法用仅位置参数。** `debug/info/…/log` 里的 `/` 把"危险的 kwarg 冲突"变成了"清晰的 ValueError"。这是 Python 3.8+ 的一个值得了解的特性 —— 它是这类 API 问题最干净的修复方式。
- **每次调用都打开文件。** 更简单、更安全，且与 `logrotate` 配合良好。每条日志多一次 `open()` 系统调用，对人可读的吞吐量来说没问题。
- **`json.dumps` 中的 `default=str`。** 这就是为什么 `logger.info("...", "...", where=Path("/tmp"))` 不会崩溃。任何有有意义 `__str__` 的类型都能直接工作。
- **适配器模块保持与日志无关。** 模块 1 没有改动。日志记录器在*边界*上集成 —— 在认知循环（未来）中，或在演示脚本里 —— 而不是在适配器内部。
