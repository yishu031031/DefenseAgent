# 模块 3 — 结构化日志记录器设计规范

**日期：** 2026-04-22
**状态：** 已批准，可以开始实现
**模块位置：** 第 3 个，共 N 个。被之后每个会发出事件的模块使用（认知循环、工具执行器、记忆、MCP 客户端）。不存在运行时反向耦合到模块 1（LLM）或模块 2（config）—— 这些模块保持与日志无关；由调用方对它们进行包装。

## 目的

为该 harness 中的每个模块提供**一个**结构化日志记录设施，使得：
1. 每个事件都是一个**原子 JSON 行**，外部聚合器可以对其建立索引。
2. 每条记录都携带 **agent_id**，让一个日志文件能无歧义地承载多个智能体的事件。
3. 日志**永远不会抛异常** —— 日志记录器的失败绝不能让智能体崩溃。
4. 通过一次调用将日志记录器**绑定到一个 `AgentProfile`**，从而让下游模块无需从零散的地方重新推导 agent_id。

运行本模块会产生如下 JSON：

```json
{"timestamp": "2026-04-22T14:15:32.481Z", "agent_id": "student_maya_001", "level": "INFO", "event_type": "llm.request", "message": "Calling DeepSeek", "data": {"model": "deepseek-chat", "max_tokens": 200}}
```

## 范围

### 范围内
- `AgentLogger` 类，带五个级别方法（`debug`、`info`、`warning`、`error`、`critical`）。
- JSON 行输出到文件（可选）和文本流（默认 stdout）。
- `AgentLogger.from_profile(profile, ...)` 类方法，用于集成模块 2。
- 使用标准库 `logging` 级别常量进行级别过滤（`logging.DEBUG`、`logging.INFO`、……）。
- 可注入自定义时钟，以便在测试中获得确定性时间戳。
- 线程安全的写入路径（围绕两个 sink 写操作加锁）。
- 无外部依赖 —— 仅使用标准库（`json`、`logging`、`threading`、`pathlib`、`datetime`）。
- 针对每个行为的单元测试，外加一个 `scripts/logger_demo.py` 集成 logger + profile + LLM。

### 范围外（推迟）
- 文件轮转 / 大小限制（交给操作系统或 `logrotate` 处理）。
- 基于异步 / 队列的 handler（当前 API 是同步的；对学习规模的 harness 来说可以接受）。
- EventBus 集成 —— EventBus 尚不存在（模块 4 或更晚）。当事件总线落地时，添加一个自动日志订阅者只是一个单文件改动。
- 网络 sink（Datadog、Cloud Logging 等）。调用方可以继承或自行编写流包装器。
- 美化的控制台输出 / 着色。JSON 行以机器可读为先；操作员如果想要美化输出，可通过 `jq` 进行管道处理。
- 修改模块 1 的 adapter 以发出日志。Adapter 保持纯净；调用方在 `adapter.chat(...)` 外包裹日志调用。

## 设计

### 记录形状

每次调用恰好产生如下 JSON 对象（一行，以 `\n` 结尾）：

| Key          | 类型   | 来源                                           |
|--------------|--------|--------------------------------------------------|
| `timestamp`  | string | ISO-8601 UTC，毫秒精度，后缀 `Z`                 |
| `agent_id`   | string | 来自 `AgentLogger` 实例的值                      |
| `level`      | string | `DEBUG/INFO/WARNING/ERROR/CRITICAL` 之一         |
| `event_type` | string | 调用方提供的点分标识符（例如 `llm.request`）     |
| `message`    | string | 调用方提供的人类可读摘要                         |
| `data`       | object | 调用方提供的 kwargs；若无则为 `{}`               |

时间戳始终为 UTC，形如 `2026-04-22T14:15:32.481Z`。后缀 `Z` 符合 ISO 规范，并且比解析时区偏移更便于过滤。

### API

```python
class AgentLogger:
    def __init__(
        self,
        agent_id: str,
        *,
        log_file: str | Path | None = None,
        stream: TextIO | None = sys.stdout,
        level: int = logging.INFO,
        clock: Callable[[], datetime] | None = None,
    ): ...

    @classmethod
    def from_profile(
        cls,
        profile: AgentProfile,
        **kwargs,
    ) -> "AgentLogger": ...

    def debug   (self, event_type: str, message: str, **data) -> None: ...
    def info    (self, event_type: str, message: str, **data) -> None: ...
    def warning (self, event_type: str, message: str, **data) -> None: ...
    def error   (self, event_type: str, message: str, **data) -> None: ...
    def critical(self, event_type: str, message: str, **data) -> None: ...

    def log(self, level: int, event_type: str, message: str, **data) -> None:
        """Underlying dispatch; the five level methods delegate here."""
```

### 设计决策

**D1. 每个智能体一个实例，而非模块级别的单例。**
每个 `AgentLogger` 都携带 `agent_id`。当多个智能体在同一进程中运行时（未来的范围），每个智能体都有自己的日志记录器。下游模块通过依赖注入接收日志记录器 —— 它们绝不会伸手去拿一个全局变量。

**D2. `kwargs` 会成为 `data`；保留键会被拒绝。**
`logger.info("chat.request", "sending prompt", model="gpt-4o", tokens=200)` 会把 `{model: "gpt-4o", tokens: 200}` 放到 `data` 下面。但如果传入以下键（与顶层记录形状冲突），必须抛出 `ValueError`：
- `agent_id`、`level`、`timestamp`、`event_type`、`message`、`data`

理由：歧义比严格更糟糕。如果你真的需要把 `"message": "foo"` 作为负载记录，请嵌套： `logger.info("...", "...", payload={"message": "foo"})`。

**D3. 级别过滤在记录构造之前发生。**
如果 `level < self._level`，方法立即返回 —— 不构造字典，不生成 JSON，不获取锁，不做 I/O。让 `logger.debug(...)` 在生产环境中开销极低。

**D4. Sink：stream + 可选 file。**
- `stream` 默认是 `sys.stdout`。设为 `None` 可以静默 stdout。
- `log_file` 是可选的。设置后，每条记录还会追加到该文件（每次写入都开关一次文件 —— 简单、崩溃安全）。
- 如果两个 sink 都未配置，日志记录器即为空操作（在静默测试时仍然有用）。

**D5. 日志记录器在 I/O 失败时永远不抛出。**
如果 `stream.write` 或文件打开 / 写入失败，异常会被静默吞掉。另一种做法 —— 因为日志磁盘满了而让智能体崩溃 —— 比丢失一行日志糟糕得多。（未来增强：暴露一个 `on_error=` 回调，让调用方选择处理方式，但默认保持静默。）

**D6. 唯一会抛出的是针对保留键误用的 `ValueError`。**
那是程序员错误（调用代码的 bug），不是运行时的环境故障。它应该在开发阶段大声地暴露出来。

**D7. 时钟注入以获得确定性测试。**
`clock` 是一个可选的 `Callable[[], datetime]`。测试传入一个固定时间的 lambda，以便 `timestamp` 字段可预测。生产环境使用 `datetime.now(timezone.utc)`。

**D8. 通过围绕两次写入的一个锁实现线程安全。**
一个 `threading.Lock` 保护 stream 写入加文件写入这对操作。如果没有锁，两个线程同时调用 `info()` 可能会让字节在 stdout 中交错。

**D9. `from_profile` 工厂方法把模块 2 集成最小化。**
只需一行集成代码：
```python
@classmethod
def from_profile(cls, profile: AgentProfile, **kwargs) -> "AgentLogger":
    return cls(agent_id=profile.id, **kwargs)
```
不存在循环导入：`ops/logger.py` 仅为类型注解导入 `AgentProfile`；我们可以用 `TYPE_CHECKING` 让它甚至在运行时也不被导入。

**D10. 不修改任何 adapter。**
本模块不触碰模块 1 的 `LLMAdapter` 子类。认知循环（未来）会在 adapter 调用外围记录日志。`scripts/` 中的 logger demo 脚本演示了这一模式。

### 文件布局

```
DefenseAgent/ops/
├── __init__.py          # re-exports AgentLogger
└── logger.py            # AgentLogger + helpers

tests/DefenseAgent/ops/
├── __init__.py
└── test_logger.py       # all logger tests (stdlib-only; uses io.StringIO + tmp_path)

scripts/
└── logger_demo.py       # profile + logger + LLM demo
```

### 依赖

仅使用标准库。不需要更新 `requirements.txt`。

## 测试策略

所有测试完全在内存中或 `tmp_path` 上运行。无网络、无 sleep、无依赖时间的断言（时钟是注入的）。

覆盖大纲：

**构造**
- `AgentLogger(agent_id=...)` 会保存 agent_id。
- 默认级别是 `logging.INFO`。
- 默认流是 `sys.stdout`。

**记录形状**
- 每一行发出的内容都是合法 JSON。
- 恰好有六个顶层键： `timestamp`、`agent_id`、`level`、`event_type`、`message`、`data`。
- 当不传 kwargs 时，`data` 为 `{}`。
- `data` 原样携带 kwargs。

**级别过滤**
- 在 DEBUG 级别的 logger 上调用 `info()` 会发出。
- 在 INFO 级别的 logger 上调用 `debug()` **不会**发出。
- 每个级别方法都映射到正确的 `logging.*` 常量。

**Sink**
- 仅 stream：写入提供的 stream，不写文件。
- 仅 file：`stream=None`，`log_file=path` —— 写入文件。
- 两者都有：把同一行写到两处。
- 两者都没有（`stream=None, log_file=None`）：空操作。
- 在重复调用时，文件是追加，而非截断。
- 如果父目录不存在，会被创建。

**时钟注入**
- `clock=lambda: datetime(2026, 4, 22, 10, 15, 30, 480000, tzinfo=timezone.utc)` → `timestamp` 为 `"2026-04-22T10:15:30.480Z"`。
- 默认时钟产生带时区感知、以 `Z` 结尾的时间戳。

**保留键拒绝**
- `logger.info("e", "m", agent_id="other")` → `ValueError`。
- 对 `level`、`timestamp`、`event_type`、`message`、`data` 同样如此。

**非 JSON 安全的 data**
- 在 data 中传入 `Path` 不会崩溃 —— 序列化器使用 `default=str`。
- 在 data 中传入 `datetime` 不会崩溃 —— 序列化为其 `str()` 形式。

**I/O 失败容错**
- 某个 `write` 会抛异常的 stream，异常不会向外传播。
- 指向不可写目录的路径，异常不会向外传播。
- （这些测试使用 `monkeypatch` 注入一个会抛异常的 write 函数。）

**`from_profile`**
- `AgentLogger.from_profile(profile)` 返回一个 `agent_id == profile.id` 的 logger。
- 额外的 kwargs 会原样传入（`level`、`log_file` 等）。

**集成健全性检查（在 `tests/DefenseAgent/integration/` 中）：**
- 一个测试演练：加载 profile → `from_profile` → StubAdapter.chat → 记录 request，记录 response → 断言日志文件同时包含两个事件，且 `event_type` 与 `agent_id` 正确。

## 执行流程（一次 `logger.info(...)` 调用发生了什么）

```
logger.info("llm.request", "Calling model", model="deepseek-chat")
│
├─ log(logging.INFO, "llm.request", "Calling model", model="...")
│
├─ if logging.INFO < self._level: return          ← early exit
│
├─ check kwargs for reserved keys → raise ValueError if any
│
├─ record = {
│     "timestamp": iso8601(self._clock()),
│     "agent_id":  self.agent_id,
│     "level":     "INFO",
│     "event_type":"llm.request",
│     "message":   "Calling model",
│     "data":      {"model": "deepseek-chat"},
│  }
│
├─ line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
│
└─ with self._lock:
      try: self._stream.write(line); self._stream.flush()
      except: pass
      if self._file:
          try: append line to file
          except: pass
```

## 与更早模块的集成

- **模块 2（config）：** `AgentLogger.from_profile(profile)` 读取 `profile.id`。这是唯一的接触点。
- **模块 1（LLM）：** 保持不变。demo 脚本展示了围绕 `adapter.chat(...)` 推荐的包装模式。

## 未来扩展

当事件总线（大约模块 4 左右）到来时，添加：
- `AgentLogger.subscribe_to(bus)` —— 为总线上每个事件自动发出一行日志。
- 通过一张小的 event-type → log-level 表进行级别映射。

当真正的服务器到来时，添加：
- `NetworkSink` 子类，支持 HTTP 或 syslog 输出。
- `QueueHandler` 风格的 asyncio 包装器，让写入不会阻塞事件循环。

两者都是可选接入的；核心 API 不会改变。

## 开放问题

在设计规范批准时无。上述设计决策 D1–D10 已经确定了每一个选择。
