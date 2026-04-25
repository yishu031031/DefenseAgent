# 模块 4 — Memory（写入、存储、检索）设计规范

**日期：** 2026-04-22
**状态：** Draft, awaiting user approval
**模块位置：** 4 / N。认知循环的基础，每一步都会读写记忆。

> **修订说明（2026-04-24）：SQLite 持久化。**
> 最初的设计规范将持久化推迟到暂停/恢复时才引入；随着我们采用按智能体（per-agent）的 bundle，这一时机已经到来。`MemoryStream` 现在接受一个可选的 `db_path` —— 当提供该参数时，每次 `add()` 会向 `<db_path>` 写入一行（SQLite、WAL、每条记录提交），而每次 `__init__` 都会从已有记录中回水（rehydrate）。`Memory.from_profile(profile)` 是新的门面入口：它会自动解析 `<profile.source_dir>/memory/stream.db`，因此每个智能体在自己的档案旁都会拥有独立的数据库。向量嵌入通过 `numpy.float32.tobytes()` / `numpy.frombuffer()` 进行序列化；BM25 仍然只驻留在内存中，并在打开时通过一次表扫描重建。辅助脚本 `scripts/dump_memory.py` 会按时间顺序打印流中的内容（不含嵌入）用于调试。原设计规范中"无持久化"一行已被覆盖。

## 目的

给予整个 harness 一份**可查询的智能体经历记录**，并配备现代化的检索流水线。只有三件事，别无其他：

1. **写入** —— 对一条新的 observation/fact/plan/preference/reflection 进行嵌入、去重检查、打时间戳，并返回一个结构化的 `MemoryRecord`。
2. **存储** —— 一个只追加的记录流，加上一个增量更新的倒排索引（用于 BM25）。
3. **检索** —— 给定一个查询，使用**混合检索**返回 top-k 记忆：密集嵌入 + 稀疏 BM25，通过倒数排名融合（RRF）合并，然后按照智能体档案中的 recency × importance × 按 kind 区分的规则进行加权。

本模块并**不**字面复刻 2023 年 Generative Agents 的公式。它保留了**三条打分轴**（recency、importance、relevance），因为这些确实是有价值的认知信号，但把朴素的余弦相似度替换为如今业界标准的混合检索器。

**本模块明确不涵盖的内容：**
- **Reflection**（由 LLM 将近期记忆综合成更高层次洞见）—— 由模块 5 负责。
- **基于 LLM 的 importance 打分** —— 模块 5 会加入。当前由调用方手动提供 importance；默认值为 5.0。
- **重排器（reranker）阶段**（在 top-K 之上的 cross-encoder）—— 当检索质量成为瓶颈时才有用；放到模块 5 或更晚。
- **文件持久化**（JSON/SQLite 转储与加载）—— 推迟到需要暂停/恢复时再做。
- **MemGPT 风格的自编辑分层** —— 需要工具调用；放在模块 6+。
- **通过 LLM 从原始 observation 中抽取事实**（Mem0 风格）—— 跨入了模块 1/5 的领域。

## 塑造本设计的四个核心理念

1. **记忆是只追加的。** 记录一旦创建就不可变。反思与更新都是*新*记录，从不编辑旧记录。这样保持存储/检索简单，也让时间旅行变得轻而易举。

2. **检索是质量的杠杆。** 当记忆多达数千条时，为 LLM 的上下文挑选约 10 条，正是质量的来源。检索器也是本模块代码量最大的部分。

3. **混合检索胜过纯余弦。** 密集嵌入擅长同义改写，但对偏关键词的查询（"我之前怎么说 Greg 的？"）效果不佳。BM25 恰好能接住这类查询。RRF 融合两种排名时不需要校准排序器之间的分数尺度。这是 2025 年标准的生产实践。

4. **记忆的 kind 会改变行为。** 并不是每条记录都是事件。事实（"我对花生过敏"）和偏好（"我讨厌早上的课"）是*稳定*的；对它们应用 recency 衰减就错了。计划有状态。检索器会根据 `kind` 分支应用正确的规则。

## 设计

### 数据模型

```python
# DefenseAgent/memory/types.py

MemoryKind = Literal[
    "observation",   # raw event ("attended the 9 AM lecture")            — decays with time
    "fact",          # stable proposition ("I'm in my second year")        — no recency decay
    "preference",    # persistent disposition ("hate morning classes")     — no recency decay
    "plan",          # intent ("study graphs tonight")                     — filtered by metadata["status"]
    "reflection",    # synthesized insight (Module 5 produces these)       — decays like observations
]

@dataclass
class MemoryRecord:
    id: str                    # UUID4 hex, assigned at creation
    content: str               # human-readable text
    timestamp: datetime        # UTC, millisecond precision
    kind: MemoryKind           # determines retrieval behavior
    importance: float          # 1.0–10.0, caller-supplied (Module 5 adds LLM scoring)
    embedding: list[float]     # non-empty
    metadata: dict             # freeform; plans use metadata["status"] = "active" | "done"

@dataclass
class ScoredMemory:
    """A memory record with its retrieval score components — returned by MemoryRetriever."""
    record: MemoryRecord
    score: float                # composite weighted score
    recency_score: float        # [0, 1]; 1.0 for facts/preferences (no decay)
    importance_score: float     # [0.1, 1.0]
    relevance_score: float      # [0, 1] — normalized hybrid RRF score
    dense_rank: int | None      # 1-indexed rank in dense ranking; None if not in top-candidates
    sparse_rank: int | None     # 1-indexed rank in BM25 ranking; None if missing
```

`ScoredMemory` 暴露了每一个检索组件，包括每个排序器的 rank，因此调用方和测试能准确理解某条记忆**为什么**排在某个位置。可观测性优先。

### `.env` 块（已于 2026-04-22 添加）

```dotenv
EMBEDDING_PROVIDER=qwen                 # one of: openai, qwen, vllm
EMBEDDING_API_KEY=                      # blank until populated
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v3
```

校验（工厂层面）：
- `EMBEDDING_PROVIDER` 为空或不在支持集合中 → `EmbeddingConfigError`。
- `EMBEDDING_MODEL` 为空 → `EmbeddingConfigError`。
- 对 `qwen` 或 `vllm`，`EMBEDDING_BASE_URL` 为空 → 报错。对 `openai` 为空 → OK（使用 SDK 默认值）。
- 对 `openai`/`qwen`，`EMBEDDING_API_KEY` 为空 → 报错。对 `vllm` 为空 → 默认为 `"token-not-needed"`。

### 嵌入适配器（Embedding adapter）

```python
# DefenseAgent/memory/embedding_adapter.py

class EmbeddingAdapter(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class OpenAICompatibleEmbeddingAdapter(EmbeddingAdapter):
    """Covers OpenAI, Qwen (DashScope), vLLM — they all speak /embeddings."""
    def __init__(self, *, api_key: str, base_url: str, model: str, client=None): ...


def make_embedding_adapter_from_env(dotenv_path=None, *, load_env=True) -> EmbeddingAdapter:
    """Read EMBEDDING_* env block and build the configured adapter."""
```

**设计决策：**
- **客户端注入式的测试缝（test seam）**，与模块 1 的 chat 适配器完全一致。测试传入 `MagicMock`；生产中则延迟创建 `AsyncOpenAI`。
- 本轮**不为 `EMBEDDING_*` 设置覆盖层级**。只有一个块。后续如有需要再加。
- **维度不校验。** 不同模型返回的维度不同；流在记录时隐式地保存了 `len(embedding)`。在同一个流里混用不同维度会破坏余弦计算 —— 每个流使用单一嵌入模型是调用方的责任。

### BM25 索引（纯 Python）

```python
# DefenseAgent/memory/bm25.py

class BM25Index:
    """Okapi BM25 over the content of memory records. Pure Python, ~40 lines core.

    k1 = 1.5, b = 0.75 (standard defaults).
    """
    def __init__(self, *, k1: float = 1.5, b: float = 0.75): ...

    def add(self, doc_id: str, text: str) -> None:
        """Tokenize `text`, update per-document term counts, update corpus stats."""

    def score(self, query: str) -> dict[str, float]:
        """Return {doc_id: bm25_score} for every doc (includes zeros for non-matchers)."""

    def __len__(self) -> int: ...

def tokenize(text: str) -> list[str]:
    """Lowercase + split on non-alnum + strip tiny tokens. No stemming for this pass."""
```

**为什么选择纯 Python 而不是 `rank-bm25`：**
1. 教学价值 —— BM25 是一个短小、经典的公式，值得一读。
2. 零新增依赖。
3. 增量更新的语义（添加一次只加一条）很直观。

分词刻意做得简单：转小写，按非字母数字分割，丢弃单字符 token。没有词干化、没有停用词、没有语种检测。现代 BM25 变体会加入这些，但我们目前还不需要。

和索引一起维护的语料统计信息：
- `doc_count` —— 文档总数。
- `avg_doc_len` —— 每篇文档 token 数的滑动平均。
- `df[term]` —— 每个 term 的文档频率。
- `idf[term]` —— 在 score() 时基于 `df[term]` 和 `doc_count` 惰性重算。

### `MemoryStream`

```python
# DefenseAgent/memory/stream.py

class MemoryStream:
    def __init__(
        self,
        embedding_adapter: EmbeddingAdapter,
        *,
        clock: Callable[[], datetime] | None = None,
        dedup_threshold: float | None = 0.95,   # cosine threshold; None disables
    ): ...

    async def add(
        self,
        content: str,
        *,
        kind: MemoryKind = "observation",
        importance: float = 5.0,
        metadata: dict | None = None,
    ) -> MemoryRecord:
        """Embed, dedup-check, timestamp, append, return the record.

        If dedup_threshold is set and a memory of the same kind has cosine
        similarity > threshold with the new embedding, the existing record is
        returned unchanged and nothing is added. The caller can detect dedup
        by comparing `returned.id` against the record's previous absence.
        """

    def add_record(self, record: MemoryRecord) -> None:
        """Append a pre-built record (e.g. from a future persistence layer).
        Also updates the BM25 index. Bypasses dedup check by design.
        """

    def get_recent(self, n: int) -> list[MemoryRecord]: ...
    def get_all(self) -> list[MemoryRecord]: ...
    def get_by_id(self, record_id: str) -> MemoryRecord: ...        # raises MemoryNotFoundError
    def __len__(self) -> int: ...

    # Internal: the BM25Index lives here so adds can update it.
    # Exposed via a read-only property for the retriever:
    @property
    def bm25(self) -> BM25Index: ...
```

**设计决策：**
- **`add()` 是 async**（需要调用嵌入）；**`add_record()` 是同步的**（不需要嵌入）。
- **只在同 kind 内部去重。** 一条 observation 和一条内容相近的 fact 在语义上不同；跨 kind 去重会把它们混为一谈。
- **去重时返回已有记录。** 不报错，不提升 importance，也不给消息。真正关心的调用方可以通过 `record.timestamp != now` 来判断是否触发了去重。
- **BM25 索引由 stream 持有**，在 `add()` / `add_record()` 时增量更新。检索器来查询它。

### `MemoryRetriever` —— 混合检索

```python
# DefenseAgent/memory/retriever.py

RRF_K = 60   # Cormack et al. (2009), the accepted default for RRF

class MemoryRetriever:
    def __init__(
        self,
        stream: MemoryStream,
        embedding_adapter: EmbeddingAdapter,
        memory_config: MemoryConfig,                            # from Module 2
        *,
        clock: Callable[[], datetime] | None = None,
        recency_half_life_hours: float = 24.0,
    ): ...

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,                                # defaults to memory_config.retrieval_top_k
        kinds: tuple[MemoryKind, ...] | None = None,             # None = all kinds
    ) -> list[ScoredMemory]:
        """Hybrid-retrieve top-k memories for `query`."""
```

#### 检索流水线

```
retrieve(query, top_k, kinds)
│
├─ Step 1: candidate set
│     records = stream.get_all()
│     • filter out plans with metadata["status"] == "done"
│     • if `kinds` is provided, filter records by kind
│
├─ Step 2: dense ranking
│     query_emb = embedding_adapter.embed(query)
│     for each record: cosine = cos_sim(query_emb, record.embedding); clip to [0, 1]
│     rank records by cosine descending → `dense_rank[record.id]`
│
├─ Step 3: sparse ranking
│     scores = stream.bm25.score(query)              # one scan
│     rank records by scores[record.id] desc → `sparse_rank[record.id]`
│
├─ Step 4: RRF fusion
│     raw_rrf[r.id] = 1/(RRF_K + dense_rank[r.id]) + 1/(RRF_K + sparse_rank[r.id])
│     rrf_max = max(raw_rrf.values())
│     hybrid_relevance[r.id] = raw_rrf[r.id] / rrf_max   # normalized to [0, 1]
│
├─ Step 5: per-kind adjustment
│     recency_score = match kind:
│         "fact", "preference" → 1.0                                            # no decay
│         else                 → 2 ** (-age_hours / half_life)                  # exponential
│     importance_score = record.importance / 10.0
│
├─ Step 6: composite
│     final[r.id] = (
│         w_recency    * recency_score +
│         w_importance * importance_score +
│         w_relevance  * hybrid_relevance
│     )
│     (weights from profile.memory.{recency,importance,relevance}_weight)
│
├─ Step 7: sort by final desc, truncate to top_k
│
└─ return list[ScoredMemory]
```

#### 为什么偏偏选 RRF

倒数排名融合（Reciprocal Rank Fusion，Cormack、Clarke、Büttcher，SIGIR 2009）解决了一个真实的问题：密集的 cosine 分数与 BM25 分数处于彼此不可比较的尺度上。把它们各自独立归一化到 [0, 1] 对离群值非常敏感。RRF 完全忽略分数，只基于 rank 工作：

$$\text{rrf}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + \text{rank}_r(d)}$$

取 `k = 60`，在两个排序器里都排第 1 的文档得分约为 `1/61 + 1/61 ≈ 0.033`；在两个排序器里都排第 100 的文档约为 `2/160 ≈ 0.013`。稳健、单调、不需要校准。然后我们把 RRF 值 min-max 归一化到 [0, 1]，这样它就能干净地与 recency 和 importance 分数组合起来。

#### 按 kind 的检索规则（概要）

| Kind          | 是否 Recency 衰减？    | 是否状态过滤？           | Demo 中的默认 importance   |
|---------------|------------------------|--------------------------|----------------------------|
| observation   | 是（指数衰减，半衰期 24h） | —                        | 5                          |
| fact          | **否**（recency = 1.0）  | —                        | 6–9                        |
| preference    | **否**（recency = 1.0）  | —                        | 7–9                        |
| plan          | 是                     | **若为 "done" 则过滤掉** | 6–8                        |
| reflection    | 是                     | —                        | 7–10                       |

#### 余弦相似度

纯 Python，仅使用标准库 math：

```python
def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b): raise MemoryError("embedding dimension mismatch")
    dot = sum(x*y for x, y in zip(a, b))
    na  = math.sqrt(sum(x*x for x in a)) or 1e-12
    nb  = math.sqrt(sum(x*x for x in b)) or 1e-12
    return dot / (na * nb)
```

对于 1024 维 × 每次查询几百条记忆，运行时间远低于 50ms。如果未来需要更高性能，代码已经可以直接替换为 numpy 实现。

### 错误

```python
# DefenseAgent/memory/errors.py

class MemoryError(Exception): ...                  # base
class EmbeddingConfigError(MemoryError): ...       # .env misconfigured
class EmbeddingProviderError(MemoryError): ...     # provider API error; __cause__ preserved
class MemoryNotFoundError(MemoryError): ...        # get_by_id() miss
```

## 文件布局

> **备注（2026-04-24）：** 下方原本的 7 文件布局已被合并为 5 个文件 —— 当前结构请见 [walkthrough](../../walkthroughs_ZN/module-04-memory.md#2-directory-map)。公共 API（从 `DefenseAgent.memory` 导出）保持不变。

```
DefenseAgent/memory/
├── __init__.py                  # re-exports public API
├── types.py                     # MemoryKind, MemoryRecord, ScoredMemory
├── errors.py                    # MemoryError + 3 subclasses
├── embedding_adapter.py         # EmbeddingAdapter + OpenAICompatibleEmbeddingAdapter + factory
├── bm25.py                      # BM25Index + tokenize()
├── stream.py                    # MemoryStream (owns embedding adapter + BM25 index, does dedup)
└── retriever.py                 # MemoryRetriever + RRF + cosine + per-kind scoring

tests/DefenseAgent/memory/
├── __init__.py
├── test_types.py
├── test_errors.py
├── test_embedding_adapter.py    # mocked openai client
├── test_bm25.py                 # tokenizer, IDF, score ordering, edge cases
├── test_stream.py               # stub embedding adapter, deterministic clock, dedup cases
└── test_retriever.py            # full hybrid pipeline on hand-built records

tests/DefenseAgent/integration/
└── test_memory_integration.py   # load profile → build stream → add records → retrieve with real MemoryConfig

scripts/
└── memory_demo.py               # day-in-Maya's-life observations + facts + prefs + plans, several queries
```

## 依赖

**没有新增的运行时依赖。**

- `openai` SDK —— 已经存在（嵌入适配器使用它）。
- `stdlib math`、`uuid`、`datetime` —— 已在标准库中。
- BM25 —— 纯 Python，无包。

`requirements.txt` 保持不变。

## 与之前模块的集成

| 模块 | 如何使用 | 是否修改该模块？ |
|---|---|---|
| 模块 1（`llm`） | **在模块 4 中完全不使用。** 模块 5 会包装 `LLMAdapter.chat()` 用于 importance 打分和 reflection。 | 否 |
| 模块 2（`config`） | `MemoryRetriever` 读取 `MemoryConfig.{retrieval_top_k, recency_weight, importance_weight, relevance_weight}`。 | 否 |
| 模块 3（`ops`） | 可选：调用方可以把 `stream.add()` / `retriever.retrieve()` 包在 `logger.info("memory.added", ...)` 调用里。 | 否 |

零返工。

## 测试策略

所有测试都离线进行。关键的测试缝：

- **嵌入适配器**的测试与模块 1 一样使用 openai 客户端注入缝。
- **Stream / retriever / integration** 测试注入一个 `StubEmbeddingAdapter`，它返回预先指定的向量，从而让 cosine 的结果可以手动验证。
- **BM25** 测试是纯确定性的（无随机性、无外部调用）。
- 任何出现时间戳或年龄的地方，**Clock** 都被注入。

### 覆盖率概览

**`test_types.py`** —— 字段默认值、每个 kind 字面量都被接受、dataclass 往返。

**`test_errors.py`** —— 层级结构、`__cause__` 保留。

**`test_embedding_adapter.py`** —— `embed()` / `embed_batch()` 的 happy path、`.env` 各个错误分支、provider 异常包装、`embed` 和 `embed_batch` 都能正确翻译请求/响应。

**`test_bm25.py`** ——
- `tokenize` 丢弃极短 token 并按非字母数字分割。
- 单文档语料：BM25 分数随 term frequency 单调。
- 多文档语料：稀有 term 的 IDF 更高。
- 没有匹配的查询：全零。
- `len()` 反映文档数。

**`test_stream.py`** ——
- `add()` 生成嵌入并追加。
- 时间戳来自注入的 clock。
- `get_recent(n)` 按最新优先返回。
- `get_by_id` 命中与未命中。
- `add_record` 跳过去重。
- 去重：同 kind 的近似重复不添加；向量相距较远的会被添加；*不同* kind 的重复*会*被添加。
- `dedup_threshold=None` 禁用去重。

**`test_retriever.py`** ——
- Dense-only 胜出（BM25 rank 打平）：dense 排名胜出。
- Sparse-only 胜出（嵌入相同，tokens 不同）：BM25 胜出。
- 手工构造 rank 上的 RRF 融合正确性。
- Recency-weight 占主导时，较新的排名在较旧之前。
- Importance-weight 占主导时，较高 importance 排在较低之前。
- Facts 跳过 recency 衰减。
- Preferences 跳过 recency 衰减。
- `metadata["status"] == "done"` 的 plans 被排除。
- `kinds` 过滤只返回请求的 kinds。
- `top_k` 的默认值来自 `MemoryConfig.retrieval_top_k`。
- 空流返回 `[]` 而不报错。
- `ScoredMemory` 暴露了预期的组件字段。

**`test_memory_integration.py`**（位于 `tests/DefenseAgent/integration/`）—— 加载 Maya 的档案，基于她的 `MemoryConfig` 构建 stream + retriever，添加一批手写的、覆盖全部五种 kind 的记忆，然后验证对某个特定查询检索会挑选哪些记忆，且排名遵循她的权重。

任何测试中都没有真实的 API 调用。

## 执行流程

### 写入一条记忆（一次 `await stream.add(...)` 调用）

```
stream.add(
    "I attended the 9 AM data structures lecture.",
    kind="observation",
    importance=6,
)
│
├─ 1. query_emb = embedding_adapter.embed(content)
│     → HTTPS POST to provider's /embeddings (e.g. 1024-float vector for text-embedding-v3)
│
├─ 2. dedup check
│     for each existing record with same kind:
│         if cosine(query_emb, record.embedding) > dedup_threshold:
│             return record           # ← no new record; caller sees existing one
│
├─ 3. build MemoryRecord
│     record = MemoryRecord(
│         id=uuid4().hex,
│         content=content,
│         timestamp=clock(),
│         kind=kind,
│         importance=importance,
│         embedding=query_emb,
│         metadata=metadata or {},
│     )
│
├─ 4. append
│     self._records.append(record)
│     self._bm25.add(record.id, content)     # update BM25 index
│
└─ return record
```

### 检索（一次 `await retriever.retrieve(query)` 调用）

```
retriever.retrieve("how am I doing on the homework?", top_k=5)
│
├─ 1. candidates = [r for r in stream if not (r.kind == "plan" and r.metadata.get("status") == "done")]
│     (optionally filter by `kinds` parameter)
│
├─ 2. query_emb = embedding_adapter.embed(query)          # ONE embedding call per retrieval
│
├─ 3. dense_ranks  = rank candidates by cosine(query_emb, r.embedding) desc
│     sparse_ranks = rank candidates by stream.bm25.score(query)        desc
│
├─ 4. for each candidate r:
│         raw_rrf[r] = 1/(60 + dense_rank[r]) + 1/(60 + sparse_rank[r])
│     hybrid_relevance[r] = raw_rrf[r] / max(raw_rrf)
│
├─ 5. for each candidate r:
│         recency    = 1.0 if r.kind in ("fact", "preference") else 2**(-age_hours/24)
│         importance = r.importance / 10
│         final      = w_r*recency + w_i*importance + w_v*hybrid_relevance[r]
│
├─ 6. sort candidates by final desc, take top_k
│
└─ return [ScoredMemory(record=r, score=final, recency_score=..., ...) for r in top_k]
```

### 失败模式

| 情况 | 行为 |
|---|---|
| `.env` 缺少 `EMBEDDING_PROVIDER` | 工厂抛出 `EmbeddingConfigError` |
| provider 在 embed 时返回 429/500 | 适配器抛出 `EmbeddingProviderError`；传播到 `add()` 或 `retrieve()` |
| `get_by_id("bogus")` | 抛出 `MemoryNotFoundError` |
| 嵌入维度不匹配（用户在流中途换了模型） | `cosine()` 抛出 `MemoryError` 并给出清晰的诊断 |
| 空流 | `retrieve()` 返回 `[]` 而不报错 |
| 重复 `add()`（高于阈值） | 返回已有记录；不新增、不报错 |
| 查询没有命中任何 BM25 term | sparse rank 全部打平；RRF 仍然通过 dense 排名工作 |

## 未来扩展（模块 5+）

- **反思引擎。** `ReflectionEngine(stream, llm_adapter, retriever, profile)` —— 负责调度、提示、将 reflection 类型的记录综合出来，importance 取自 LLM 的置信度。
- **基于 LLM 的 importance 打分。** `score_importance(text, llm_adapter)` —— 薄薄的 `chat()` 包装。调用方将结果喂给 `stream.add(..., importance=score)`。
- **重排器。** 可选的 cross-encoder 或 API 重排器，对混合阶段得到的 top-30 候选进行重排，再裁到 top-k。当质量瓶颈在检索时，质量提升巨大。
- **持久化。** `save_to(path)` / `load_from(path)` —— JSON lines。记录序列化干净（标准库友好的类型）。
- **实体索引。** 在写入时抽取实体、单独建索引，启用"所有提到 X 的记忆"这种快速路径。
- **事实抽取（Mem0 风格）。** 用 LLM 从原始 observation 中抽取原子事实；把事实存下来，替代或补充原始事件。
- **睡眠时整合（sleep-time consolidation）。** 后台任务将每日 observation 重新总结为更高层次的事实和反思。

## Token 预算检索（2026-04-22 下午修订）

`MemoryConfig.max_working_memory_tokens`（已在模块 2 中声明，默认 4000）
现在是 `retrieve()` 的**第二个绑定约束**，与
`retrieval_top_k` 并列。此前，它是一个空置的配置项。

### 工作方式

在检索器对复合打分列表排序后：

1. 按 rank 顺序遍历列表。
2. 用 `_estimate_tokens(content)` 估算每条记录的 token 成本 ——
   这是一个廉价启发式（`max(1, len(content) // 4)` ≈ 英语类文本的 OpenAI tokens；
   不引入 `tiktoken` 依赖）。
3. 在 `running_tokens + cost <= budget` 且条数仍低于 `top_k` 时累积记录。
4. **例外：** 排名第一的那条总是会被纳入，即便它单条就超出预算。
   紧张的预算永远不应让一个非空候选集产出空结果。

`top_k` 和 `max_working_memory_tokens` 中先被触发的那一个会决定返回长度。

### 为什么用简单启发式而不是 tiktoken

用户在本轮的指令是："保持这个初版轻量"。一个正经的分词器：
- 新增了运行时依赖（`tiktoken` 有 2 MB + 模型下载）
- 把我们绑定到 OpenAI 风格的分词器上，而 Anthropic/Google/Qwen
  并不共享它
- 对于 ~20% 误差区间就已够用的排序决策，提供了过于精确的计数

`len // 4` 规则：
- 无依赖
- 对英语大约 ~85% 准确，方差很低
- 足以胜任"别溢出 LLM 的上下文窗口"

需要精确计数的调用方，可以使用自己 LLM 所用的分词器，在检索器之前做一层预过滤。

### 这解锁了什么

当调用方把检索到的记忆喂给 LLM 上下文时，现在可以通过
`profile.memory.max_working_memory_tokens` 指定每次查询的上下文预算，
并信赖检索器返回一个放得下的集合。此前唯一的上限是 `retrieval_top_k`，
它对内容长度一无所知。

## 未来扩展（模块 6+）

- **反思引擎。** 已落地于模块 5（`Reflector`）。
- **基于 LLM 的 importance 打分。** 已落地于模块 5（`Reflector.score_importance`）。
- **重排器。** 可选的 cross-encoder 或 API 重排器，对混合阶段的 top-30 候选重排，再裁到 top-k。当检索是瓶颈时，质量提升巨大。
- **持久化。** `save_to(path)` / `load_from(path)` —— JSON lines。记录能干净地序列化。
- **实体索引。** 在写入时抽取实体、单独索引，启用"所有提到 X 的记忆"快速路径。
- **分层记忆（MemGPT 风格）。** 具备自编辑能力的 core/archival 二层分层。这是一个独立的未来模块 —— 而非对本模块的小幅扩展。
- **精确分词器。** 基于 `tiktoken` 的计数器，作为 `_estimate_tokens` 的可选替代。

## 尚未解决的问题

在用户于 2026-04-22 的决策之后，已无遗留问题：
- 三项升级（hybrid retrieval / typed schema / dedup）全部采纳。
- BM25 以纯 Python 实现，无 `rank-bm25` 依赖。
- 仅保留混合检索 —— 不额外提供"经典"检索器作为对比。
- `.env` 块已添加，并填入 Qwen/DashScope 的占位值；provider 可以在不更新设计规范的情况下后续切换。
- Reflection 和基于 LLM 的 importance 打分明确放在模块 5（现已构建）。
- 文件持久化延后。
- `max_working_memory_tokens` 已接入为 retrieve() 的第二个绑定约束（2026-04-22 下午新增）；分词器采用 `len // 4` 的启发式而非 `tiktoken`。
