# 模块 4 代码导览 —— Memory

> 配套[设计规范](../superpowers/specs_ZN/2026-04-22-module-04-memory-design.md)。规范记录了我们决定了**什么**以及**为什么**；本代码导览解释经过 2026-04-24 整合为五个文件后，代码**如何**实现这些决定。

---

## 核心类：`Memory`

从这里开始。该模块唯一的公开入口：

```python
from DefenseAgent.config import AgentProfile
from DefenseAgent.memory import Memory

profile = AgentProfile.from_yaml("agents/maya_rodriguez/profile.yaml")
memory  = Memory.from_env(profile)

await memory.remember("I attended the 9 AM lecture.", importance=6)
results = await memory.recall("how's class going?")
```

`Memory` 拥有三个协作者。它们是公开属性，以便测试和高级调用者能绕过门面类直接访问：

- `memory.embedding_adapter` —— 将文本转换为向量
- `memory.stream` —— 仅追加存储 + BM25 索引 + 去重
- `memory.retriever` —— 混合检索流水线（稠密 + 稀疏 + RRF）

---

## 1. 本模块要解决的问题

智能体在一次运行中会累积成千上万条观察、事实、计划和反思。有两个问题必须回答得漂亮：

1. **写入** —— 我们如何记录一次新的经历，以便后续检索能够找到它？
2. **检索** —— 给定一个查询，**此刻**应该将哪大约 10 条记录放入 LLM 的上下文窗口？

对于问题 2，仅靠普通的余弦相似度是不够的：

- 一条语义上相近但过时的记忆，可能不如 10 分钟前那条略微不那么相关的记忆重要。
- 以关键词为主的查询（"我关于 Greg 说过什么？"）在稠密嵌入上得分很低。
- 事实（"我对花生过敏"）不应像事件那样衰减。

**模块 4 为整个 harness 提供了：**

- 具有 **5 种 kind 值**（`observation / fact / preference / plan / reflection`）的类型化记录。
- 在写入时带有**近重复抑制**的仅追加存储。
- **混合检索** —— 稠密嵌入 + BM25 + 倒数排名融合（RRF）。
- **按 kind 区分的检索规则** —— 事实绕过时效衰减、已完成的计划被过滤掉等等。
- 一个拥有上述所有功能的单一门面类（`Memory`）。

---

## 2. 目录结构

```
DefenseAgent/memory/                   # 6 files, one concern per file
├── __init__.py                         # re-exports the public API
├── memory.py                           # Memory facade + .from_env + .from_profile   ← START HERE
├── stream.py                           # storage layer:
│                                       #   • MemoryKind, MemoryRecord
│                                       #   • BM25Index + tokenize()
│                                       #   • cosine()
│                                       #   • MemoryStream (add / get / dedup / optional SQLite persistence)
├── retriever.py                        # retrieval layer:
│                                       #   • ScoredMemory
│                                       #   • estimate_tokens, ranks_descending
│                                       #   • RRF_K, _NO_DECAY_KINDS
│                                       #   • MemoryRetriever
├── sqlite_store.py                     # persistence (added 2026-04-24):
│                                       #   • schema + pragmas
│                                       #   • write_record / load_records
│                                       #   • numpy float32 blob serialization
└── embedding.py                        # external I/O:
                                        #   • MemoryError + 3 subclasses
                                        #   • EmbeddingAdapter (ABC)
                                        #   • OpenAICompatibleEmbeddingAdapter

tests/DefenseAgent/memory/
├── __init__.py
├── test_memory.py                      # facade + stream + retriever + cosine
├── test_bm25.py                        # BM25 index + tokenizer
├── test_embedding.py                   # embedding adapter
└── test_persistence.py                 # SQLite round-trip + from_profile (12 tests)

scripts/
├── memory_demo.py                      # seeds Maya with 14 memories, runs 3 queries
└── dump_memory.py                      # pretty-prints a stream.db chronologically (no embeddings)
```

依赖方向是单向的，没有循环：

```
embedding.py  ←  stream.py  ←  retriever.py  ←  memory.py  ←  __init__.py
```

`embedding.py` 位于最底层，这就是 `MemoryError` 基类放在那里的原因 —— `stream.py` 需要抛出 `MemoryNotFoundError`（一个子类），而 `embedding.py` 恰好是 `stream` 已经导入的那一个文件。

---

## 3. 塑造该设计的四个核心思想

### 3.1 记忆是仅追加的

记录一经创建便不可变。反思（模块 5）作为**新**记录存储，`kind="reflection"`，绝不作为对旧观察的编辑。没有一致性规则、没有迁移，时光回溯也很简单（只需按 `timestamp < T` 过滤）。

### 3.2 检索才是质量杠杆

为 LLM 的上下文挑选出大约 10 条记录，这里才是智能体智能的来源。该模块的大部分代码都在检索器中，而规范里大部分设计决策都围绕着评分。

### 3.3 混合检索胜过纯余弦

稠密嵌入擅长同义改写，但对关键词为主的查询力不从心。BM25 正好相反。我们的检索器同时运行两者，并用**倒数排名融合**（Cormack 等，2009）来融合：

$$\text{raw\_rrf}(m) = \frac{1}{60 + \text{dense\_rank}(m)} + \frac{1}{60 + \text{sparse\_rank}(m)}$$

然后我们将融合后的分数归一化到 `[0, 1]`，并乘以档案中的 `relevance_weight`。不需要对分数尺度进行校准 —— 排名是无量纲的。

### 3.4 记忆的 kind 会改变检索行为

| Kind | 时效衰减？ | 何时被过滤掉 | 示例 |
|---|---|---|---|
| `observation` | 是（指数衰减，半衰期 24 小时） | 从不 | "attended the 9 AM lecture" |
| `reflection` | 是 | 从不 | "I learn faster when stuck" |
| `plan` | 是 | `metadata["status"] == "done"` | "finish homework by Friday" |
| `fact` | **否**（始终为 1.0） | 从不 | "I'm a second-year CS major" |
| `preference` | **否**（始终为 1.0） | 从不 | "I hate 8 AM classes" |

事实和偏好是*稳定的*；把它们当作事件来处理会让它们逐渐淡出，这是错的。

---

## 4. 文件：`embedding.py` —— 错误 + 向量嵌入 I/O

### 4.1 错误层级

```python
class MemoryError(Exception): ...                 # base
class MemoryNotFoundError(MemoryError): ...       # get_by_id() miss
class EmbeddingConfigError(MemoryError): ...      # EMBEDDING_* env misconfigured
class EmbeddingProviderError(MemoryError): ...    # provider API error; __cause__ preserved
```

四个类都在这里，一起放在 `embedding.py` 的顶部。调用方可以捕获 `MemoryError` 来处理记忆模块抛出的所有异常。

### 4.2 `EmbeddingAdapter` —— 抽象基类

```python
class EmbeddingAdapter(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...
    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
```

两个方法的接口。同一文件中的具体实现 `OpenAICompatibleEmbeddingAdapter` 覆盖了 OpenAI、Qwen（DashScope）和 vLLM —— 它们都说 OpenAI 的 `/embeddings` 线上协议。

### 4.3 `OpenAICompatibleEmbeddingAdapter`

线上层。构造时接收 `api_key`、`base_url`、`model`，也可选传入一个预构建的客户端（测试接缝）。`embed()` 和 `embed_batch()` 都会把 provider 故障包装为 `EmbeddingProviderError`，并把原始异常链接为 `__cause__`。`embed_batch()` 还会按 `index` 字段重新排序 provider 的响应，以保证输入与输出始终对齐。

---

## 5. 文件：`stream.py` —— 类型 + 存储

### 5.1 `MemoryKind` 与 `MemoryRecord`

```python
MemoryKind = Literal["observation", "fact", "preference", "plan", "reflection"]

@dataclass
class MemoryRecord:
    id: str                    # UUID4 hex, assigned at creation
    content: str
    timestamp: datetime        # UTC
    kind: MemoryKind           # drives retrieval behavior
    importance: float          # 1.0–10.0
    embedding: list[float]
    metadata: dict             # plans use {"status": "active" | "done"}
```

### 5.2 `cosine()` —— 相似度辅助函数

```python
def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise MemoryError(f"embedding dimension mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
```

仪式感都在两个边界情况上：

- **维度不匹配** → 硬抛异常。意味着调用者在一个流中混用了不同的嵌入模型；静默地对垃圾数据打分比直接崩溃更糟糕。
- **零长度向量** → 返回 `0.0`，而不是 NaN。

被 `MemoryStream.add()`（去重检查）和 `MemoryRetriever.retrieve()`（稠密排名）共同使用。

### 5.3 `BM25Index` + `tokenize()` —— 稀疏索引

分词刻意保持简单：

```python
_TOKEN_RE = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2]
```

小写化 + 仅保留字母数字 + 丢弃单字符 token。不做词干化，也没有停用词表 —— 对于类英文的智能体观察而言足矣。

公式（标准 Okapi BM25，`k1=1.5`，`b=0.75`）：

```
idf(t)       = log( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
score(t, d)  = idf(t) * tf(t, d) * (k1 + 1)
             / ( tf(t, d) + k1 * (1 - b + b * |d| / avgdl) )
score(q, d)  = sum over t in q of score(t, d)
```

`BM25Index` 维护四个增量统计量，使 `add()` 和 `score()` 保持 O(tokens) 而非 O(corpus)：

```python
self._doc_terms: dict[str, list[str]]       # tokens per doc (for |d|)
self._doc_freqs: dict[str, Counter[str]]    # tf(t, d) lookup
self._df:        Counter[str]               # df(t) across corpus
self._total_doc_len: int                    # for avgdl
```

### 5.4 `MemoryStream` —— 写入侧

三个保持同步的集合：

```python
self._records: list[MemoryRecord]          # insertion order
self._records_by_id: dict[str, MemoryRecord]
self._bm25: BM25Index
```

#### `add()` —— 完整流程

```python
async def add(self, content, *, kind="observation", importance=5.0, metadata=None):
    embedding = await self.embedding_adapter.embed(content)          # (1)

    if self.dedup_threshold is not None:                             # (2)
        duplicate = self._find_same_kind_duplicate(embedding, kind)
        if duplicate is not None:
            return duplicate                                         #     ← existing record

    record = MemoryRecord(                                           # (3)
        id=uuid.uuid4().hex,
        content=content,
        timestamp=self._clock(),
        kind=kind,
        importance=importance,
        embedding=embedding,
        metadata=dict(metadata) if metadata else {},
    )
    self._append_record(record)                                      # (4)
    return record
```

**去重是同 kind 内进行的。** 一条 `observation` 和一条 `fact` 即使嵌入相近也不会被折叠 —— 不同的 kind 携带不同的语义权重。

**`_append_record` 会更新全部三个索引：**

```python
def _append_record(self, record):
    self._records.append(record)
    self._records_by_id[record.id] = record
    self._bm25.add(record.id, record.content)
```

#### `add_record()` —— 逃生门

```python
def add_record(self, record: MemoryRecord) -> None:
    self._append_record(record)
```

绕过嵌入 + 去重。便于测试（注入带伪造时间戳的记录），也为未来从快照恢复的路径做准备。

---

## 6. 文件：`retriever.py` —— 读取侧

### 6.1 `ScoredMemory`

```python
@dataclass
class ScoredMemory:
    record: MemoryRecord
    score: float                # composite weighted score
    recency_score: float        # [0, 1]; 1.0 for facts/preferences
    importance_score: float     # [0.1, 1.0]
    relevance_score: float      # [0, 1], normalized hybrid RRF
    dense_rank: int | None      # 1-indexed rank in dense ranking
    sparse_rank: int | None     # 1-indexed rank in BM25 ranking
```

**每一个评分组件都被显式暴露出来**，包括每个排序器的排名。结果是可检视的：在 `memory_demo.py` 的输出中，你能准确看到为什么 memory N 排在 memory M 的上面。

### 6.2 `MemoryRetriever.retrieve()` —— 七个步骤

```python
async def retrieve(self, query, top_k=None, kinds=None):
    # 1. candidate set — drops done plans, applies `kinds` filter
    candidates = self._select_candidates(kinds)
    if not candidates:
        return []

    k = top_k if top_k is not None else self.memory_config.retrieval_top_k
    if k <= 0:
        return []

    # 2. dense ranking
    query_embedding = await self.embedding_adapter.embed(query)
    dense_scores = {r.id: max(0.0, cosine(query_embedding, r.embedding))
                    for r in candidates}
    dense_rank = ranks_descending(dense_scores)

    # 3. sparse ranking (BM25)
    sparse_raw = self.stream.bm25.score(query)
    sparse_scores = {r.id: sparse_raw.get(r.id, 0.0) for r in candidates}
    sparse_rank = ranks_descending(sparse_scores)

    # 4 + 5. RRF fusion, min-max normalized to [0, 1]
    raw_rrf = {r.id: 1.0 / (RRF_K + dense_rank[r.id])
                   + 1.0 / (RRF_K + sparse_rank[r.id])
               for r in candidates}
    rrf_max = max(raw_rrf.values()) or 1.0
    hybrid_relevance = {rid: raw / rrf_max for rid, raw in raw_rrf.items()}

    # 6. per-kind recency + importance + weighted composite
    now = self._clock()
    scored = []
    for r in candidates:
        recency_score    = self._compute_recency_score(r, now)
        importance_score = r.importance / 10.0
        relevance_score  = hybrid_relevance[r.id]
        composite = (
            self.memory_config.recency_weight    * recency_score
            + self.memory_config.importance_weight * importance_score
            + self.memory_config.relevance_weight  * relevance_score
        )
        scored.append(ScoredMemory(
            record=r, score=composite,
            recency_score=recency_score,
            importance_score=importance_score,
            relevance_score=relevance_score,
            dense_rank=dense_rank[r.id],
            sparse_rank=sparse_rank[r.id],
        ))

    # 7. sort + walk under TWO binding caps: top_k AND token budget
    scored.sort(key=lambda s: s.score, reverse=True)
    return self._apply_token_budget(scored, k)
```

**三个权重来自 `profile.memory`：** `recency_weight`、`importance_weight`、`relevance_weight`。用户通过编辑 YAML 来调整检索行为。

**两个约束上限**（谁先生效谁赢）：

- **`retrieval_top_k`** —— 来自 `profile.memory` 的硬性数量上限。
- **`max_working_memory_tokens`** —— 来自 `profile.memory` 的 token 预算上限（默认 4000）。检索器使用 `estimate_tokens()`（一个便宜的 `len(content) // 4` 启发式）并在下一条记录会溢出之前停止。

**例外：** 即使 top-1 单条就超过预算，也始终会被包含。紧张的预算绝不应在非空候选集上产出空结果。

### 6.3 按 kind 区分的时效衰减

```python
_NO_DECAY_KINDS = frozenset({"fact", "preference"})

def _compute_recency_score(self, record, now):
    if record.kind in _NO_DECAY_KINDS:
        return 1.0
    age_hours = max(0.0, (now - record.timestamp).total_seconds() / 3600)
    return 2.0 ** (-age_hours / self.recency_half_life_hours)
```

默认 24 小时半衰期的指数衰减：t=0 时为 1.0，24 小时为 0.5，48 小时为 0.25，一周之后实际上为 0。

### 6.4 `_select_candidates` —— 计划状态过滤器

```python
def _select_candidates(self, kinds):
    candidates = []
    for r in self.stream.get_all():
        if kinds is not None and r.kind not in kinds:
            continue
        if r.kind == "plan" and r.metadata.get("status") == "done":
            continue
        candidates.append(r)
    return candidates
```

---

## 7. 文件：`memory.py` —— 门面类

### 7.1 构造

```python
class Memory:
    def __init__(self, profile, embedding_adapter, *,
                 stream=None, retriever=None,
                 clock=None, dedup_threshold=0.95,
                 recency_half_life_hours=24.0):
        self.profile = profile
        self.embedding_adapter = embedding_adapter
        self.stream = stream or MemoryStream(embedding_adapter,
                                             clock=clock,
                                             dedup_threshold=dedup_threshold)
        self.retriever = retriever or MemoryRetriever(self.stream,
                                                      embedding_adapter,
                                                      profile.memory,
                                                      clock=clock,
                                                      recency_half_life_hours=recency_half_life_hours)

    async def remember(self, content, *, kind="observation",
                       importance=5.0, metadata=None):
        return await self.stream.add(content, kind=kind,
                                     importance=importance, metadata=metadata)

    async def recall(self, query, *, top_k=None, kinds=None):
        return await self.retriever.retrieve(query, top_k=top_k, kinds=kinds)
```

stream 和 retriever 可以被注入（用于测试、自定义配置）。默认情况下门面类自己构建它们。

### 7.2 `Memory.from_env()` —— 由环境变量驱动的构造

读取 `.env` 中的 `EMBEDDING_*` 块：

```
EMBEDDING_PROVIDER=qwen              # openai | qwen | vllm
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v3
```

校验规则：

- `EMBEDDING_PROVIDER` 缺失或不受支持 → `EmbeddingConfigError`。
- `EMBEDDING_MODEL` 缺失 → `EmbeddingConfigError`。
- `qwen` 或 `vllm` 的 `EMBEDDING_BASE_URL` 缺失 → `EmbeddingConfigError`（OpenAI 可以使用默认值）。
- `openai` 或 `qwen` 的 `EMBEDDING_API_KEY` 缺失 → `EmbeddingConfigError`。对 `vllm` 则默认为 `"token-not-needed"`。

---

## 8. 执行流程：`scripts/memory_demo.py`

```
$ python scripts/memory_demo.py
│
├─ AgentProfile.from_yaml("agents/maya_rodriguez/profile.yaml")
├─ AgentLogger.from_profile(profile, log_file="logs/...memory.log")
├─ Memory.from_env(profile)
│        • reads EMBEDDING_* block
│        • constructs OpenAICompatibleEmbeddingAdapter
│        • constructs MemoryStream + MemoryRetriever
│
├─ seed 14 memories spanning all 5 kinds (clock back-dated per entry)
│        for each:  stream.add(content, kind, importance, metadata)
│                     └─ embed() + dedup + _append_record + bm25.add
│                   logger.info("memory.added", ...)
│
└─ 3 queries
         memory.recall(query, top_k=5)
           └─ retriever.retrieve(...)
                ├─ _select_candidates()        # drops done plans
                ├─ embed(query)                # 1 API call
                ├─ dense_scores  via cosine()
                ├─ sparse_scores via stream.bm25.score()
                ├─ RRF fusion + min-max normalize
                ├─ per-kind recency + importance + weighted sum
                └─ sorted, top-k under token budget
```

### 输出样例

```
QUERY: How am I doing on the data structures homework?
========================================================================
  [observation   imp= 8.0] score=2.478  rec=0.94 imp_n=0.80 rel=1.00  (dense#1 bm25#1)
    Got stuck on problem 3 for an hour, finally worked it out with the TA.
  [observation   imp= 7.0] score=2.334  rec=0.89 imp_n=0.70 rel=0.96  (dense#2 bm25#2)
    Spent two hours in the library working on the BST homework set.
  [plan          imp= 8.0] score=2.172  rec=0.71 imp_n=0.80 rel=0.87  (dense#4 bm25#3)
    Finish the BST homework set by Friday.
  ...
```

排在首位的记忆在稠密和稀疏排名中都是 #1 —— 既语义对齐，又关键词匹配。那条 plan 时效较低（12 小时前），但由于 `importance=8.0` 且关键词匹配，依然排得不错。

### 事情可能出错的地方

| 情形 | 表现形式 |
|---|---|
| `EMBEDDING_PROVIDER` 未设置或未知 | 在 `Memory.from_env` 处抛出 `EmbeddingConfigError` |
| 非 vLLM 场景下缺少 `EMBEDDING_API_KEY` / `_MODEL` | 在 `Memory.from_env` 处抛出 `EmbeddingConfigError` |
| `embed()` 期间网络 / 429 / 鉴权失败 | `EmbeddingProviderError`，原始异常以 `__cause__` 保留 |
| `stream.get_by_id("bogus")` | `MemoryNotFoundError` |
| 在一个流中混用了不同的嵌入模型 | 维度不匹配时 `cosine()` 抛出 `MemoryError` |

---

## 9. 测试覆盖图

| 文件 | 测试数 | 亮点 |
|---|---|---|
| `test_memory.py` | 67 | 门面类 + stream + retriever + cosine；按 kind 区分的规则、混合打破平局、token 预算上限、`from_env` 每个分支 |
| `test_bm25.py` | 18 | 分词、增量语料统计、IDF、已知值的 BM25 分数、边界情况 |
| `test_embedding.py` | 4 | `embed()`/`embed_batch()` 的正常路径、批次重排正确性、provider 异常包装 |

所有测试都完全离线。真实的嵌入仅发生在 `memory_demo.py` 中。

---

## 10. 值得注意的事项

- **每个评分组件都暴露在 `ScoredMemory` 上**，包括每个排序器的排名。按设计即可观测：demo 的读者能看到某条记忆*为什么*排在它所处的位置。

- **BM25 和 cosine 都是纯 Python** —— 不用 `numpy`，不用 `rank-bm25`。在 harness 的规模上（数千条记忆、嵌入维度约 1000），两者每查询都远低于 50 ms。

- **去重是同 kind 内、在写入路径上进行的。** 它在任何排名之前运行，因此重复的 observation 不会通过向池中添加近乎相同的向量来污染未来的检索。

- **持久化是按智能体并按需启用的。** `Memory.from_profile(profile)` 默认到 `<profile.source_dir>/memory/stream.db` —— 每个智能体包拥有自己的 SQLite 文件。当调用方使用 `from_env()` 或在不传 `db_path` 的情况下构造 `Memory` / `MemoryStream` 时，仍然默认为纯内存。关于线上格式和 BM25 重建细节，见下文 §11。

- **模块 1（LLM）没有被导入。** 记忆模块与 `DefenseAgent.llm` 零耦合。这正是让模块 5（反思）成为有意义补充的原因：它正是那个在 LLM 与 Memory 之间架桥的模块。

- **检索上的两个约束上限。** 有一段时间 `max_working_memory_tokens` 是无效配置 —— 只有 `retrieval_top_k` 在限制结果。现在两者都生效，并且有 top-1 始终包含的规则，以保证紧张的预算不会让结果变成空。

---

## 11. 持久化：`sqlite_store.py` + `MemoryStream(db_path=...)` *（新增于 2026-04-24）*

### 11.1 持久化何时生效

三个入口点，接线程度由浅到深：

| 入口点 | 默认行为 | 何时使用 |
|---|---|---|
| `MemoryStream(adapter, db_path=None)` | 仅内存（原始行为） | 测试、短生命周期的 demo |
| `Memory.from_env(profile, db_path=<path>)` | 可选地在你传入的路径上启用 SQLite | 混合配置，你想控制文件位置 |
| `Memory.from_profile(profile)` | **SQLite 位于 `<profile.source_dir>/memory/stream.db`** | 典型的生产路径 —— 每个智能体在其档案旁边拥有自己的 DB |
| `Memory.from_profile(profile, persist=False)` | 仅内存，不在磁盘上留文件 | 针对已存在智能体包的一次性实验 —— 智能体的持久化 stream.db 保持原样 |

使用默认 `persist=True` 的 `from_profile` 要求档案是通过 `AgentProfile.from_yaml(...)` 加载的（这样 `profile.source_dir` 才会被填充）。对于内存中的档案，要么显式传 `memory_dir=...`，要么传 `persist=False`。同时传递 `persist=False` 与 `memory_dir=` 是矛盾的，会抛出 `ValueError`。

### 11.2 Schema

```sql
CREATE TABLE memory_records (
    id            TEXT PRIMARY KEY,
    content       TEXT NOT NULL,
    kind          TEXT NOT NULL,
    importance    REAL NOT NULL,
    timestamp     TEXT NOT NULL,     -- ISO-8601 UTC, lex-sortable
    embedding     BLOB NOT NULL,     -- numpy float32 bytes
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX idx_memory_kind      ON memory_records(kind);
CREATE INDEX idx_memory_timestamp ON memory_records(timestamp);

PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;
```

`metadata_json` 是必需的，尽管最初的 schema 草图里没有它 —— plan 使用 `metadata["status"] = "done"` 来进行检索过滤，丢掉它会在重新加载之后静默地打破该规则。

### 11.3 嵌入序列化

```python
def _embedding_to_blob(vec: list[float]) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()

def _embedding_from_blob(blob: bytes) -> list[float]:
    return np.frombuffer(blob, dtype=np.float32).tolist()
```

使用 Float32（不是 float64）。Qwen/OpenAI 的嵌入模型输出的是 float32 精度；把存储减半是免费的。一条 4096 维的 Qwen 嵌入约占 16KB。往返转换在 `1e-6` 以内完全精确。

### 11.4 打开时的重建

```python
def _open_db(self, db_path):
    self._db = sqlite_store.open_db(db_path)
    for record in sqlite_store.load_records(self._db):
        self._index_record(record)         # in-memory indexing only, no re-write
```

记录按 `(timestamp, id)` 顺序取出，以便 `get_all()` 仍然返回按时间顺序的插入顺序。`_index_record`（相对于 `_append_record`）只更新内存中的结构，而**不**回写到 DB —— 这一点很关键，因为我们正在读取的是我们也会写入的那个文件。

BM25 索引会从扫描中重建：每次调用 `_index_record` 都会包含 `self._bm25.add(record.id, record.content)`。在 harness 的规模上（每个智能体数千条记录），启动时这仅需毫秒级时间。如果哪天它变慢了，规范中已记录 FTS5 作为迁移目标 —— 不过目前还不到那一步。

### 11.5 跨会话去重

`_records_by_kind` 索引会在重建时被填充，因此 `_find_same_kind_duplicate` 能正确地捕获来自上次运行记录的近重复嵌入。去重阈值（默认 0.95）既适用于全新的 add，也适用于匹配到已重建嵌入的 add —— 完全相同的代码路径。

### 11.6 检视工具：`scripts/dump_memory.py`

DB 是可检视的，但嵌入 BLOB 让原始的 `sqlite3` dump 难以阅读。`dump_memory.py` 发起一个只读连接，过滤掉嵌入列，并把每条记录渲染成：

```
[observation  imp= 8.0]  2026-04-24T18:45:06+00:00  <uuid>
    Maya finished the BST homework problem 3 with the TA.
```

支持 `--kind <kind>` 和 `--limit N` 过滤。除标准库外零依赖 —— 可以在智能体仍然在写入的活动 DB 上运行（WAL 使其安全）。

### 11.7 这一切带来了什么（以及没带来什么）

- ✅ 按智能体在磁盘上隔离：`agents/maya/memory/` 和 `agents/alice/memory/` 永不混合。
- ✅ 崩溃安全：每次 `remember()` 都会提交；最坏情况也不过丢失最后一次在途调用。
- ✅ Schema 演进：增加字段时 `ALTER TABLE` 胜过重写 JSONL。
- ❌ **没有驱逐 / 保留策略。** 运行数月的智能体会无界增长。到那时，自然的调节旋钮是截止年龄 + 归档表；现在还不需要。
- ❌ **跨进程没有并发写入者。** SQLite WAL 允许并发*读取*，但两个活着的 `Memory` 实例写同一个 `stream.db` 会在去重上竞态（两者都对 "x" 做嵌入、两者都看不到重复、两者都插入）。如果我们哪天真要对同一个智能体共同运行两个进程，就会加写锁或迁到 FTS5 + 单写入者。
