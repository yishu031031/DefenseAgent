# Providers & Multimodal

Which LLM and embedding providers DefenseAgent speaks, the per-provider quirks, and how to use vision-capable models.

## Provider table

`AGENT_LAB_LLM_PROVIDER` selects the adapter. Each provider has its own `<PROVIDER>_*` env block. The cross-provider `LLM_*` tier (`LLM_API_KEY` / `LLM_MODEL_ID` / `LLM_BASE_URL`) overrides per-provider when set.

| Provider | Adapter | Typical key format | Default base URL | Example chat models |
|---|---|---|---|---|
| `openai` | `OpenAICompatibleAdapter` | `sk-…` or `sk-proj-…` | `https://api.openai.com/v1` | `gpt-4o-mini`, `gpt-4o`, `o3-mini` |
| `anthropic` | `AnthropicAdapter` | `sk-ant-…` | `https://api.anthropic.com` | `claude-sonnet-4-6`, `claude-opus-4-7` |
| `deepseek` | `OpenAICompatibleAdapter` | `sk-…` | `https://api.deepseek.com/v1` | `deepseek-chat`, `deepseek-reasoner` |
| `qwen` (DashScope, OpenAI-compat) | `OpenAICompatibleAdapter` | `sk-…` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus`, `qwen-max`, `qwen-turbo` |
| `google` (OpenAI-compat endpoint) | `OpenAICompatibleAdapter` | `sk-…` | `https://generativelanguage.googleapis.com/v1beta/openai` | `gemini-2.0-flash` |
| `vllm` (self-hosted) | `OpenAICompatibleAdapter` | any string (`EMPTY` / `token-not-needed`) | depends on deployment, e.g. `http://localhost:8000/v1` | whatever the vLLM server is serving |

## Embedding pairings

Embedding lives in a separate `EMBEDDING_*` block:

| Embedder | `EMBEDDING_BASE_URL` | `EMBEDDING_MODEL` | `EMBEDDING_DIMS` |
|---|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `text-embedding-3-small` | 1536 |
| OpenAI | `https://api.openai.com/v1` | `text-embedding-3-large` | 3072 |
| DashScope | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `text-embedding-v3` | 1024 |
| ModelScope | `https://api-inference.modelscope.cn/v1` | `Qwen/Qwen3-Embedding-0.6B` | 1024 |
| ModelScope | `https://api-inference.modelscope.cn/v1` | `Qwen/Qwen3-Embedding-8B` | 4096 |
| Ollama (local) | `http://localhost:11434/v1` | `bge-m3` | 1024 |
| vLLM (local) | `http://localhost:8000/v1` | `BAAI/bge-large-zh-v1.5` | 1024 |

`EMBEDDING_DIMS` **must match** what the model emits or the Qdrant collection rejects writes — set it from the model's documented vector size.

For local embedding setup (Ollama / vLLM): they speak OpenAI-compatible `/v1/embeddings`, so set `EMBEDDING_PROVIDER=openai` and point `EMBEDDING_BASE_URL` at the local server.

## Per-field fallback in practice

For each LLM field, top to bottom (first non-empty wins):

1. `llm.<field>:` in profile YAML
2. Cross-provider env tier — `LLM_API_KEY` / `LLM_MODEL_ID` / `LLM_BASE_URL`
3. Per-provider env tier — `<PROVIDER>_API_KEY` / `<PROVIDER>_MODEL` / `<PROVIDER>_BASE_URL`
4. Schema default

Concrete example. Given:

```yaml
# profile.yaml
llm:
  provider: deepseek
  model: deepseek-reasoner             # profile sets this explicitly
```

```bash
# .env
LLM_API_KEY=sk-shared                  # cross-provider override, wins over per-provider
DEEPSEEK_API_KEY=sk-deepseek           # per-provider, used if LLM_API_KEY absent
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat           # ignored — profile's model wins
```

Final resolution:
- `provider` → `deepseek` (profile)
- `model` → `deepseek-reasoner` (profile beats `DEEPSEEK_MODEL`)
- `base_url` → `https://api.deepseek.com/v1` (profile empty → falls to `DEEPSEEK_BASE_URL`)
- `api_key` → `sk-shared` (cross-provider `LLM_API_KEY` beats `DEEPSEEK_API_KEY`)

## Per-provider notes

| Provider | Things to know |
|---|---|
| `openai` | Both `sk-…` and `sk-proj-…` keys work. Reasoning models (`o3-mini`, `o1`) cost more and require a slightly different request shape — adapter handles it transparently. |
| `anthropic` | Tool calls supported. The Anthropic wire format for non-text content differs from OpenAI's, so list-shape `content` reaches the adapter as `LLMAdapterError`. See [Multimodal](#multimodal-input) for vision provider choices. |
| `deepseek` | `deepseek-reasoner` returns thinking tokens in `reasoning_content` — the adapter strips them from `Message.content` so downstream code doesn't see the chain-of-thought. To inspect them, look at the raw response. |
| `google` | Uses Google's OpenAI-compatible endpoint at `generativelanguage.googleapis.com/v1beta/openai`. Native Gemini SDK is not used. |
| `vllm` | `VLLM_API_KEY=EMPTY` (literal string) is the convention. `VLLM_MODEL` must match what's loaded on the server (see vLLM's `--served-model-name`). |

## Programmatic LLM injection

`AgentConfig` accepts a pre-built `LLM` instance — when given, **the env-driven construction path is skipped entirely** for the LLM:

```python
from DefenseAgent.llm import LLM
from DefenseAgent.llm.openai_compat import OpenAICompatibleAdapter
from DefenseAgent.llm.anthropic import AnthropicAdapter

# 1. Test with a scripted/mocked LLM
config = AgentConfig(profile="…", llm=ScriptedLLM(responses=[...]))

# 2. Multiple agents with different providers in the same process
config_a = AgentConfig(profile=p, llm=LLM(adapter=OpenAICompatibleAdapter(
    api_key="...", base_url="https://api.openai.com/v1", model="gpt-4o",
)))
config_b = AgentConfig(profile=p, llm=LLM(adapter=AnthropicAdapter(
    api_key="...", model="claude-sonnet-4-6",
)))

# 3. Custom adapter (subclass LLMAdapter)
config = AgentConfig(profile="…", llm=LLM(adapter=MyCustomAdapter()))
```

Same pattern applies to every component — see [`architecture.md`](architecture.md#customization--dependency-injection).

---

## Multimodal input

DefenseAgent can attach images to the user turn so the LLM reasons about visual content alongside text. **Opt-in** — you only pay the multimodal cost when you pass `images=`.

### What "multimodal" means here

The OpenAI chat-completions API allows the `content` field to be a **list of content blocks** instead of a plain string — each block is either text or an `image_url`. DefenseAgent's `Message` type already supports this shape; `agent.run(task, images=[...])` is the ergonomic helper that builds the list.

Useful for:
- Visual Q&A — "what's in this screenshot?", "is the chart showing growth?"
- OCR — receipts, scanned PDFs, screenshots of code
- Visual debugging — UI screenshots to suggest CSS fixes
- Image-grounded reasoning — comparing two product photos, layout review

Not for: image generation, video, audio. Just static images going *into* the model.

### Pick a vision-capable model

| Provider | Vision-capable models | Notes |
|---|---|---|
| OpenAI | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo` (vision endpoint) | `gpt-4o-mini` is the cheap default for OCR-style tasks |
| Qwen (DashScope) | `qwen-vl-max`, `qwen-vl-plus`, `qwen-vl-max-latest` | The `-vl-` prefix signals visual; non-VL Qwen models won't accept images |
| GLM (智谱, OpenAI-compat) | `glm-4v`, `glm-4v-flash` | Hit GLM's OpenAI-compatible endpoint via `provider: openai` + `OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4` |
| Kimi (Moonshot, OpenAI-compat) | `moonshot-v1-32k-vision-preview` | Same pattern — point `OPENAI_BASE_URL` at Moonshot |
| vLLM (self-hosted) | `Qwen/Qwen2-VL-7B-Instruct`, `llava-hf/llava-1.5-13b-hf` | Server must be launched with `--limit-mm-per-prompt image=N` |
| **Anthropic** | **Not yet supported** in this version — see limitation below |

Setup is the same as any other model — just point `<PROVIDER>_MODEL` at a vision-capable id:

```bash
# .env — Qwen-VL via DashScope
AGENT_LAB_LLM_PROVIDER=qwen
QWEN_API_KEY=sk-…
QWEN_MODEL=qwen-vl-max
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### End-to-end example

```python
import asyncio
from pathlib import Path
from DefenseAgent.agent import AgentConfig, ReActAgent
from DefenseAgent.examples import EXAMPLE_PROFILE_PATH

async def main():
    agent = ReActAgent(AgentConfig(profile=EXAMPLE_PROFILE_PATH))
    result = await agent.run(
        "Describe what's in this image, including any text you can read.",
        images=[Path("./screenshot.png")],
    )
    print(result.final_answer)

asyncio.run(main())
```

The agent treats the image as part of the user turn — the LLM sees it natively, no separate OCR pass.

### How images flow through the system

`agent.run(task, images=[...])` walks each entry, normalises it into a single URL string, and builds the OpenAI content-block message. Three input types are accepted:

| Input | What happens to it |
|---|---|
| `Path` / local file path string | Read, base64-encoded, turned into `data:<mime>;base64,…`. MIME inferred from extension; unknown defaults to `image/png`. |
| `http://` or `https://` URL string | Passed through unchanged. Provider fetches it. |
| `data:` URL string | Passed through unchanged — useful when you already have an in-memory encoding. |

The agent does **no preprocessing** — no resizing, no compression. Implications:

1. **Base64 inflates by ~33%.** A 5 MB PNG becomes ~6.7 MB. Resize before passing if the model can work with smaller dimensions.
2. **Provider size limits apply.** OpenAI rejects requests above ~20 MB; DashScope varies by model. Hit it → 4xx from the provider, not a friendly DefenseAgent error.

### Constraints & good practice

- **Multiple images per turn**: unbounded on DefenseAgent's side, but most providers cap (OpenAI: up to ~10; Qwen-VL: similar). Hit cap → request fails.
- **Supported formats**: PNG and JPEG universal; WebP, GIF (first frame), BMP work on most providers; HEIC and AVIF spotty.
- **Transparency**: PNG alpha passed through verbatim, ignored by vision models.
- **OCR-heavy**: high resolution, OCR-marketed model (`qwen-vl-max`, `gpt-4o`).
- **Batch**: fire many parallel `agent.run()` calls rather than stuffing one turn — same total token cost, better latency, easier error isolation.

### Where images get carried across multi-step agents

| Agent | Image-carrying behaviour |
|---|---|
| `SimpleAgent` | One turn, one call. Images attached to that single user message. |
| `ReActAgent` | Images attached **only to the initial user turn**. Tool-result messages stay text — the LLM already saw the images. |
| `PlanAndSolveAgent` | Images attached to Phase 1 (plan) AND every Phase 2 (execute-step) message. Phase 3 (synthesis) is text-only. |

So an n-step ReAct over an image makes one image-carrying call and (n-1) text-only follow-ups. Cost ≈ `1 × (text + image) + (n-1) × text`, not `n × image`.

### Anthropic limitation

Claude's wire format for non-text content uses Anthropic's own `{"type": "image", "source": {...}}` block shape, **not** OpenAI's `{"type": "image_url", ...}`. The `AnthropicAdapter` does not currently translate between them — passing list-shape `content` raises:

```python
LLMAdapterError: AnthropicAdapter received list-shape content but does not yet
support multimodal translation. Use an OpenAI-compatible vision provider, or
pass plain text content.
```

The `Message` type itself already accepts list content — the missing piece is a content-block translator inside the Anthropic adapter. PRs welcome — the change is localised to [`DefenseAgent/llm/anthropic.py`](../DefenseAgent/llm/anthropic.py).

For now, if you need vision: pick any OpenAI-compatible provider above.
