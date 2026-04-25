"""Tests for OpenAICompatibleEmbeddingAdapter.

Uses the client-injection seam: tests pass a MagicMock where production
passes None and gets a real AsyncOpenAI. No network.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from DefenseAgent.memory import EmbeddingProviderError
from DefenseAgent.memory.embedding import OpenAICompatibleEmbeddingAdapter


@pytest.fixture
def fake_openai_client():
    client = MagicMock()
    client.embeddings = MagicMock()
    client.embeddings.create = AsyncMock()
    return client


def make_fake_embedding_response(vectors: list[list[float]], model: str = "test-model"):
    return SimpleNamespace(
        data=[SimpleNamespace(embedding=v, index=i) for i, v in enumerate(vectors)],
        model=model,
        usage=SimpleNamespace(prompt_tokens=5, total_tokens=5),
    )


# ---------- embed() ----------


async def test_embed_single_text_returns_vector(fake_openai_client):
    fake_openai_client.embeddings.create.return_value = make_fake_embedding_response(
        [[0.1, 0.2, 0.3]]
    )
    adapter = OpenAICompatibleEmbeddingAdapter(
        api_key="k", base_url="u", model="m", client=fake_openai_client,
    )

    vec = await adapter.embed("hello world")

    assert vec == [0.1, 0.2, 0.3]
    kwargs = fake_openai_client.embeddings.create.call_args.kwargs
    assert kwargs["input"] == "hello world"
    assert kwargs["model"] == "m"


# ---------- embed_batch() ----------


async def test_embed_batch_returns_list_of_vectors(fake_openai_client):
    fake_openai_client.embeddings.create.return_value = make_fake_embedding_response(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    )
    adapter = OpenAICompatibleEmbeddingAdapter(
        api_key="k", base_url="u", model="m", client=fake_openai_client,
    )

    vecs = await adapter.embed_batch(["a", "b", "c"])

    assert vecs == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    kwargs = fake_openai_client.embeddings.create.call_args.kwargs
    assert kwargs["input"] == ["a", "b", "c"]


async def test_embed_batch_preserves_order_of_input(fake_openai_client):
    """If the API returns items out of order, the adapter must re-sort by `index`."""
    out_of_order = SimpleNamespace(
        data=[
            SimpleNamespace(embedding=[0.9], index=2),
            SimpleNamespace(embedding=[0.1], index=0),
            SimpleNamespace(embedding=[0.5], index=1),
        ],
        model="m",
        usage=SimpleNamespace(prompt_tokens=5, total_tokens=5),
    )
    fake_openai_client.embeddings.create.return_value = out_of_order
    adapter = OpenAICompatibleEmbeddingAdapter(
        api_key="k", base_url="u", model="m", client=fake_openai_client,
    )
    vecs = await adapter.embed_batch(["a", "b", "c"])
    assert vecs == [[0.1], [0.5], [0.9]]


# ---------- error wrapping ----------


async def test_provider_exceptions_are_wrapped(fake_openai_client):
    fake_openai_client.embeddings.create.side_effect = RuntimeError("429")
    adapter = OpenAICompatibleEmbeddingAdapter(
        api_key="k", base_url="u", model="m", client=fake_openai_client,
    )

    with pytest.raises(EmbeddingProviderError) as excinfo:
        await adapter.embed("x")

    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "429" in str(excinfo.value)
