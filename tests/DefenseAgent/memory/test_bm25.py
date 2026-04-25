"""Tests for DefenseAgent.memory.bm25.

BM25 is a classical ranking function. Tests check:
  - tokenization rules (lowercase, split on non-alnum, drop 1-char tokens)
  - scoring monotonicity (more term freq → higher score, for a given corpus)
  - IDF behavior (rare terms score higher than common terms)
  - edge cases (empty index, query with no matches, tie cases)
"""
from DefenseAgent.memory.stream import BM25Index, tokenize


# ---------- tokenize ----------


def test_tokenize_lowercases():
    assert tokenize("Hello WORLD") == ["hello", "world"]


def test_tokenize_splits_on_non_alnum():
    # punctuation, whitespace, mixed — all are separators
    assert tokenize("Hello, world!  Data-structures?") == [
        "hello", "world", "data", "structures",
    ]


def test_tokenize_drops_single_char_tokens():
    # 1-char tokens are noise (a, I, etc.); 2+ kept
    assert tokenize("I am a student") == ["am", "student"]


def test_tokenize_keeps_numbers():
    assert tokenize("I have 3 classes on 2025-04-22") == [
        "have", "classes", "on", "2025", "04", "22",
    ]


def test_tokenize_empty_string_returns_empty_list():
    assert tokenize("") == []


def test_tokenize_idempotent_on_already_clean_text():
    assert tokenize("hello world") == ["hello", "world"]


# ---------- BM25Index basic structure ----------


def test_empty_index_has_zero_length():
    idx = BM25Index()
    assert len(idx) == 0


def test_add_increments_length():
    idx = BM25Index()
    idx.add("d1", "hello world")
    idx.add("d2", "foo bar")
    assert len(idx) == 2


# ---------- scoring: single-doc corpus ----------


def test_single_doc_query_with_match_returns_positive_score():
    idx = BM25Index()
    idx.add("d1", "hello world")
    scores = idx.score("hello")
    assert scores["d1"] > 0


def test_single_doc_query_with_no_match_returns_zero():
    idx = BM25Index()
    idx.add("d1", "hello world")
    scores = idx.score("goodbye")
    assert scores["d1"] == 0


def test_query_with_no_tokens_returns_all_zeros():
    idx = BM25Index()
    idx.add("d1", "hello world")
    scores = idx.score("!!!")  # tokenizes to []
    assert scores["d1"] == 0


# ---------- scoring: multi-doc IDF behavior ----------


def test_rarer_term_scores_higher_than_common_term():
    """IDF: a term appearing in 1 doc should score higher than one in 9 docs."""
    idx = BM25Index()
    # "common" appears in every doc, "rare" appears only in d1
    for i in range(9):
        idx.add(f"common_{i}", "common word here")
    idx.add("d1", "common rare word here")

    common_score = idx.score("common")["d1"]
    rare_score = idx.score("rare")["d1"]
    assert rare_score > common_score


def test_term_frequency_effect_is_monotonic():
    """More occurrences of a query term in a doc should score higher — for two docs of equal length."""
    idx = BM25Index()
    # Pad both docs to equal length (7 tokens) so length normalization doesn't bias the test.
    # d1 has 'python' once; d2 has 'python' twice.
    idx.add("d1", "python java cpp rust go js css")
    idx.add("d2", "python python java cpp rust go js")
    scores = idx.score("python")
    assert scores["d2"] > scores["d1"]


def test_doc_length_penalty():
    """A shorter doc with the same term should score higher than a longer doc with the same term."""
    idx = BM25Index()
    idx.add("short", "python")
    idx.add("long", "python " + " ".join(["filler"] * 50))
    scores = idx.score("python")
    assert scores["short"] > scores["long"]


# ---------- multi-term queries ----------


def test_multi_term_query_sums_per_term_scores():
    idx = BM25Index()
    idx.add("d1", "python data structures course")
    idx.add("d2", "python programming")
    # "python" matches both; "data" matches only d1 → d1 should beat d2.
    scores = idx.score("python data")
    assert scores["d1"] > scores["d2"]


def test_case_insensitive():
    idx = BM25Index()
    idx.add("d1", "Python is great")
    assert idx.score("PYTHON")["d1"] > 0


# ---------- empty corpus / missing docs ----------


def test_score_on_empty_index_returns_empty_dict():
    idx = BM25Index()
    assert idx.score("anything") == {}


def test_score_includes_all_docs_with_zero_for_non_matchers():
    idx = BM25Index()
    idx.add("d1", "hello world")
    idx.add("d2", "goodbye cruel world")
    scores = idx.score("hello")
    assert set(scores.keys()) == {"d1", "d2"}
    assert scores["d1"] > 0
    assert scores["d2"] == 0
