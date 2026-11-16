import pytest

from plumbline.evals.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from plumbline.retrieve.bm25 import BM25Index
from plumbline.retrieve.fusion import reciprocal_rank_fusion
from plumbline.retrieve.rerank import _min_window


def test_bm25_ranks_the_matching_document_first():
    index = BM25Index(["cats chase mice", "dogs chase cats and cats", "rockets reach orbit"])
    hits = index.search("rocket orbit", 3)
    assert hits[0][0] == 2 and len(hits) == 1  # zero-score documents are not returned
    assert index.search("unicorn", 3) == []


def test_bm25_prefers_rare_terms():
    index = BM25Index(["common common rare", "common common common", "common other words"])
    assert index.idf["rare"] > index.idf["common"]


def test_rrf_rewards_agreement_between_rankers():
    fused = reciprocal_rank_fusion({"a": [1, 2, 3], "b": [3, 1, 4]}, k=60)
    assert [item for item, _, _ in fused][:2] == [1, 3]
    assert fused[0][2] == {"a": 1, "b": 2}


def test_min_window():
    assert _min_window([[0, 10], [11], [12, 40]]) == 3
    assert _min_window([[5]]) == 1


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "hybrid+rerank"])
def test_every_mode_finds_the_right_paper(retriever, mode):
    result = retriever.search("How many trainable parameters does low-rank tuning save?", mode=mode)
    assert result.hits[0].chunk.short == "beta"
    assert result.mode == mode and result.timings_ms


def test_hybrid_records_per_retriever_ranks(retriever):
    hit = retriever.search("in-batch negatives during training", mode="hybrid").hits[0]
    assert hit.chunk.chunk_id == "alpha:0001" and set(hit.ranks) == {"bm25", "dense"}


def test_confidence_is_low_for_off_topic_queries(retriever):
    on = retriever.search("Beta low-rank update matrices trainable parameters").confidence
    off = retriever.search("medieval castle architecture in Scotland").confidence
    assert on > 0.8 and off < 0.3


def test_unknown_mode_is_rejected(retriever):
    with pytest.raises(ValueError):
        retriever.search("x", mode="psychic")


def test_ranking_metrics():
    ranked, relevant = ["a", "b", "c", "d"], {"c", "z"}
    assert recall_at_k(ranked, relevant, 2) == 0.0 and recall_at_k(ranked, relevant, 3) == 1.0
    assert reciprocal_rank(ranked, relevant) == pytest.approx(1 / 3)
    assert 0 < ndcg_at_k(ranked, relevant, 10) < 1
    assert ndcg_at_k(["c", "z"], relevant, 10) == pytest.approx(1.0)
