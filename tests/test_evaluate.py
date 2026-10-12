import pandas as pd

import evaluate
from evaluate import (
    check_quality_gate,
    evaluate_final_contexts,
    load_testset,
    retrieval_metrics,
    select_sample,
    summarise_ragas,
)

PASSAGE_A = "the drawer of such cheque fails to make the payment of the said amount of money"
PASSAGE_B = "it shall be presumed, unless the contrary is proved, that the holder of a cheque"
NOISE = "The appeal is accordingly dismissed and the parties shall bear their own costs."


def test_retrieval_metrics():
    texts = [NOISE, f"... {PASSAGE_A} ...", NOISE, f"{PASSAGE_B} received the cheque"]
    metrics = retrieval_metrics([PASSAGE_A, PASSAGE_B], texts, ks=(1, 2))
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 0.5
    assert metrics["mrr"] == 0.5
    assert metrics["recall@1"] == 0.0
    assert metrics["recall@2"] == 0.5


def test_retrieval_metrics_nothing_found():
    metrics = retrieval_metrics([PASSAGE_A], [NOISE, NOISE], ks=(5,))
    assert metrics == {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "recall@5": 0.0}
    assert retrieval_metrics([PASSAGE_A], [], ks=())["precision"] == 0.0


def test_one_relevant_chunk_out_of_four_is_not_full_precision():
    """The case RAGAS context precision scores as 1.0."""
    metrics = retrieval_metrics([PASSAGE_A], [PASSAGE_A, NOISE, NOISE, NOISE], ks=())
    assert metrics["precision"] == 0.25
    assert metrics["mrr"] == 1.0


def test_evaluate_final_contexts_skips_unanswerable():
    eval_list = [
        {"user_input": "q1", "reference_contexts": [PASSAGE_A], "retrieved_contexts": [PASSAGE_A, NOISE]},
        {"user_input": "q2", "reference_contexts": [], "retrieved_contexts": [NOISE]},
    ]
    result = evaluate_final_contexts(eval_list)
    assert list(result["question"]) == ["q1"]
    assert result.iloc[0]["precision"] == 0.5


def test_select_sample_is_stable_and_keeps_unanswerable():
    testset = load_testset()
    sample = select_sample(testset, 20)
    assert sample["answerable"].sum() == 20
    assert (~sample["answerable"]).sum() == (~testset["answerable"]).sum()
    assert list(sample["question"]) == list(select_sample(testset, 20)["question"])
    assert len(select_sample(testset, 0)) == len(testset)


def test_summarise_ragas_reports_nan():
    result_df = pd.DataFrame({"faithfulness": [1.0, 0.5, None], "context_recall": [None, None, None]})
    summary = summarise_ragas(result_df)
    assert summary["faithfulness"] == {"mean": 0.75, "scored": 2, "nan": 1}
    assert summary["context_recall"] == {"mean": None, "scored": 0, "nan": 3}


def _summary(recall=0.95, failed=0, faithfulness=0.9, nan=0):
    return {
        "retrieval": {"candidates": {"recall": recall}},
        "generation": {"attempted": 20, "failed": failed},
        "ragas": {"faithfulness": {"mean": faithfulness, "scored": 20 - nan, "nan": nan}},
    }


def test_quality_gate():
    assert check_quality_gate(_summary()) == []
    assert check_quality_gate({"retrieval": {"candidates": {"recall": 0.95}}}) == []
    assert "candidate recall" in check_quality_gate(_summary(recall=0.5))[0]
    assert "failed to generate" in check_quality_gate(_summary(failed=10))[0]
    assert "faithfulness 0.5000" in check_quality_gate(_summary(faithfulness=0.5))[0]
    assert "NaN" in check_quality_gate(_summary(nan=10))[0]
    assert check_quality_gate(_summary(faithfulness=None, nan=20)) == ["faithfulness could not be scored"]


def test_stale_cache_is_ignored(tmp_path, monkeypatch):
    cache_file = tmp_path / "cache.json"
    monkeypatch.setattr(evaluate, "CACHE_FILE", str(cache_file))
    assert evaluate.load_cache("abc") == {}
    cache_file.write_text('{"fingerprint": "abc", "responses": {"q": {"response": "a"}}}')
    assert evaluate.load_cache("abc") == {"q": {"response": "a"}}
    assert evaluate.load_cache("different") == {}
    # A cache written before fingerprints existed
    cache_file.write_text('{"q": {"response": "a"}}')
    assert evaluate.load_cache("abc") == {}


def test_fingerprint_changes_with_prompt_version():
    assert evaluate.pipeline_fingerprint("v1") == evaluate.pipeline_fingerprint()
    assert evaluate.pipeline_fingerprint("v1") != evaluate.pipeline_fingerprint("v2")


def test_samples_are_accepted_by_ragas():
    from ragas import EvaluationDataset

    dataset = EvaluationDataset.from_list([{
        "user_input": "q",
        "response": "a",
        "reference": "ground truth",
        "reference_contexts": [PASSAGE_A],
        "retrieved_contexts": [PASSAGE_A, NOISE],
    }])
    assert len(dataset) == 1


def test_run_evaluation_end_to_end(tmp_path, monkeypatch):
    """The whole run with the retriever, LLM and RAGAS judge faked out."""
    import asyncio
    import json
    import shutil

    import pytest
    from langchain_core.documents import Document
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.retrievers import BaseRetriever

    from main import build_rag_chain

    testset = load_testset()
    testset_path = tmp_path / "testset.csv"
    testset.drop(columns="answerable").to_csv(testset_path, index=False)
    contexts = dict(zip(testset["question"], testset["context"]))

    class OracleRetriever(BaseRetriever):
        """Returns the reference context plus one irrelevant chunk."""
        def _get_relevant_documents(self, query, *, run_manager):
            return [Document(page_content=contexts[query] or NOISE), Document(page_content=NOISE)]

    refusal = "The documents I have access to don't fully address this question."
    chain = build_rag_chain(retriever=OracleRetriever(), llm=FakeListChatModel(responses=[refusal]))
    monkeypatch.setattr(evaluate, "build_rag_chain", lambda prompt_version=None: chain)
    monkeypatch.setattr(evaluate, "get_hybrid_retriever", lambda: OracleRetriever())

    async def no_sleep(seconds):
        return None
    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    judged = []
    def fake_ragas(eval_list):
        judged.extend(eval_list)
        return pd.DataFrame({metric: [0.9] * len(eval_list) for metric in evaluate.RAGAS_METRICS})
    monkeypatch.setattr(evaluate, "run_ragas", fake_ragas)

    # Outputs are written to the working directory, so run in a scratch copy
    shutil.copy("prompts.yaml", tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        evaluate.run_evaluation(max_samples=6, testset_path=str(testset_path))
    assert exit_info.value.code == 0

    unanswerable = int((~testset["answerable"]).sum())
    summary = json.loads((tmp_path / "evaluation_summary.json").read_text())
    assert summary["retrieval"]["candidates"]["questions"] == int(testset["answerable"].sum())
    assert summary["retrieval"]["candidates"]["recall"] == 1.0
    assert summary["retrieval"]["final"] == {"questions": 6, "recall": 1.0, "precision": 0.5, "mrr": 1.0}
    assert summary["generation"] == {"attempted": 6 + unanswerable, "failed": 0}
    assert summary["refusals"] == {"questions": unanswerable, "declined": unanswerable}
    assert summary["ragas"]["faithfulness"] == {"mean": 0.9, "scored": 6, "nan": 0}
    # Only answerable questions are sent to the judge
    assert len(judged) == 6 and all(sample["reference_contexts"] for sample in judged)
    assert (tmp_path / "evaluation_results.csv").exists()

    # A second run answers from the cache and fails the gate when faithfulness drops
    monkeypatch.setattr(evaluate, "build_rag_chain", lambda prompt_version=None: None)
    monkeypatch.setattr(
        evaluate, "run_ragas",
        lambda eval_list: pd.DataFrame({"faithfulness": [0.2] * len(eval_list)}),
    )
    with pytest.raises(SystemExit) as exit_info:
        evaluate.run_evaluation(max_samples=6, testset_path=str(testset_path))
    assert exit_info.value.code == 1


def test_rejected_request_stops_the_run_without_retrying(monkeypatch):
    """A retired model or a bad key fails every question the same way."""
    import asyncio

    import pytest

    class ModelNotFound(Exception):
        status_code = 404

    class RateLimited(Exception):
        status_code = 429

    assert not evaluate.is_retryable(ModelNotFound())
    assert evaluate.is_retryable(RateLimited())
    assert evaluate.is_retryable(TimeoutError())

    calls = []

    class DeadModelChain:
        async def ainvoke(self, inputs):
            calls.append(inputs["input"])
            raise ModelNotFound("The model `old-model` has been decommissioned")

    testset = load_testset().head(4)
    with pytest.raises(SystemExit) as exit_info:
        asyncio.run(evaluate.generate_responses_async(testset, DeadModelChain(), "fingerprint"))
    assert "decommissioned" in str(exit_info.value)
    assert "LLM_MODEL" in str(exit_info.value)
    # Only the first batch of two was attempted, once each
    assert len(calls) == 2
