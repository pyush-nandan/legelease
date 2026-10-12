"""Evaluate LegalEase.

    python evaluate.py                    # retrieval metrics + RAGAS on a 20-question sample
    python evaluate.py --max-samples 0    # same, on every question (needs a paid-tier quota)
    python evaluate.py --retrieval-only   # retrieval metrics only: no API keys, no LLM

Two kinds of numbers come out of this script:

* Retrieval metrics are computed by text-matching retrieved chunks against the
  verbatim passages in the `context` column of testset.csv. They are
  deterministic and cost nothing, so they run on every question.
* RAGAS metrics are scored by an LLM judge, so they are slower, rate-limited and
  only as reliable as the judge model.
"""
import argparse
import asyncio
import hashlib
import json
import os
import sys
import pandas as pd
from tenacity import retry, wait_exponential, stop_after_attempt
from dotenv import load_dotenv
from ingest_data import CHUNK_OVERLAP, CHUNK_SIZE, INDEX_VERSION
from main import LLM_MODEL, build_rag_chain, load_prompt_config
from retriever import BM25_K, RERANK_MODEL, RERANK_TOP_N, VECTOR_K, get_hybrid_retriever
from text_utils import CONTEXT_SEPARATOR, contains_passage, is_no_answer, split_reference_contexts

load_dotenv()

FAITHFULNESS_THRESHOLD = 0.75
# Share of the reference passages that must reach the reranker (measured: 0.95)
CANDIDATE_RECALL_THRESHOLD = 0.85
# A run is rejected when more than this share of questions errored, or of the
# gated metric's scores came back NaN. Averaging only the survivors hides failures.
MAX_FAILED_FRACTION = float(os.getenv("MAX_FAILED_FRACTION", 0.2))

# LLM-judged samples per run. 20 keeps a run inside Groq's free-tier daily quota.
DEFAULT_MAX_SAMPLES = int(os.getenv("EVAL_MAX_SAMPLES", 20))
# The judge defaults to the generator's model so the run fits one free-tier quota.
# A stronger judge is stricter and more reliable if your quota allows it, e.g.
#   JUDGE_MODEL=llama-3.3-70b-versatile JUDGE_MAX_TOKENS=4096
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama-3.1-8b-instant")
JUDGE_MAX_TOKENS = int(os.getenv("JUDGE_MAX_TOKENS", 1024))

CACHE_FILE = "rag_responses_cache.json"
RETRIEVAL_KS = (5, 10)
RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


# ── Test set ────────────────────────────────────────────────────────────

def load_testset(testset_path: str = "testset.csv") -> pd.DataFrame:
    """Load and sanity-check the test set.

    A row with an empty `context` is a question the corpus cannot answer: the
    expected behaviour is a refusal, so it is scored separately.
    """
    testset = pd.read_csv(testset_path).fillna("")
    for column in ("question", "ground_truth", "context"):
        if column not in testset.columns:
            raise ValueError(f"{testset_path} is missing the '{column}' column")

    problems = []
    for i, row in enumerate(testset.itertuples(index=False), start=2):
        if not row.question.strip() or not row.ground_truth.strip():
            problems.append(f"line {i}: empty question or ground_truth")
        elif row.question.strip() == row.ground_truth.strip():
            problems.append(f"line {i}: ground_truth is a copy of the question")
    if testset["question"].duplicated().any():
        problems.append("duplicate questions")
    if problems:
        raise ValueError(f"{testset_path} has invalid rows:\n  " + "\n  ".join(problems))

    testset["answerable"] = testset["context"].str.strip() != ""
    return testset


def select_sample(testset: pd.DataFrame, max_samples: int) -> pd.DataFrame:
    """Pick the questions that go through the LLM pipeline.

    Answerable questions are sampled evenly across the file (so statutes and
    every judgment stay represented) and the choice is the same on every run.
    Unanswerable questions are always kept: they are few and only need one call.
    """
    answerable = testset[testset["answerable"]]
    unanswerable = testset[~testset["answerable"]]
    if max_samples and max_samples < len(answerable):
        step = len(answerable) / max_samples
        answerable = answerable.iloc[[int(i * step) for i in range(max_samples)]]
    return pd.concat([answerable, unanswerable])


# ── Deterministic retrieval metrics ─────────────────────────────────────

def retrieval_metrics(passages: list, texts: list, ks=RETRIEVAL_KS) -> dict:
    """Score one ranked list of retrieved texts against the reference passages.

    recall     share of reference passages found anywhere in the list
    recall@k   share found in the first k results
    precision  share of results that contain a reference passage. This is a
               lower bound: a chunk can be relevant without quoting a passage.
    mrr        1 / rank of the first result containing a reference passage
    """
    first_hit = {}      # passage index -> rank of the first text containing it
    relevant = [False] * len(texts)
    for rank, text in enumerate(texts, start=1):
        for i, passage in enumerate(passages):
            if contains_passage(text, passage):
                relevant[rank - 1] = True
                first_hit.setdefault(i, rank)

    metrics = {
        "recall": len(first_hit) / len(passages),
        "precision": sum(relevant) / len(texts) if texts else 0.0,
        "mrr": 1 / min(first_hit.values()) if first_hit else 0.0,
    }
    for k in ks:
        metrics[f"recall@{k}"] = sum(rank <= k for rank in first_hit.values()) / len(passages)
    return metrics


def evaluate_candidates(testset: pd.DataFrame, retriever=None) -> pd.DataFrame:
    """Retrieval metrics for the hybrid (vector + BM25) candidates, before reranking."""
    retriever = retriever or get_hybrid_retriever()
    rows = []
    for row in testset[testset["answerable"]].itertuples(index=False):
        docs = retriever.invoke(row.question)
        # A statute piece stands for its whole section, which is what the LLM is given
        texts = [doc.metadata.get("parent_text") or doc.page_content for doc in docs]
        rows.append({
            "question": row.question,
            **retrieval_metrics(split_reference_contexts(row.context), texts),
        })
    return pd.DataFrame(rows)


def evaluate_final_contexts(eval_list: list) -> pd.DataFrame:
    """Retrieval metrics for the reranked chunks that were actually sent to the LLM."""
    rows = []
    for sample in eval_list:
        if sample["reference_contexts"]:
            passages = [
                passage
                for context in sample["reference_contexts"]
                for passage in split_reference_contexts(context)
            ]
            rows.append({
                "question": sample["user_input"],
                **retrieval_metrics(passages, sample["retrieved_contexts"], ks=()),
            })
    return pd.DataFrame(rows)


# ── Generation ──────────────────────────────────────────────────────────

def pipeline_fingerprint(prompt_version=None) -> str:
    """Hash of everything that changes the pipeline's output, so cached answers
    from an older prompt, model or retrieval setup are never reused."""
    prompt_config = load_prompt_config()
    version = prompt_version or prompt_config["active_version"]
    config = {
        "prompt": prompt_config["prompts"][version]["system"] + prompt_config["prompts"][version]["human"],
        "llm": LLM_MODEL,
        "reranker": [RERANK_MODEL, RERANK_TOP_N],
        "retrieval": [VECTOR_K, BM25_K],
        "index": [INDEX_VERSION, CHUNK_SIZE, CHUNK_OVERLAP],
    }
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


def load_cache(fingerprint: str) -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
        if cache.get("fingerprint") == fingerprint:
            return cache.get("responses", {})
        print("  Pipeline changed since the cache was written. Ignoring cached responses.")
    return {}


@retry(wait=wait_exponential(multiplier=1, min=4, max=15), stop=stop_after_attempt(5))
async def safe_ainvoke(rag_chain, query):
    return await rag_chain.ainvoke({"input": query})

async def generate_responses_async(testset, rag_chain, fingerprint=""):
    cache = load_cache(fingerprint)

    eval_list = []
    total = len(testset)
    batch_size = 2  # Process 2 at a time

    for i in range(0, total, batch_size):
        batch = testset.iloc[i:i+batch_size]
        tasks = []
        rows_to_process = []

        for row in batch.itertuples(index=False):
            if row.question in cache:
                print(f"  [Cached] {row.question[:80]}...")
                eval_list.append(cache[row.question])
            else:
                print(f"  [API] {row.question[:80]}...")
                tasks.append(safe_ainvoke(rag_chain, row.question))
                rows_to_process.append(row)

        if tasks:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for row, response in zip(rows_to_process, responses):
                if isinstance(response, Exception):
                    print(f"  Error on '{row.question[:40]}': {response}")
                else:
                    result_dict = {
                        "user_input": row.question,
                        "response": response["answer"],
                        "reference": row.ground_truth,
                        "reference_contexts": [c for c in row.context.split(CONTEXT_SEPARATOR) if c.strip()],
                        "retrieved_contexts": [doc.page_content for doc in response["context"]],
                    }
                    cache[row.question] = result_dict
                    eval_list.append(result_dict)

            # Save cache to disk
            with open(CACHE_FILE, "w") as f:
                json.dump({"fingerprint": fingerprint, "responses": cache}, f, indent=2)

            # Throttle to stay within 30 RPM (2 requests / 5 seconds = 24 RPM)
            print("  Sleeping for 5s to avoid rate limits...")
            await asyncio.sleep(5)

    return eval_list


def generate_responses(testset: pd.DataFrame, prompt_version=None) -> list:
    """Run the RAG chain on every row of `testset` (cached per pipeline version)."""
    rag_chain = build_rag_chain(prompt_version=prompt_version)
    eval_list = asyncio.run(
        generate_responses_async(testset, rag_chain, pipeline_fingerprint(prompt_version))
    )
    print(f"\nBuilt {len(eval_list)}/{len(testset)} evaluation samples.")
    return eval_list


def build_eval_dataset(testset_path: str = "testset.csv"):
    """Run the RAG chain on every answerable test question and build an EvaluationDataset."""
    from ragas import EvaluationDataset

    testset = load_testset(testset_path)
    eval_list = generate_responses(testset[testset["answerable"]])
    return EvaluationDataset.from_list(eval_list)


# ── RAGAS (LLM-judged) ──────────────────────────────────────────────────

def run_ragas(eval_list: list) -> pd.DataFrame:
    from openai import AsyncOpenAI
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
    from ragas.run_config import RunConfig

    client = AsyncOpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        max_retries=5,
        timeout=120.0,
    )

    llm = llm_factory(
        model=JUDGE_MODEL,
        provider="openai",
        client=client,
        max_tokens=JUDGE_MAX_TOKENS,
    )

    embedding_model = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

    metrics = [
        Faithfulness(llm = llm),
        AnswerRelevancy(llm = llm, embeddings = embedding_model),
        ContextPrecision(llm = llm),
        ContextRecall(llm = llm)
    ]

    # Rate-limit-friendly config for Groq free tier
    run_config = RunConfig(
        max_workers=2,
        max_retries=10,
        max_wait=90,
        timeout=300,
    )

    result = evaluate(
        dataset=EvaluationDataset.from_list(eval_list),
        metrics=metrics,
        llm=llm,
        embeddings=embedding_model,
        run_config=run_config,
        raise_exceptions=False,
        batch_size=5,
    )

    print("\n=== Evaluation Results ===")
    print(result)
    return result.to_pandas()


def summarise_ragas(result_df: pd.DataFrame) -> dict:
    """Mean of each RAGAS metric, with how many samples it is based on.

    pandas' mean() silently skips NaN, so the count of unscored samples is
    reported next to every average instead of being hidden by it.
    """
    summary = {}
    for metric in RAGAS_METRICS:
        if metric in result_df.columns:
            scores = pd.to_numeric(result_df[metric], errors="coerce")
            summary[metric] = {
                "mean": None if scores.notna().sum() == 0 else round(float(scores.mean()), 4),
                "scored": int(scores.notna().sum()),
                "nan": int(scores.isna().sum()),
            }
    return summary


# ── Quality gate ────────────────────────────────────────────────────────

def check_quality_gate(summary: dict) -> list:
    """Return the reasons this run fails the quality gate (empty list = pass)."""
    failures = []

    candidate_recall = summary["retrieval"]["candidates"]["recall"]
    if candidate_recall < CANDIDATE_RECALL_THRESHOLD:
        failures.append(
            f"candidate recall {candidate_recall:.4f} < {CANDIDATE_RECALL_THRESHOLD}"
        )

    generation = summary.get("generation")
    if generation is None:
        return failures

    if generation["failed"] > MAX_FAILED_FRACTION * generation["attempted"]:
        failures.append(
            f"{generation['failed']}/{generation['attempted']} questions failed to generate"
        )

    faithfulness = summary.get("ragas", {}).get("faithfulness")
    if faithfulness is None or faithfulness["mean"] is None:
        failures.append("faithfulness could not be scored")
    else:
        total = faithfulness["scored"] + faithfulness["nan"]
        if faithfulness["nan"] > MAX_FAILED_FRACTION * total:
            failures.append(f"faithfulness is NaN for {faithfulness['nan']}/{total} samples")
        if faithfulness["mean"] < FAITHFULNESS_THRESHOLD:
            failures.append(
                f"faithfulness {faithfulness['mean']:.4f} < {FAITHFULNESS_THRESHOLD}"
            )
    return failures


def print_summary(summary: dict):
    print("\n=== Summary ===")
    for stage, stats in summary["retrieval"].items():
        label = {
            "candidates": "Hybrid candidates (before rerank)",
            "final": "Reranked chunks sent to the LLM",
        }[stage]
        scores = "  ".join(f"{k} {v:.3f}" for k, v in stats.items() if k != "questions")
        print(f"{label}, {stats['questions']} questions:\n  {scores}")

    for metric, stats in summary.get("ragas", {}).items():
        mean = "n/a" if stats["mean"] is None else f"{stats['mean']:.4f}"
        warning = f"   <-- {stats['nan']} unscored (NaN)" if stats["nan"] else ""
        print(f"{metric}: {mean} over {stats['scored']} samples{warning}")

    refusals = summary.get("refusals")
    if refusals and refusals["questions"]:
        print(
            f"Unanswerable questions correctly declined: "
            f"{refusals['declined']}/{refusals['questions']}"
        )


def run_evaluation(retrieval_only=False, max_samples=DEFAULT_MAX_SAMPLES,
                   testset_path="testset.csv", prompt_version=None):
    """Run the evaluation, print + save the results and exit non-zero if the gate fails."""
    testset = load_testset(testset_path)
    summary = {"retrieval": {}}

    print("=== Retrieval metrics (deterministic) ===")
    candidates_df = evaluate_candidates(testset)
    candidates_df.to_csv("retrieval_results.csv", index=False)
    summary["retrieval"]["candidates"] = {
        "questions": len(candidates_df),
        **candidates_df.drop(columns="question").mean().round(4).to_dict(),
    }

    if not retrieval_only:
        print("\n=== Building evaluation dataset ===")
        sample = select_sample(testset, max_samples)
        eval_list = generate_responses(sample, prompt_version)
        summary["generation"] = {
            "attempted": len(sample),
            "failed": len(sample) - len(eval_list),
        }

        final_df = evaluate_final_contexts(eval_list)
        if len(final_df):
            summary["retrieval"]["final"] = {
                "questions": len(final_df),
                **final_df.drop(columns="question").mean().round(4).to_dict(),
            }

        unanswerable = [s for s in eval_list if not s["reference_contexts"]]
        summary["refusals"] = {
            "questions": len(unanswerable),
            "declined": sum(is_no_answer(s["response"]) for s in unanswerable),
        }

        answerable = [s for s in eval_list if s["reference_contexts"]]
        if answerable:
            print("\n=== Running RAGAS evaluation ===")
            result_df = run_ragas(answerable)

            # Persist results
            result_df.to_csv("evaluation_results.csv", index=False)
            print("\nResults saved to evaluation_results.csv")
            summary["ragas"] = summarise_ragas(result_df)

    print_summary(summary)
    with open("evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    #quality gate
    failures = check_quality_gate(summary)
    if failures:
        print("\nQuality Gate Failed:")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print("\nQuality Gate Passed")
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the LegalEase RAG pipeline")
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="only the deterministic retrieval metrics (no API keys needed)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=DEFAULT_MAX_SAMPLES,
        help="questions sent through the LLM pipeline and RAGAS (0 = all)",
    )
    parser.add_argument("--testset", default="testset.csv")
    parser.add_argument(
        "--prompt-version", default=None,
        help="prompt version from prompts.yaml (default: active_version)",
    )
    args = parser.parse_args()
    run_evaluation(args.retrieval_only, args.max_samples, args.testset, args.prompt_version)
