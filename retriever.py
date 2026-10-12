import os
from functools import lru_cache
from typing import Optional, Sequence
from dotenv import load_dotenv
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_cohere import CohereRerank
from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from ingest_data import ingest_new_data
from text_utils import tokenize

load_dotenv()

# How many candidates each retriever contributes before reranking, and how many
# chunks the reranker keeps for the LLM.
VECTOR_K = int(os.getenv("VECTOR_K", 25))
BM25_K = int(os.getenv("BM25_K", 25))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", 6))
RERANK_MODEL = os.getenv("RERANK_MODEL", "rerank-v4.0-pro")


class StatuteExpander(BaseDocumentCompressor):
    """Swap clause-sized statute pieces for the full section they came from.

    Statutes are indexed in pieces so every clause is searchable, but the LLM
    should read the whole section. Pieces of the same section collapse into one.
    """

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        expanded, seen_sections = [], set()
        for doc in documents:
            parent_text = doc.metadata.get("parent_text")
            if not parent_text:
                expanded.append(doc)
                continue
            if doc.metadata.get("source") in seen_sections:
                continue
            seen_sections.add(doc.metadata.get("source"))
            metadata = {k: v for k, v in doc.metadata.items() if k != "parent_text"}
            expanded.append(Document(page_content=parent_text, metadata=metadata))
        return expanded


def get_hybrid_retriever(vectorstore=None):
    """Vector + BM25 candidates, before reranking. Needs no API key."""
    if vectorstore is None:
        vectorstore = ingest_new_data()

    ##semantic search
    vector_retriever = vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k" : VECTOR_K}
    )

    ##keyword search
    # Built in memory from the vectorstore so it can never go stale (takes well under a second)
    raw_data = vectorstore.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw_data["documents"], raw_data["metadatas"])
    ]
    if not docs:
        # BM25 cannot be built over an empty corpus
        return vector_retriever

    bm25_retriever = BM25Retriever.from_documents(
        documents=docs,
        bm25_params={"k1": 1.2, "b": 0.75},
        preprocess_func=tokenize,
        k=BM25_K
    )

    ##hybrid search
    hybrid_retriever = EnsembleRetriever(
        retrievers = [vector_retriever, bm25_retriever],
        weights = [0.5, 0.5],
        c = 60
    )

    return hybrid_retriever


@lru_cache(maxsize=1)
def get_retriever() -> ContextualCompressionRetriever:
    hybrid_retriever = get_hybrid_retriever()

    ##Reranking
    reranker = CohereRerank(
        model = RERANK_MODEL,
        top_n = RERANK_TOP_N
    )

    compression_retriever = ContextualCompressionRetriever(
        base_retriever = hybrid_retriever,
        base_compressor = DocumentCompressorPipeline(
            transformers = [reranker, StatuteExpander()]
        )
    )

    return compression_retriever
