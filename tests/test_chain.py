import asyncio

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.retrievers import BaseRetriever

from main import build_rag_chain, condense_question, load_prompt_config, prepare_custom_prompt
from retriever import StatuteExpander

SECTION = "Section 139 in The Negotiable Instruments Act, 1881\n139. Presumption in favour of holder."


class FakeRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager):
        return [
            Document(page_content=SECTION, metadata={"filename": "Section_139.PDF"}),
            Document(page_content="Rangappa vs Sri Mohan\nThe presumption is rebuttable."),
        ]


class BrokenLLM:
    def invoke(self, *args, **kwargs):
        raise RuntimeError("rate limited")

    __call__ = invoke


@pytest.mark.parametrize("version", list(load_prompt_config()["prompts"]))
def test_every_prompt_version_builds_and_runs(version):
    chain = build_rag_chain(
        retriever=FakeRetriever(), llm=FakeListChatModel(responses=["an answer"]), prompt_version=version
    )
    response = chain.invoke({"input": "What does Section 139 presume?"})
    assert response["answer"] == "an answer"
    assert len(response["context"]) == 2


def test_chain_runs_async_like_the_evaluation():
    chain = build_rag_chain(retriever=FakeRetriever(), llm=FakeListChatModel(responses=["an answer"]))
    response = asyncio.run(chain.ainvoke({"input": "What does Section 139 presume?"}))
    assert response["answer"] == "an answer"


def test_custom_prompt_without_context_placeholder():
    system_prompt = prepare_custom_prompt('Answer briefly. Use JSON like {"answer": "..."}.')
    assert system_prompt.endswith("{context}")
    chain = build_rag_chain(FakeRetriever(), FakeListChatModel(responses=["ok"]), system_prompt, "{input}")
    assert chain.invoke({"input": "q"})["answer"] == "ok"


def test_custom_prompt_keeps_its_own_placeholders():
    system_prompt = prepare_custom_prompt("Context:\n{context}\n\nQuestion: {input}")
    assert system_prompt == "Context:\n{context}\n\nQuestion: {input}"


def test_condense_question():
    history = [("What is the punishment under Section 138?", "Up to two years, or a fine, or both.")]
    llm = FakeListChatModel(responses=['"What is the notice period under Section 138?"'])
    assert condense_question(llm, history, "and the notice period?") == "What is the notice period under Section 138?"


def test_condense_question_falls_back_to_the_original():
    history = [("q", "a")]
    assert condense_question(FakeListChatModel(responses=["unused"]), [], "first question") == "first question"
    assert condense_question(BrokenLLM(), history, "follow up?") == "follow up?"
    # The model answered instead of rewriting
    rambling = FakeListChatModel(responses=["The notice period is thirty days.\n\nKey Takeaway: ..."])
    assert condense_question(rambling, history, "follow up?") == "follow up?"
    assert condense_question(FakeListChatModel(responses=["  "]), history, "follow up?") == "follow up?"


def test_statute_expander_restores_full_section_once():
    full = SECTION + "\nIt shall be presumed, unless the contrary is proved ..."
    docs = [
        Document(page_content="piece 1", metadata={"source": "data/s139.pdf", "parent_text": full, "doc_type": "statute"}),
        Document(page_content="a judgment chunk", metadata={"source": "data/case.pdf", "doc_type": "judgement"}),
        Document(page_content="piece 2", metadata={"source": "data/s139.pdf", "parent_text": full, "doc_type": "statute"}),
    ]
    expanded = StatuteExpander().compress_documents(docs, "query")
    assert [d.page_content for d in expanded] == [full, "a judgment chunk"]
    assert "parent_text" not in expanded[0].metadata
    assert expanded[0].metadata["doc_type"] == "statute"
