"""Drives the Streamlit app end to end with the retriever and LLM faked out."""
import os

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.retrievers import BaseRetriever
from streamlit.testing.v1 import AppTest

import ingest_data
import main
import retriever


APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


class FakeRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager):
        return [
            Document(
                page_content="Section 138 in The Negotiable Instruments Act, 1881\n138. Dishonour of cheque ...",
                metadata={"filename": "Section_138.PDF", "page": 0, "doc_type": "statute"},
            )
        ]


class FakeVectorstore:
    def get(self, include=None):
        return {"ids": ["a"], "metadatas": [{"filename": "Section_138.PDF"}]}


def test_app_chat_flow(monkeypatch):
    llm = FakeListChatModel(responses=[
        "An offence arises when the cheque is returned because funds are insufficient.",
        "What is the notice period under Section 138?",
        "The documents I have access to don't fully address this question. Here is what I can share: ...",
    ])
    monkeypatch.setattr(retriever, "get_retriever", lambda: FakeRetriever())
    monkeypatch.setattr(main, "build_llm", lambda: llm)
    monkeypatch.setattr(ingest_data, "init_vectorstore", lambda: FakeVectorstore())

    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert app.title[0].value.endswith("LegalEase")

    # First question: "insufficient funds" must not raise the no-answer banner
    app.chat_input[0].set_value("When is a bounced cheque an offence?").run()
    assert not app.exception
    assert len(app.session_state.messages) == 1
    first = app.session_state.messages[0]
    assert first["answer"].startswith("An offence arises")
    assert first["no_answer"] is False
    assert first["search_query"] is None
    assert first["sources"][0]["filename"] == "Section_138.PDF"
    assert not app.warning

    # Follow-up: rewritten into a standalone question, and a real refusal is flagged
    app.chat_input[0].set_value("and the notice period?").run()
    assert not app.exception
    second = app.session_state.messages[1]
    assert second["question"] == "and the notice period?"
    assert second["search_query"] == "What is the notice period under Section 138?"
    assert second["no_answer"] is True
    assert len(app.warning) == 1


def test_app_uploads_can_be_disabled(monkeypatch):
    monkeypatch.setattr(ingest_data, "init_vectorstore", lambda: FakeVectorstore())
    monkeypatch.setenv("ENABLE_UPLOADS", "false")
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert any("Uploads are disabled" in caption.value for caption in app.sidebar.caption)
