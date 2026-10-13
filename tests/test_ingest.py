import os
import re

from ingest_data import CHUNK_SIZE, DATA_DIR, _clean_text, build_chunks

SECTION_138 = os.path.join(DATA_DIR, "Section_138_in_The_Negotiable_Instruments_Act_1881.PDF")
SECTION_139 = os.path.join(DATA_DIR, "Section_139_in_The_Negotiable_Instruments_Act_1881.PDF")
RANGAPPA = os.path.join(DATA_DIR, "Rangappa_vs_Sri_Mohan_on_7_May_2010.PDF")


def test_clean_text_rejoins_broken_lines():
    assert _clean_text("liabil-\nity of the\ndrawer.\nNext") == "liability of the drawer.\nNext"


def test_long_statute_is_split_but_keeps_full_section():
    chunks = build_chunks(SECTION_138)
    assert len(chunks) > 1
    full_text = chunks[0].metadata["parent_text"]
    assert "within fifteen days of the receipt of the said notice" in full_text
    assert "legally enforceable debt or other liability" in full_text
    for chunk in chunks:
        assert chunk.metadata["doc_type"] == "statute"
        assert chunk.metadata["parent_text"] == full_text
        assert chunk.page_content.startswith("Section 138 in The Negotiable Instruments Act, 1881")
    # The clauses that used to fall outside the embedding window now have their own piece
    assert any("fifteen days" in c.page_content for c in chunks)
    assert any("Explanation" in c.page_content for c in chunks)


def test_short_statute_stays_one_chunk():
    chunks = build_chunks(SECTION_139)
    assert len(chunks) == 1
    assert "It shall be presumed" in chunks[0].page_content


def test_judgement_chunks_are_clean_and_labelled():
    chunks = build_chunks(RANGAPPA)
    title = "Rangappa vs Sri Mohan on 7 May, 2010"
    assert len(chunks) > 10
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)
    pages = [c.metadata["page"] for c in chunks]
    assert pages == sorted(pages) and pages[0] == 0 and pages[-1] == 10
    for chunk in chunks:
        assert chunk.metadata["doc_type"] == "judgement"
        assert chunk.metadata["title"] == title
        assert chunk.page_content.startswith(title)
        assert "Indian Kanoon" not in chunk.page_content
        assert not re.search("[€□‚ƒ„]", chunk.page_content)
        assert len(chunk.page_content) <= CHUNK_SIZE + len(title) + 1
        assert "parent_text" not in chunk.metadata


def test_ingest_syncs_added_changed_and_deleted_files(tmp_path, monkeypatch):
    import shutil

    from langchain_core.embeddings import DeterministicFakeEmbedding

    import ingest_data

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(ingest_data, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(ingest_data, "PERSIST_DIR", str(tmp_path / "chroma_db"))
    monkeypatch.setattr(
        ingest_data, "HuggingFaceEmbeddings", lambda model_name: DeterministicFakeEmbedding(size=16)
    )

    def indexed():
        stored = ingest_data.init_vectorstore().get(include=["metadatas"])
        files = sorted({meta["filename"] for meta in stored["metadatas"]})
        return files, len(stored["ids"])

    # Empty corpus, then one statute plus files that must be ignored or skipped
    assert indexed() == ([], 0)
    empty_store = ingest_data.ingest_new_data()
    from retriever import get_hybrid_retriever
    assert get_hybrid_retriever(empty_store).invoke("anything") == []
    shutil.copy(SECTION_139, data_dir / "Section_139.pdf")
    (data_dir / ".DS_Store").write_bytes(b"\x00\x01")
    (data_dir / "notes.txt").write_text("not a pdf")
    (data_dir / "broken.pdf").write_bytes(b"this is not really a pdf")
    ingest_data.ingest_new_data()
    assert indexed() == (["Section_139.pdf"], 1)

    # Unchanged file: nothing is re-embedded
    calls = []
    original = ingest_data.build_chunks
    monkeypatch.setattr(ingest_data, "build_chunks", lambda path: calls.append(path) or original(path))
    ingest_data.ingest_new_data()
    assert [c for c in calls if c.endswith("Section_139.pdf")] == []

    # Changed contents under the same name: old chunks are replaced, not duplicated
    shutil.copy(SECTION_138, data_dir / "Section_139.pdf")
    ingest_data.ingest_new_data()
    files, count = indexed()
    assert files == ["Section_139.pdf"] and count == len(build_chunks(SECTION_138))
    stored = ingest_data.init_vectorstore().get(include=["documents"])
    assert all("Section 138" in text for text in stored["documents"])

    # Deleted file: its chunks are removed
    (data_dir / "Section_139.pdf").unlink()
    ingest_data.ingest_new_data()
    assert indexed() == ([], 0)
