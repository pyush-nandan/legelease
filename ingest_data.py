import bisect
import hashlib
import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from text_utils import fix_pdf_characters, strip_page_furniture

DATA_DIR = "data"
PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Bump whenever cleaning, chunking or the embedding model changes. The version is
# part of the collection name, so chunks built by an older pipeline are never
# mixed with new ones — the corpus is simply re-ingested into a fresh collection.
INDEX_VERSION = "v2"
COLLECTION_NAME = f"embeddings_{INDEX_VERSION}"

# all-MiniLM-L6-v2 only embeds the first 256 tokens (~1000 characters of this corpus)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
STATUTE_PIECE_SIZE = 900

Judgement_splitter = RecursiveCharacterTextSplitter(
    separators = [
        r"\n(?=\d{1,3}\.\s)",   # numbered paragraphs: "\n12. "
        r"\n",
        r"(?<=[.;:])\s",        # sentence / clause ends
        " "
    ],
    is_separator_regex = True,
    keep_separator = True,
    add_start_index = True,
    chunk_size = CHUNK_SIZE,
    chunk_overlap = CHUNK_OVERLAP
)

Statute_splitter = RecursiveCharacterTextSplitter(
    separators = [
        r"(?=Provided that)",
        r"(?=Explanation)",
        r"(?<=[;:.])(?=\(\w{1,3}\))",   # sub-clauses: ";(b)"
        r"\n",
        r"(?<=[.;:])\s",
        " "
    ],
    is_separator_regex = True,
    keep_separator = True,
    chunk_size = STATUTE_PIECE_SIZE,
    chunk_overlap = 0
)

def _clean_text(text : str) -> str:
    # Rejoin hyphenated line breaks (e.g. "liabil-\nity" → "liability")
    text = re.sub(r"-\n(\S)", r"\1", text)

    # Rejoin lines that are broken mid-word without hyphen
    # Only join if the line doesn't end with sentence-ending punctuation
    text = re.sub(r"(?<![.!?:,])\n(?=[a-z])", " ", text)

    # Collapse 3+ newlines into a paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading/trailing whitespace per line
    text = "\n".join(line.strip() for line in text.splitlines())

    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)

    return text.strip()

def _file_hash(file_path : str) -> str:
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def _join_pages(pages : list[str]) -> tuple[str, list[int]]:
    """Join cleaned pages into one text, returning the offset where each page starts."""
    full_text, page_starts = "", []
    for page_text in pages:
        if page_text and full_text:
            # A sentence that runs over the page break stays on one line
            mid_sentence = full_text[-1] not in ".!?:;" and page_text[0].islower()
            full_text += " " if mid_sentence else "\n"
        page_starts.append(len(full_text))
        full_text += page_text
    return full_text, page_starts

def load_clean_text(file_path : str) -> tuple[str, str, list[int]]:
    """Load one PDF and return (title, cleaned full text, offset of each page in that text)."""
    pages = PyPDFLoader(file_path).load()
    title = (pages[0].metadata.get("title") or "").strip() if pages else ""
    cleaned_pages = [
        _clean_text(fix_pdf_characters(strip_page_furniture(page.page_content, title)))
        for page in pages
    ]
    full_text, page_starts = _join_pages(cleaned_pages)
    title = title or os.path.splitext(os.path.basename(file_path))[0].replace("_", " ")
    return title, full_text, page_starts

def build_chunks(file_path : str) -> list[Document]:
    """Load one PDF and return its cleaned chunks, ready for the vectorstore."""
    file_name = os.path.basename(file_path)
    title, full_text, page_starts = load_clean_text(file_path)
    if not full_text.strip():
        return []

    metadata = {
        "source" : file_path,
        "filename" : file_name,
        "title" : title,
        "file_hash" : _file_hash(file_path)
    }

    if file_name.startswith("Section_"):
        # Long sections are indexed as clause-sized pieces so the whole text is
        # searchable, but every piece carries the full section in `parent_text`
        # and the retriever hands that full section to the LLM.
        body = full_text[len(title):].strip() if full_text.startswith(title) else full_text
        pieces = Statute_splitter.split_text(body) if len(full_text) > STATUTE_PIECE_SIZE else [body]
        chunks = [
            Document(
                page_content = f"{title}\n{piece.strip()}",
                metadata = {**metadata, "doc_type" : "statute", "page" : 0, "parent_text" : full_text}
            )
            for piece in pieces if piece.strip()
        ]
    else:
        chunks = []
        for chunk in Judgement_splitter.create_documents([full_text]):
            text = chunk.page_content.strip()
            if not text:
                continue
            page = max(0, bisect.bisect_right(page_starts, chunk.metadata["start_index"]) - 1)
            chunks.append(
                Document(
                    # The case name leads every chunk so both retrievers and the LLM know its source
                    page_content = f"{title}\n{text}",
                    metadata = {**metadata, "doc_type" : "judgement", "page" : page}
                )
            )

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{file_name}::{i}"
    return chunks

def init_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL
    )

    vectorstore = Chroma(
        collection_name = COLLECTION_NAME,
        embedding_function = embedding_model,
        persist_directory = PERSIST_DIR
    )

    return vectorstore

def ingest_new_data():
    """Bring the vectorstore in sync with the PDFs in ./data.

    New files are added, edited files are re-ingested and chunks of deleted
    files are removed. Unchanged files are skipped.
    """
    vectorstore = init_vectorstore()

    stored = vectorstore.get(include=["metadatas"])
    indexed = {}
    for chunk_id, meta in zip(stored["ids"], stored["metadatas"]):
        entry = indexed.setdefault(meta["source"], {"ids" : [], "file_hash" : meta.get("file_hash")})
        entry["ids"].append(chunk_id)

    file_names = sorted(
        f for f in (os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR) else [])
        if f.lower().endswith(".pdf")
    )
    file_paths = [os.path.join(DATA_DIR, f) for f in file_names]

    for source in set(indexed) - set(file_paths):
        print(f"{source} is no longer in {DATA_DIR}. Removing its chunks")
        vectorstore.delete(ids=indexed[source]["ids"])

    for file_name, file_path in zip(file_names, file_paths):
        existing = indexed.get(file_path)
        try:
            if existing and existing["file_hash"] == _file_hash(file_path):
                # print(f"{file_name} already ingested. Skipping!")
                continue

            print(f"Adding file {file_name} to vectorstore")
            chunks = build_chunks(file_path)
        except Exception as e:
            # One unreadable PDF should not stop the rest of the corpus loading
            print(f"Could not read {file_name}: {e}. Skipping!")
            continue

        if not chunks:
            print(f"No extractable text in {file_name} (scanned PDF?). Skipping!")
            continue

        if existing:
            vectorstore.delete(ids=existing["ids"])
        vectorstore.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
        print(f"Split into {len(chunks)} chunks and added to the vectorstore")

    return vectorstore

if __name__ == "__main__":
    ingest_new_data()
