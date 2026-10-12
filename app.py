import streamlit as st
import yaml
import os
from dotenv import load_dotenv

load_dotenv()

# ── Page config (must be first Streamlit command) ───────────────────────
st.set_page_config(
    page_title="LegalEase — Legal research assistant",
    page_icon=":material/gavel:",
    layout="centered",
)


# ═══════════════════════════════════════════════════════════════════════
#  CACHED RESOURCES
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data
def load_prompts():
    """Load all prompt configurations from prompts.yaml."""
    with open("prompts.yaml", "r") as f:
        return yaml.safe_load(f)


@st.cache_resource(show_spinner="Loading knowledge base and models…")
def load_retriever():
    """Cache the full retriever pipeline — embedding model, BM25 index, reranker.

    This is the expensive initialisation that should only happen once.
    """
    from retriever import get_retriever
    return get_retriever()


@st.cache_resource(show_spinner=False)
def load_llm():
    """Cache the Groq LLM client."""
    from main import build_llm
    return build_llm()


def build_chain(retriever, llm, system_prompt, human_prompt):
    """Build a retrieval chain with the given prompts.

    This is cheap — it only wires together already-loaded objects.
    """
    from main import build_rag_chain
    return build_rag_chain(retriever, llm, system_prompt, human_prompt)


@st.cache_data(show_spinner=False)
def get_ingested_documents():
    """Query ChromaDB for all unique ingested document filenames."""
    from ingest_data import init_vectorstore
    try:
        vs = init_vectorstore()
        raw = vs.get(include=["metadatas"])
        filenames = set()
        for meta in raw["metadatas"]:
            if "filename" in meta:
                filenames.add(meta["filename"])
        return sorted(filenames)
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════

st.session_state.setdefault("messages", [])
st.session_state.setdefault("use_custom_prompt", False)
st.session_state.setdefault("custom_system_prompt", "")
st.session_state.setdefault("ingested_this_session", set())

# Uploads write into the knowledge base every visitor shares. Set
# ENABLE_UPLOADS=false on a public deployment to turn them off.
UPLOADS_ENABLED = os.getenv("ENABLE_UPLOADS", "true").strip().lower() not in (
    "false", "0", "no", "off",
)
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", 25))

# Load prompt config
prompt_config = load_prompts()
prompt_versions = list(prompt_config["prompts"].keys())
default_version = prompt_config.get("active_version", prompt_versions[0])


# ═══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:

    # ── Document management ─────────────────────────────────────────────
    st.header(":material/folder_open: Documents", anchor=False)

    if UPLOADS_ENABLED:
        uploaded_files = st.file_uploader(
            "Upload legal documents",
            type=["pdf"],
            accept_multiple_files=True,
        )
    else:
        uploaded_files = []
        st.caption(":material/lock:  Uploads are disabled on this deployment.")

    if uploaded_files:
        new_uploads = [
            f for f in uploaded_files
            if f.name not in st.session_state.ingested_this_session
        ]

        if new_uploads:
            os.makedirs("data", exist_ok=True)
            file_statuses = {}
            existing_docs = set(get_ingested_documents())

            # Save files to ./data
            for uf in new_uploads:
                # Never trust the client-supplied name as a path
                safe_name = os.path.basename(uf.name.replace("\\", "/"))
                filepath = os.path.join("data", safe_name)
                if safe_name != uf.name or not safe_name.lower().endswith(".pdf"):
                    file_statuses[uf.name] = "error:invalid file name"
                elif uf.size > MAX_UPLOAD_MB * 1024 * 1024:
                    file_statuses[uf.name] = (
                        f"error:larger than the {MAX_UPLOAD_MB} MB limit"
                    )
                elif not uf.getvalue().startswith(b"%PDF"):
                    file_statuses[uf.name] = "error:not a valid PDF file"
                elif uf.name in existing_docs and os.path.exists(filepath):
                    file_statuses[uf.name] = "already"
                else:
                    try:
                        with open(filepath, "wb") as f:
                            f.write(uf.getvalue())
                        file_statuses[uf.name] = "saved"
                    except Exception as e:
                        file_statuses[uf.name] = f"error:{e}"

            # Run ingestion for newly saved files
            needs_ingest = [n for n, s in file_statuses.items() if s == "saved"]
            if needs_ingest:
                with st.spinner("Ingesting documents…"):
                    try:
                        from ingest_data import ingest_new_data
                        ingest_new_data()

                        # Clear document list cache so it re-queries ChromaDB
                        get_ingested_documents.clear()
                        new_existing = set(get_ingested_documents())

                        for name in needs_ingest:
                            if name in new_existing:
                                file_statuses[name] = "success"
                            else:
                                file_statuses[name] = "failed"
                                # Don't leave an unreadable file to be retried on every start
                                try:
                                    os.remove(os.path.join("data", name))
                                except OSError:
                                    pass

                        # Clear retriever caches so new docs are picked up
                        load_retriever.clear()
                        from retriever import get_retriever as _gr
                        _gr.cache_clear()

                    except Exception as e:
                        for name in needs_ingest:
                            file_statuses[name] = f"error:{e}"

            # Show per-file status
            for name, status in file_statuses.items():
                if status == "already":
                    st.info(
                        f"`{name}` — already ingested",
                        icon=":material/check_circle:",
                    )
                elif status == "success":
                    st.success(
                        f"`{name}` — successfully ingested",
                        icon=":material/upload_file:",
                    )
                elif status == "failed":
                    st.error(
                        f"`{name}` — ingestion failed",
                        icon=":material/error:",
                    )
                elif status.startswith("error:"):
                    st.error(
                        f"`{name}` — {status[6:]}",
                        icon=":material/error:",
                    )

            st.session_state.ingested_this_session |= {
                f.name for f in new_uploads
            }

    # List all ingested documents
    all_docs = get_ingested_documents()
    if all_docs:
        with st.expander(
            f"{len(all_docs)} documents in knowledge base",
            icon=":material/library_books:",
        ):
            for doc_name in all_docs:
                is_statute = doc_name.startswith("Section_")
                badge = (
                    ":blue-badge[Statute]"
                    if is_statute
                    else ":orange-badge[Case]"
                )
                st.markdown(f"{badge}  `{doc_name}`")
    else:
        st.caption("No documents ingested yet.")

    st.space("large")

    # ── Prompt management ───────────────────────────────────────────────
    st.header(":material/tune: Prompt settings", anchor=False)

    use_custom = st.toggle(
        "Custom prompt mode",
        value=st.session_state.use_custom_prompt,
    )
    st.session_state.use_custom_prompt = use_custom

    if not use_custom:
        selected_version = st.selectbox(
            "Prompt version",
            prompt_versions,
            index=(
                prompt_versions.index(default_version)
                if default_version in prompt_versions
                else 0
            ),
            key="prompt_version_select",
        )

        meta = prompt_config["prompts"][selected_version].get("metadata", {})
        st.caption(
            f":material/info:  {meta.get('description', 'No description')}"
        )
        author = meta.get("author")
        added = meta.get("added_on")
        if author:
            st.caption(
                f":material/person:  {author}"
                + (f" · {added}" if added else "")
            )
    else:
        custom_text = st.text_area(
            "System prompt",
            value=st.session_state.custom_system_prompt,
            height=200,
            placeholder=(
                "Write your custom system prompt here.\n"
                "Include {context} where retrieved documents should be "
                "inserted."
            ),
        )
        st.session_state.custom_system_prompt = custom_text
        st.caption(
            ":material/info:  Include `{context}` for document injection. "
            "Changes apply on the next question."
        )

    st.space("large")

    # ── Clear conversation ──────────────────────────────────────────────
    if st.button(
        "Clear conversation",
        icon=":material/delete:",
        type="secondary",
    ):
        st.session_state.messages = []
        st.rerun()

    st.space("small")
    st.caption("LegalEase · LLaMA 3.1 · Cohere Rerank · ChromaDB")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ═══════════════════════════════════════════════════════════════════════

st.title(":material/gavel: LegalEase", anchor=False)
st.caption(
    " · AI-powered legal research assistant · "
)


# ── Helper: render source chunks ────────────────────────────────────────

def render_sources(sources):
    """Render retrieved source chunks inside a collapsed expander."""
    if not sources:
        return
    with st.expander(
        f"{len(sources)} retrieved sources",
        icon=":material/description:",
    ):
        for i, src in enumerate(sources):
            page_info = (
                f" · Page {src['page'] + 1}"
                if src.get("page") is not None
                else ""
            )
            is_statute = src.get("doc_type") == "statute"
            badge = (
                ":blue-badge[Statute]"
                if is_statute
                else ":orange-badge[Case]"
            )
            st.markdown(
                f"**Source {i + 1}** · {badge} · "
                f"`{src['filename']}`{page_info}"
            )
            text = src["text"]
            st.caption(
                text[:600] + ("…" if len(text) > 600 else "")
            )
            if i < len(sources) - 1:
                st.space("small")


# ── Chat history ────────────────────────────────────────────────────────

for entry in st.session_state.messages:
    with st.chat_message("user"):
        st.markdown(entry["question"])

    with st.chat_message("assistant"):
        if entry.get("search_query"):
            st.caption(f":material/search:  Searched as: {entry['search_query']}")
        if entry.get("no_answer"):
            st.warning(
                "The documents don't contain sufficient information "
                "to fully answer this question.",
                icon=":material/help:",
            )
        st.markdown(entry["answer"])
        render_sources(entry.get("sources", []))


# ── Chat input ──────────────────────────────────────────────────────────

if user_query := st.chat_input("Ask a legal question…"):

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_query)

    # Build and invoke the chain
    with st.chat_message("assistant"):
        with st.spinner("Searching legal knowledge base…"):
            retriever = load_retriever()
            llm = load_llm()

            from main import condense_question, prepare_custom_prompt
            from text_utils import is_no_answer

            # Resolve prompt
            if (
                st.session_state.use_custom_prompt
                and st.session_state.custom_system_prompt.strip()
            ):
                system_p = prepare_custom_prompt(
                    st.session_state.custom_system_prompt
                )
                human_p = "{input}"
            else:
                ver = st.session_state.get(
                    "prompt_version_select", default_version
                )
                system_p = prompt_config["prompts"][ver]["system"]
                human_p = prompt_config["prompts"][ver]["human"]

            # Follow-ups ("what about the notice period?") are rewritten into a
            # standalone question using the recent conversation
            history = [
                (m["question"], m["answer"])
                for m in st.session_state.messages
            ]
            search_query = condense_question(llm, history, user_query)

            chain = build_chain(retriever, llm, system_p, human_p)
            response = chain.invoke({"input": search_query})

        answer = response["answer"]
        sources = [
            {
                "text": doc.page_content,
                "filename": doc.metadata.get("filename", "Unknown"),
                "page": doc.metadata.get("page"),
                "doc_type": doc.metadata.get("doc_type", ""),
            }
            for doc in response["context"]
        ]

        # Detect "no answer" / "I don't know" responses
        no_answer = is_no_answer(answer)
        rewritten_query = search_query if search_query != user_query else None

        if rewritten_query:
            st.caption(f":material/search:  Searched as: {rewritten_query}")

        if no_answer:
            st.warning(
                "The documents don't contain sufficient information "
                "to fully answer this question.",
                icon=":material/help:",
            )

        st.markdown(answer)
        render_sources(sources)

    # Persist to session state
    st.session_state.messages.append({
        "question": user_query,
        "answer": answer,
        "sources": sources,
        "no_answer": no_answer,
        "search_query": rewritten_query,
    })
