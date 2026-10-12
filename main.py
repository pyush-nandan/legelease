import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import yaml
from retriever import get_retriever

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
DOCUMENT_SEPARATOR = "\n\n---\n\n"
CONTEXT_BLOCK = "\n\nRetrieved Context:\n{context}"

CONDENSE_SYSTEM_PROMPT = (
    "Rewrite the user's latest question as a standalone question that can be understood "
    "without the conversation above. Keep section numbers, case names and legal terms "
    "exactly as written. If it is already standalone, return it unchanged. "
    "Return only the question, with no explanation."
)

def build_llm():
    return ChatGroq(
        model_name = LLM_MODEL,
        temperature = 0,
        max_tokens = 1000,
        timeout = None,
        max_retries = 3
    )

def load_prompt_config(path = "prompts.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def prepare_custom_prompt(system_prompt):
    """Make a user-written system prompt safe to use as a template.

    Braces other than {context} and {input} are escaped so they are not read as
    template variables, and the context block is appended if it was left out.
    """
    escaped = system_prompt.replace("{", "{{").replace("}", "}}")
    escaped = re.sub(r"\{\{(context|input)\}\}", r"{\1}", escaped)
    if "{context}" not in escaped:
        escaped += CONTEXT_BLOCK
    return escaped

def build_rag_chain(retriever = None, llm = None, system_prompt = None, human_prompt = None, prompt_version = None):
    """Build the retrieval chain. With no arguments it uses the active prompt in prompts.yaml."""
    if system_prompt is None:
        prompt_dict = load_prompt_config()
        version = prompt_version or prompt_dict["active_version"]
        system_prompt = prompt_dict["prompts"][version]["system"]
        human_prompt = human_prompt or prompt_dict["prompts"][version]["human"]

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt),
        ("human", human_prompt or "{input}")]
    )

    doc_chain = create_stuff_documents_chain(
        llm or build_llm(), prompt, document_separator = DOCUMENT_SEPARATOR
    )
    rag_chain = create_retrieval_chain(retriever or get_retriever(), doc_chain)
    return rag_chain

def condense_question(llm, history, question, max_turns = 3):
    """Rewrite a follow-up question so it can be retrieved on without the chat history.

    `history` is a list of (question, answer) pairs, oldest first. Falls back to
    the original question if there is no history or the rewrite fails.
    """
    if not history:
        return question

    messages = []
    for past_question, past_answer in history[-max_turns:]:
        messages.append(HumanMessage(past_question))
        messages.append(AIMessage(past_answer[:800]))

    prompt = ChatPromptTemplate.from_messages(
        [("system", CONDENSE_SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{question}")]
    )
    try:
        rewritten = (prompt | llm | StrOutputParser()).invoke(
            {"history": messages, "question": question}
        )
    except Exception:
        return question

    rewritten = rewritten.strip().strip('"').strip()
    # A rewrite is one short line; anything else means the model answered instead
    if not rewritten or "\n" in rewritten or len(rewritten) > max(300, 4 * len(question)):
        return question
    return rewritten

def run_rag():
    rag_chain = build_rag_chain()
    llm = build_llm()
    history = []
    flag = True
    while flag:
        query = input("\n\nAsk your question: ")
        standalone_query = condense_question(llm, history, query)
        response = rag_chain.invoke({"input": standalone_query})
        for i, doc in enumerate(response["context"]):
            print(f"Chunk {i+1}: {doc.page_content}")
            print("---")
        print("\n\n\n -----------Answer----------- \n\n\n")
        print(response["answer"])
        history.append((query, response["answer"]))

        while True:
            more_questions = input("\n\nDo you want to ask more questions (y/n): ")
            if more_questions.lower() == "n":
                flag = False
                break
            elif more_questions.lower() == "y":
                break
            else:
                print("Invalid input. Please try again.")

if __name__ == "__main__":
    run_rag()
