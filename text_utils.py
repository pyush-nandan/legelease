"""Small text helpers shared by ingestion, retrieval, the app and evaluation.

Kept free of heavy imports so they can be unit-tested without loading models.
"""
import re
from rapidfuzz import fuzz

CONTEXT_SEPARATOR = " + "

# Phrases the prompts in prompts.yaml tell the model to use when the context
# cannot answer the question. Deliberately specific: a bare "insufficient"
# would match every answer about "insufficient funds".
NO_ANSWER_PHRASES = [
    "i don't know",
    "don't have enough information",
    "cannot answer",
    "don't fully address this question",
    "do not contain sufficient",
    "insufficient regulatory context",
]

_FOOTER = re.compile(
    r"^\s*Indian Kanoon - http://indiankanoon\.org/doc/\d+/\s*\d*\s*$", re.MULTILINE
)

# PyPDF decodes the quotes and dashes in some Indian Kanoon PDFs to these characters
_CHAR_FIXES = {
    "□": "'",   # □  closing quote / apostrophe
    "‚": "'",   # ‚  opening single quote
    "ƒ": '"',   # ƒ  opening double quote
    "„": '"',   # „  closing double quote
    "†": " - ",  # †  dash
    "‡": " - ",  # ‡  dash
    "ˆ": " - ",  # ˆ  dash
}


def tokenize(text: str) -> list[str]:
    """Lowercased alphanumeric tokens for BM25, so "138," / "138." / "Section" / "section" match."""
    return re.findall(r"[a-z0-9]+", text.lower())


def is_no_answer(answer: str) -> bool:
    """True when the answer uses one of the refusal phrases from prompts.yaml."""
    low = answer.lower().replace("’", "'")
    return any(phrase in low for phrase in NO_ANSWER_PHRASES)


def fix_pdf_characters(text: str) -> str:
    """Replace mis-decoded quote/dash characters with plain ASCII equivalents."""
    text = text.replace("vis-‡-vis", "vis-a-vis")
    # Runs of € are the dotted filler above a judge's signature
    text = re.sub(r"[€.]*€{2,}[€.]*", "... ", text)
    # € between a word and a possessive "s" is an apostrophe ("bank€s slip")
    text = re.sub(r"(?<=\w)€(?=s\b)", "'", text)
    # € opening a short quoted term that closes with □ is an opening quote
    text = re.sub(r"(?<![\w.])€(?=\w[^€□\n.;:]{0,40}□)", "'", text)
    text = text.replace("€", " - ")
    for bad, good in _CHAR_FIXES.items():
        text = text.replace(bad, good)
    return text


def strip_page_furniture(page_text: str, title: str = "") -> str:
    """Remove the Indian Kanoon footer and the document title repeated at the foot of each page."""
    page_text = _FOOTER.sub("", page_text).rstrip()
    if title:
        title_pattern = r"\s+".join(re.escape(word) for word in title.split())
        page_text = re.sub(r"\n\s*" + title_pattern + r"\s*$", "", page_text)
    return page_text


def normalize_for_match(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def split_reference_contexts(context: str, min_chars: int = 40) -> list[str]:
    """Split a testset `context` cell into the verbatim passages it quotes.

    Passages are joined with " + " and elided with "..." in testset.csv.
    """
    passages = []
    for part in context.split(CONTEXT_SEPARATOR):
        for passage in re.split(r"\.\.\.|…", part):
            passage = passage.strip()
            if len(passage) >= min_chars:
                passages.append(passage)
    return passages


def contains_passage(text: str, passage: str, threshold: int = 90) -> bool:
    """True when `passage` appears in `text`, tolerating whitespace/punctuation/quote differences."""
    return fuzz.partial_ratio(normalize_for_match(passage), normalize_for_match(text)) >= threshold
