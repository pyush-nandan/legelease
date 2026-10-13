from text_utils import (
    contains_passage,
    fix_pdf_characters,
    is_no_answer,
    split_reference_contexts,
    strip_page_furniture,
    tokenize,
)


def test_tokenize_ignores_case_and_punctuation():
    assert tokenize("Section 138, of the Act.") == ["section", "138", "of", "the", "act"]
    assert tokenize("'stop payment'?") == ["stop", "payment"]


def test_no_answer_detects_prompt_refusals():
    assert is_no_answer("The documents I have access to don't fully address this question. Here is ...")
    assert is_no_answer("The provided documents do not contain sufficient legal context to address this.")
    assert is_no_answer("Insufficient regulatory context to determine compliance.")
    assert is_no_answer("The documents I have access to don’t fully address this question.")


def test_no_answer_ignores_insufficient_funds():
    answer = "Section 138 applies when the cheque is returned because funds are insufficient."
    assert not is_no_answer(answer)
    assert not is_no_answer("The cheque did not contain a signature.")


def test_fix_pdf_characters():
    assert fix_pdf_characters("bank€s slip") == "bank's slip"
    assert fix_pdf_characters("of €stop payment□ to") == "of 'stop payment' to"
    assert fix_pdf_characters("(‚the NI Act□)") == "('the NI Act')"
    assert fix_pdf_characters("ƒcause of action„") == '"cause of action"'
    assert fix_pdf_characters("Explanation.€For the, □debt‚") == "Explanation. - For the, 'debt'"
    assert "€" not in fix_pdf_characters("€€€.€€J. (T.S. Thakur)")


def test_strip_page_furniture_removes_footer_and_title():
    page = (
        "the cheque was dishonoured\n"
        "Rangappa vs Sri Mohan on 7 May, 2010\n"
        "Indian Kanoon - http://indiankanoon.org/doc/150051/ 2"
    )
    assert strip_page_furniture(page, "Rangappa vs Sri Mohan on 7 May, 2010") == "the cheque was dishonoured"
    # The title at the top of the first page is kept
    first = "Rangappa vs Sri Mohan on 7 May, 2010\nEquivalent citations: AIR 2010"
    assert strip_page_furniture(first, "Rangappa vs Sri Mohan on 7 May, 2010") == first


def test_split_reference_contexts():
    context = (
        "It shall be presumed, unless the contrary is proved, that the holder received the cheque"
        " + An accused need not examine himself... the standard of proof is preponderance of probabilities"
    )
    passages = split_reference_contexts(context)
    assert len(passages) == 2  # the short "An accused need not examine himself" is dropped
    assert passages[0].startswith("It shall be presumed")
    assert split_reference_contexts("") == []


def test_contains_passage_tolerates_formatting():
    text = "Held:\nthe drawer of the cheque in the instant case\nis only the 1st petitioner, and so on."
    assert contains_passage(text, "the drawer of the cheque in the instant case is only the 1st petitioner")
    assert not contains_passage(text, "the complaint is filed without signature of the authorised person")


def test_no_answer_accepts_reworded_refusals():
    assert is_no_answer("The documents do not fully address this question.")
    assert is_no_answer("The documents I have don't fully address your question, but here is what I can share.")
