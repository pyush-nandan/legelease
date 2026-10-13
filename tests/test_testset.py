import os

import pytest

from evaluate import load_testset
from ingest_data import DATA_DIR, load_clean_text
from text_utils import contains_passage, split_reference_contexts


@pytest.fixture(scope="module")
def corpus():
    return [
        load_clean_text(os.path.join(DATA_DIR, name))[1]
        for name in sorted(os.listdir(DATA_DIR))
        if name.lower().endswith(".pdf")
    ]


def test_testset_is_valid():
    testset = load_testset()
    assert len(testset) >= 50
    assert testset["answerable"].sum() >= 45
    assert (~testset["answerable"]).sum() >= 3


def test_every_reference_passage_exists_in_the_corpus(corpus):
    """Retrieval metrics are only meaningful if the reference passages are real quotes."""
    testset = load_testset()
    missing = []
    for row in testset[testset["answerable"]].itertuples(index=False):
        passages = split_reference_contexts(row.context)
        assert passages, f"no usable passage for: {row.question}"
        for passage in passages:
            if not any(contains_passage(text, passage) for text in corpus):
                missing.append((row.question[:60], passage[:80]))
    assert not missing, missing


def test_load_testset_rejects_copied_ground_truth(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text('question,ground_truth,context\n"What is X?","What is X?","some context"\n')
    with pytest.raises(ValueError, match="copy of the question"):
        load_testset(str(path))
