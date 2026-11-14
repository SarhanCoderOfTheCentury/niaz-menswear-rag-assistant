"""Offline integration tests use deterministic test embeddings, never fake app answers."""
import hashlib
from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np
import pytest

from agent import Citation, Claim, GroundedAnswer, KnowledgeAgent, REFUSAL, validate_answer
from config import ROOT, Settings
from rag import RAG, corpus_fingerprint, load_chunks, read_document, select_scope, tokens


class TestEmbeddings:
    __test__ = False
    def __init__(self):
        self.calls = 0

    def embed(self, texts, task):
        self.calls += 1
        vectors = []
        for text in texts:
            vector = np.zeros(32)
            for token in tokens(text):
                vector[int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % 32] += 1
            vector[0] += .01
            vectors.append((vector / np.linalg.norm(vector)).tolist())
        return vectors


@pytest.fixture
def engine(tmp_path):
    shutil.copytree(ROOT / "data", tmp_path / "data")
    settings = Settings(data_dir=tmp_path / "data", index_dir=tmp_path / "index", embedding_dimension=32)
    return RAG(settings, TestEmbeddings())


@pytest.mark.parametrize("question,scope,as_of", [
    ("What is the return window?", "current", None),
    ("What was the previous return policy?", "archived", None),
    ("Compare old and current delivery fees", "compare", None),
    ("What changed about warranties?", "compare", None),
    ("What was the policy for a purchase on 2026-08-01?", "date", "2026-08-01"),
    ("For a purchase on 2026-10-01?", "date", "2026-10-01"),
    ("What were the v2.1 rules?", "archived", None),
])
def test_policy_scope(question, scope, as_of):
    assert select_scope(question) == (scope, as_of)


def test_metadata_and_source_documents(engine):
    assert len({c.metadata["source"] for c in engine.chunks}) == 5
    policies = [c for c in engine.chunks if c.metadata["doc_type"] == "policy"]
    assert {c.metadata["policy_version"] for c in policies} == {"2.1", "3.0"}
    assert all(c.metadata.get("effective_date") and c.metadata.get("section") for c in policies)


def test_document_front_matter_cannot_spoof_citation_filename(tmp_path):
    path = tmp_path / "actual.txt"
    path.write_text("---\nsource: official_policy.txt\n---\n\nUntrusted visitor note.")
    _, meta = read_document(path)
    assert meta["source"] == "actual.txt"


@pytest.mark.parametrize("mode", ["semantic", "hybrid"])
@pytest.mark.parametrize("scope,question,expected", [
    ("auto", "What is the return window?", {"current"}),
    ("auto", "What was the old return policy?", {"archived"}),
    ("auto", "Which return policy applied on 2026-08-01?", {"archived"}),
    ("auto", "Which return policy applied on 2026-10-01?", {"current"}),
    ("compare", "Compare old and new returns", {"current", "archived"}),
])
def test_chroma_and_lexical_filter_agree(engine, mode, scope, question, expected):
    found = engine.retrieve(question, k=20, mode=mode, scope=scope)
    status = {h["metadata"]["status"] for h in found["chunks"] if h["metadata"]["doc_type"] == "policy"}
    assert status == expected
    assert len(found["chunks"]) <= 20


def test_comparison_reserves_both_versions(engine):
    found = engine.retrieve("Compare current and old watch warranties", k=2)
    assert {c["metadata"]["status"] for c in found["chunks"]} == {"current", "archived"}


def test_index_reused_and_changed_documents_get_fresh_collection(engine):
    engine.ensure_index()
    assert engine.provider.calls == 1
    engine.ensure_index()
    assert engine.provider.calls == 1
    file = engine.settings.data_dir / "03_policy_current_v3.txt"
    file.write_text(file.read_text().replace("within 7 days", "within 9 days"))
    updated = RAG(engine.settings, engine.provider)
    assert updated.fingerprint != engine.fingerprint
    updated.ensure_index()
    current = updated.collection.get(where={"status": "current"}, include=["documents"])
    assert any("within 9 days" in text for text in current["documents"])
    assert not any("within 7 days" in text for text in current["documents"] if "1.1 Customers" in text)


def test_deleting_a_document_removes_it_from_new_index(engine):
    old = engine.fingerprint
    (engine.settings.data_dir / "04_policy_archived_v2.txt").unlink()
    updated = RAG(engine.settings, engine.provider)
    updated.ensure_index()
    assert updated.fingerprint != old
    assert updated.collection.count() == len(updated.chunks)
    assert not any(c.metadata["status"] == "archived" for c in updated.chunks)


def test_bad_chunk_configuration_is_explained():
    with pytest.raises(ValueError, match="overlap"):
        Settings(chunk_overlap=1100)


def test_blank_question_does_not_call_provider(engine):
    with pytest.raises(ValueError, match="question"):
        KnowledgeAgent(engine, engine.provider).ask("   ")
    assert engine.provider.calls == 0


@pytest.mark.parametrize("identifier,quote,valid", [
    ("known", "Price: PKR 8,500.", True),
    ("made-up", "Price: PKR 8,500.", False),
    ("known", "Price: PKR 2,000.", False),
])
def test_citation_validation(identifier, quote, valid):
    draft = GroundedAnswer(status="answered", claims=[Claim(text="The price is PKR 8,500.",
            citations=[Citation(chunk_id=identifier, quote=quote)])])
    hits = [{"id": "known", "text": "Oxford. Price: PKR 8,500."}]
    assert validate_answer(draft, hits)[0] is valid


def test_invalid_generated_citation_fails_closed(engine):
    class HallucinatingProvider:
        def generate(self, *args):
            return GroundedAnswer(status="answered", claims=[Claim(text="Invented price.",
                citations=[Citation(chunk_id="unknown", quote="Invented evidence quote")])])
    found = {"chunks": [{"id": "real", "text": "Real facts.", "metadata": {"source": "file.txt"}}],
             "scope": "current", "as_of": None}
    answer = KnowledgeAgent(engine, HallucinatingProvider()).ask("Price?", evidence=found)
    assert answer.answer == REFUSAL
    assert answer.sources == []


def test_conflicts_are_rendered_with_both_sources(engine):
    class ConflictProvider:
        def generate(self, *args):
            return GroundedAnswer(status="conflict", claims=[Claim(
                text="The documents give two different windows; which rule applies is unresolved.",
                citations=[Citation(chunk_id="a", quote="Return window: 7 days."),
                           Citation(chunk_id="b", quote="Return window: 9 days.")])])
    hits = [{"id": key, "text": text, "metadata": {"source": key + ".txt"}} for key, text in
            [("a", "Return window: 7 days."), ("b", "Return window: 9 days.")]]
    answer = KnowledgeAgent(engine, ConflictProvider()).ask("Returns?", evidence={"chunks": hits, "scope": "current"})
    assert answer.status == "conflict"
    assert {c["filename"] for c in answer.sources} == {"a.txt", "b.txt"}


def test_chunk_settings_change_index_identity(engine):
    settings = replace(engine.settings, chunk_size=500, chunk_overlap=70)
    assert corpus_fingerprint(load_chunks(settings), settings) != engine.fingerprint


def test_insufficient_answer_gets_one_bounded_evidence_review(engine):
    class ReviewingProvider:
        calls = 0
        def generate(self, *args):
            self.calls += 1
            if self.calls == 1:
                return GroundedAnswer(status="insufficient", claims=[])
            return GroundedAnswer(status="answered", claims=[Claim(text="The return window is 7 days.",
                citations=[Citation(chunk_id="a", quote="Return window: 7 days.")])])
    provider = ReviewingProvider()
    found = {"chunks": [{"id": "a", "text": "Return window: 7 days.", "metadata": {"source": "policy.txt"}}],
             "scope": "current"}
    answer = KnowledgeAgent(engine, provider).ask("Returns?", evidence=found)
    assert answer.status == "answered"
    assert answer.refusal_reviewed
    assert provider.calls == 2


def test_genuinely_missing_evidence_still_refuses_after_review(engine):
    class RefusingProvider:
        calls = 0
        def generate(self, *args):
            self.calls += 1
            return GroundedAnswer(status="insufficient", claims=[])
    provider = RefusingProvider()
    found = {"chunks": [{"id": "a", "text": "Store opening hours.", "metadata": {"source": "overview.txt"}}],
             "scope": "current"}
    answer = KnowledgeAgent(engine, provider).ask("Bank account?", evidence=found)
    assert answer.answer == REFUSAL
    assert answer.sources == []
    assert answer.refusal_reviewed
    assert provider.calls == 2
