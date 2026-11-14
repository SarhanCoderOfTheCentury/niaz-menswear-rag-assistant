"""Load -> chunk -> Gemini embed -> persistent Chroma -> semantic/BM25 RRF.

The policy filter runs BEFORE retrieval in BOTH search paths. Collection names
include a content/config fingerprint: editing a policy cannot silently reuse
stale embeddings. No pickle files or untrusted serialized indexes are loaded.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
import re

import chromadb
from chromadb.config import Settings as ChromaSettings
from filelock import FileLock
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from rank_bm25 import BM25Okapi

from config import Settings

STOPWORDS = set("a an and are as at be by can do does for from how i in is it me my of on or our the this to was were what when where which who with would you your now today please tell have has had there much many".split())


def tokens(text: str) -> list[str]:
    return [word for word in re.findall(r"[\w]+(?:-[\w]+)*", text.lower()) if word not in STOPWORDS]


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict


def read_document(path: Path) -> tuple[str, dict]:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        body = "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    else:
        body = path.read_text(encoding="utf-8-sig")
    metadata = {"source": path.name, "doc_type": "reference", "status": "reference"}
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n", body, flags=re.S)
    if match:
        # The supplied front matter uses simple scalar fields; keep versions as strings.
        for line in match[1].splitlines():
            key, sep, value = line.partition(":")
            if sep:
                metadata[key.strip()] = value.strip().strip("\"'")
        body = body[match.end():]
    # A document cannot rename its own citation source through front matter.
    metadata["source"] = path.name
    metadata["status"] = metadata["status"].lower()
    if metadata["doc_type"] == "policy":
        required = ["policy_version", "effective_date"]
        if metadata["status"] not in {"current", "archived"} or any(not metadata.get(k) for k in required):
            raise ValueError(f"Policy {path.name} needs version, effective_date, and current/archived status.")
        date.fromisoformat(metadata["effective_date"])
        if metadata.get("end_date"):
            date.fromisoformat(metadata["end_date"])
    if not body.strip():
        raise ValueError(f"{path.name} contains no readable text. Use text-based documents.")
    return body.strip(), metadata


def load_chunks(settings: Settings) -> list[Chunk]:
    paths = sorted(p for p in settings.data_dir.iterdir() if p.suffix.lower() in {".txt", ".md", ".pdf"})
    if not paths:
        raise ValueError("No TXT, Markdown, or text-based PDF documents were found in the data folder.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for path in paths:
        body, metadata = read_document(path)
        # Keep each section (and each product heading) attached to its own content.
        sections = re.split(r"(?m)(?=^#{2,3} )", body)
        position = 0
        for section in sections:
            if not section.strip():
                continue
            heading = section.strip().splitlines()[0].lstrip("# ")
            for part in splitter.split_text(section.strip()):
                text = part if part.startswith("#") else f"{heading}\n{part}"
                identifier = hashlib.sha256(f"{path.name}:{position}:{text}".encode()).hexdigest()[:24]
                chunks.append(Chunk(identifier, text, dict(metadata, section=heading, position=position)))
                position += 1
    return chunks


def corpus_fingerprint(chunks: list[Chunk], settings: Settings) -> str:
    data = ["chunker-v1", settings.chunk_size, settings.chunk_overlap,
            settings.embedding_model, settings.embedding_dimension, [asdict(c) for c in chunks]]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:24]


def select_scope(question: str) -> tuple[str, str | None]:
    """Explicit old/current comparisons beat a single-version request.

    An explicit ISO date selects the policy valid on that purchase date. Other
    natural-language dates can be expressed with 'old/previous' or the UI scope.
    """
    q = question.lower()
    old = bool(re.search(r"\b(old|previous|archived|former|historical|originally)\b|\bv?2\.1\b|\b202[45]\b", q))
    current = bool(re.search(r"\b(current|new|now|today|latest)\b|\bv?3\.0\b", q))
    compare = bool(re.search(r"\b(compare|comparison|changed|difference|versus|vs)\b", q))
    if compare or (old and current):
        return "compare", None
    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", q)
    if dates:
        try:
            date.fromisoformat(dates[0])
            return "date", dates[0]
        except ValueError:
            pass
    return ("archived", None) if old else ("current", None)


def allowed(chunk: Chunk, scope: str, as_of: str | None) -> bool:
    meta = chunk.metadata
    if meta.get("doc_type") != "policy":
        return True
    if scope == "compare":
        return True
    if scope == "date":
        return meta["effective_date"] <= as_of and (not meta.get("end_date") or as_of <= meta["end_date"])
    return meta.get("status") == scope


class RAG:
    def __init__(self, settings: Settings, provider):
        self.settings = settings
        self.provider = provider
        self.chunks = load_chunks(settings)
        self.fingerprint = corpus_fingerprint(self.chunks, settings)
        self.collection_name = f"niaz-{self.fingerprint}"
        settings.index_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(settings.index_dir / "chroma"),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = None

    def ensure_index(self, force: bool = False):
        # File locking prevents multiple Streamlit sessions from racing the same rebuild.
        with FileLock(str(self.settings.index_dir / f"{self.collection_name}.lock"), timeout=180):
            try:
                collection = self.client.get_collection(self.collection_name, embedding_function=None)
                if not force and collection.count() == len(self.chunks):
                    self.collection = collection
                    return
            except chromadb.errors.NotFoundError:
                pass
            vectors = self.provider.embed([c.text for c in self.chunks], "RETRIEVAL_DOCUMENT")
            # Generate all embeddings before replacing a collection, so API failures are non-destructive.
            try:
                self.client.delete_collection(self.collection_name)
            except chromadb.errors.NotFoundError:
                pass
            collection = self.client.create_collection(
                self.collection_name, embedding_function=None,
                metadata={"hnsw:space": "cosine", "fingerprint": self.fingerprint},
            )
            collection.add(ids=[c.id for c in self.chunks], documents=[c.text for c in self.chunks],
                           metadatas=[c.metadata for c in self.chunks], embeddings=vectors)
            self.collection = collection

    def retrieve(self, question: str, k: int = 5, mode: str = "hybrid", scope: str = "auto") -> dict:
        if not question.strip():
            raise ValueError("Type a question about Niaz Men's Wear to get started.")
        if not 1 <= k <= 20:
            raise ValueError("Choose between 1 and 20 retrieved chunks.")
        if mode not in {"semantic", "hybrid"} or scope not in {"auto", "current", "archived", "compare"}:
            raise ValueError("Unknown retrieval mode or policy scope.")
        self.ensure_index()
        detected, as_of = select_scope(question)
        active_scope = detected if scope == "auto" else scope
        eligible = [c for c in self.chunks if allowed(c, active_scope, as_of)]
        if not eligible:
            return {"chunks": [], "scope": active_scope, "as_of": as_of, "mode": mode}
        q_vector = self.provider.embed([question], "RETRIEVAL_QUERY")[0]
        # IDs enforce the same filter for semantic and lexical search, even for dated policies.
        result = self.collection.query(
            query_embeddings=[q_vector], ids=[c.id for c in eligible],
            n_results=min(len(eligible), max(12, k * 3)),
            include=["documents", "metadatas", "distances"],
        )
        semantic_ids = result["ids"][0]
        semantic_score = {cid: 1 - float(d) for cid, d in zip(semantic_ids, result["distances"][0])}
        scores = {cid: 1 / (60 + rank) for rank, cid in enumerate(semantic_ids, 1)}
        if mode == "hybrid":
            bm25 = BM25Okapi([tokens(c.text) or ["_"] for c in eligible])
            lexical = bm25.get_scores(tokens(question))
            for rank, index in enumerate(np.argsort(-lexical)[:max(12, k * 3)], 1):
                if lexical[index] > 0:
                    cid = eligible[index].id
                    scores[cid] = scores.get(cid, 0) + 1 / (60 + rank)
        by_id = {c.id: c for c in eligible}
        ordered = sorted(scores, key=lambda cid: (-scores[cid], cid))
        chosen = ordered[:k]
        if active_scope == "compare" and k >= 2:
            # Reserve one relevant chunk from each policy version for a fair comparison.
            reserved = []
            for status in ("current", "archived"):
                best = next((cid for cid in ordered if by_id[cid].metadata.get("doc_type") == "policy"
                             and by_id[cid].metadata.get("status") == status), None)
                if best:
                    reserved.append(best)
            chosen = list(dict.fromkeys(reserved + ordered))[:k]
        hits = [dict(asdict(by_id[cid]), score=round(scores[cid], 6),
                     similarity=round(semantic_score.get(cid, 0), 4)) for cid in chosen]
        return {"chunks": hits, "scope": active_scope, "as_of": as_of, "mode": mode}

    def stats(self) -> dict:
        return {"documents": len({c.metadata['source'] for c in self.chunks}),
                "chunks": len(self.chunks), "fingerprint": self.fingerprint,
                "collection": self.collection_name}


if __name__ == "__main__":
    import argparse
    import os
    from providers import GeminiProvider
    parser = argparse.ArgumentParser(description="Build the local Chroma index from knowledge documents.")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    settings = Settings.from_env()
    engine = RAG(settings, GeminiProvider(os.getenv("GEMINI_API_KEY", ""), settings))
    engine.ensure_index(force=args.rebuild)
    print(json.dumps(engine.stats(), indent=2))
