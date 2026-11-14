"""Configuration is shared by the UI, command-line tools, and tests."""
from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    data_dir: Path = ROOT / "data"
    index_dir: Path = ROOT / ".cache"
    llm_model: str = "gemini-3.5-flash-lite"
    embedding_model: str = "gemini-embedding-001"
    embedding_dimension: int = 768
    chunk_size: int = 1100
    chunk_overlap: int = 140
    top_k: int = 5

    def __post_init__(self):
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("Chunk overlap must be smaller than chunk size.")
        if self.chunk_size < 200:
            raise ValueError("Chunk size must be at least 200 characters.")
        if self.embedding_dimension < 1:
            raise ValueError("Embedding dimension must be positive.")

    @classmethod
    def from_env(cls):
        def path(name, default):
            value = Path(os.getenv(name, default))
            return value if value.is_absolute() else ROOT / value
        return cls(
            data_dir=path("DATA_DIR", "data"),
            index_dir=path("INDEX_DIR", ".cache"),
            llm_model=os.getenv("LLM_MODEL", "gemini-3.5-flash-lite"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"),
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "768")),
        )
