"""Gemini generation + cached, normalized Gemini retrieval embeddings.

No local model downloads, credentials in source, or silent fake API fallbacks.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import os
import threading
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np
import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from config import Settings

T = TypeVar("T")


class ProviderError(RuntimeError):
    """Safe, user-facing API error, without a raw response or credential."""


class GeminiProvider:
    def __init__(self, api_key: str, settings: Settings):
        if not api_key or api_key == "your_gemini_api_key_here":
            raise ProviderError("Add GEMINI_API_KEY in your environment or Streamlit secrets.")
        self.settings = settings
        self._generation_lock = threading.Lock()
        self._last_generation = 0.0
        self.min_interval = float(os.getenv("GEMINI_MIN_INTERVAL_SECONDS", "0"))
        self.client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=60000)
        )
        settings.index_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = settings.index_dir / "embeddings.sqlite3"
        with sqlite3.connect(self.cache_path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    def _call(self, fn: Callable[[], T]) -> T:
        for attempt in range(3):
            try:
                return fn()
            except errors.APIError as exc:
                code = int(exc.code or 0)
                if code == 429 and any(term in str(exc).lower() for term in ("per day", "requestsperday", "daily")):
                    raise ProviderError("Gemini daily quota reached. Wait for the quota reset or configure another available free-tier model.") from None
                if (code == 429 or code >= 500) and attempt < 2:
                    time.sleep(10 * (attempt + 1) if code == 429 else 2 ** (attempt + 1))
                    continue
                messages = {
                    400: "Gemini rejected the request. Check the configured model and embedding dimensions.",
                    401: "Gemini authentication failed. Check GEMINI_API_KEY.",
                    403: "Gemini access was denied. Check the API key and project permissions.",
                    404: "The configured Gemini model is unavailable. Update the model setting.",
                    429: "Gemini quota or rate limit reached. Wait and retry, or check your AI Studio quota.",
                }
                raise ProviderError(messages.get(code, "Gemini is temporarily unavailable. Please try again.")) from None
            except (TimeoutError, ConnectionError, OSError, httpx.TimeoutException, httpx.TransportError):
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise ProviderError("Could not reach Gemini. Check the connection and try again.") from None
            except Exception:
                # Some SDK transports raise httpx-specific errors. Do not leak raw headers/URLs.
                raise ProviderError("The Gemini request failed. Check your connection and API configuration.") from None
        raise ProviderError("Gemini could not complete the request.")

    def embed(self, texts: list[str], task: str) -> list[list[float]]:
        """Use different retrieval tasks for documents and questions; normalize 768-D vectors."""
        keys = [hashlib.sha256(json.dumps([
            self.settings.embedding_model, self.settings.embedding_dimension, task, text
        ]).encode()).hexdigest() for text in texts]
        vectors: dict[str, list[float]] = {}
        with sqlite3.connect(self.cache_path) as db:
            for key in keys:
                row = db.execute("SELECT value FROM embeddings WHERE key=?", (key,)).fetchone()
                if row:
                    vectors[key] = json.loads(row[0])
        missing = list(dict.fromkeys(key for key in keys if key not in vectors))
        text_by_key = dict(zip(keys, texts))
        for start in range(0, len(missing), 32):
            batch = missing[start:start + 32]
            response = self._call(lambda: self.client.models.embed_content(
                model=self.settings.embedding_model,
                contents=[text_by_key[key] for key in batch],
                config=types.EmbedContentConfig(
                    task_type=task, output_dimensionality=self.settings.embedding_dimension
                ),
            ))
            if not response.embeddings or len(response.embeddings) != len(batch):
                raise ProviderError("Gemini returned incomplete embeddings. Please retry indexing.")
            for key, item in zip(batch, response.embeddings):
                array = np.asarray(item.values, dtype=np.float32)
                norm = np.linalg.norm(array)
                if len(array) != self.settings.embedding_dimension or not np.isfinite(array).all() or norm == 0:
                    raise ProviderError("Gemini returned an invalid embedding vector.")
                vectors[key] = (array / norm).tolist()
            with sqlite3.connect(self.cache_path) as db:
                db.executemany("INSERT OR REPLACE INTO embeddings VALUES (?, ?)",
                               [(key, json.dumps(vectors[key])) for key in batch])
        return [vectors[key] for key in keys]

    def generate(self, system: str, payload: dict, schema: type[BaseModel]) -> BaseModel:
        with self._generation_lock:
            delay = self.min_interval - (time.monotonic() - self._last_generation)
            if delay > 0:
                time.sleep(min(delay, 30))
            self._last_generation = time.monotonic()
        response = self._call(lambda: self.client.models.generate_content(
            model=self.settings.llm_model,
            contents=json.dumps(payload, ensure_ascii=False),
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0,
                max_output_tokens=4096,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
                    if "2.5" in self.settings.llm_model else None,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        ))
        try:
            return schema.model_validate_json(response.text or "")
        except ValueError:
            raise ProviderError("Gemini returned an incomplete answer. Please try again.") from None
