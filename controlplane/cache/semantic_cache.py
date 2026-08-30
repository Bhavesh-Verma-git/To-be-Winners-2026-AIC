"""
Semantic query cache.

* Embeds the (guardrail-processed) query with the shared MiniLM model.
* Keeps entries {query, embedding, answer, route, meta, ts} in memory for O(n)
  cosine lookup.
* **In-memory by default** - every process/app start begins EMPTY. Within one
  running session: a fresh query runs the full pipeline; a later similar query
  (cosine >= threshold) returns the cached answer and skips RAG entirely.
  Set `CP_CACHE_PERSIST=1` to also mirror entries to disk across restarts.

Only *safe, non-HITL, non-retried* answers are cached (see graph/nodes/finalize).
Guardrails always run before the cache so an injected query can neither be served
from nor written to it.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from controlplane.config import settings


@dataclass
class CacheHit:
    answer: str
    similarity: float
    meta: Dict[str, Any]
    query: str


class SemanticCache:
    def __init__(self, store_dir: Optional[Path] = None, threshold: Optional[float] = None,
                 persist: Optional[bool] = None) -> None:
        self.store_dir = Path(store_dir or settings.cache_store_dir)
        self.persist = settings.cache_persist if persist is None else persist
        self.file = self.store_dir / "entries.jsonl"
        self.threshold = threshold if threshold is not None else settings.cache_threshold
        self._entries: List[Dict[str, Any]] = []
        self._matrix = None
        self._lock = threading.Lock()
        self._loaded = False

    # ---- lifecycle -------------------------------------------------------------
    def load(self) -> "SemanticCache":
        if self._loaded:
            return self
        with self._lock:
            if self._loaded:
                return self
            entries: List[Dict[str, Any]] = []
            if self.persist and self.file.exists():
                for line in self.file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except Exception:
                            continue
            self._entries = entries
            self._rebuild_matrix()
            self._loaded = True
        return self

    def _rebuild_matrix(self) -> None:
        try:
            import numpy as np

            self._matrix = (
                np.asarray([e["embedding"] for e in self._entries], dtype="float32")
                if self._entries
                else None
            )
        except Exception:
            self._matrix = None

    # ---- embedding -------------------------------------------------------------
    @staticmethod
    def _embed(text: str):
        from controlplane.retrievers.registry import get_minilm

        model = get_minilm()
        if model is None:
            return None
        return model.encode([text], normalize_embeddings=True)[0]

    # ---- lookup -------------------------------------------------------------
    def lookup(self, query: str) -> Optional[CacheHit]:
        if not settings.cache_enabled:
            return None
        self.load()
        if not self._entries or self._matrix is None:
            return None
        import numpy as np

        q = self._embed(query)
        if q is None:
            return None
        sims = self._matrix @ np.asarray(q, dtype="float32")
        best = int(np.argmax(sims))
        score = float(sims[best])
        if score >= self.threshold:
            e = self._entries[best]
            return CacheHit(answer=e["answer"], similarity=score, meta=e.get("meta", {}), query=e["query"])
        return None

    # ---- write -------------------------------------------------------------
    def add(self, query: str, answer: str, meta: Optional[Dict[str, Any]] = None) -> bool:
        if not settings.cache_enabled or not answer.strip():
            return False
        self.load()
        emb = self._embed(query)
        if emb is None:
            return False
        entry = {
            "query": query,
            "embedding": [float(x) for x in emb],
            "answer": answer,
            "route": (meta or {}).get("selected_kb"),
            "meta": meta or {},
            "ts": time.time(),
        }
        with self._lock:
            self._entries.append(entry)
            self._rebuild_matrix()
            if self.persist:
                self.store_dir.mkdir(parents=True, exist_ok=True)
                with open(self.file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return True

    def clear(self) -> None:
        with self._lock:
            self._entries = []
            self._matrix = None
            if self.persist and self.file.exists():
                self.file.unlink()

    def entries(self) -> List[Dict[str, Any]]:
        self.load()
        return [{"query": e["query"], "route": e.get("route"), "ts": e.get("ts")} for e in self._entries]

    def __len__(self) -> int:
        self.load()
        return len(self._entries)


_cache: Optional[SemanticCache] = None


def get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache
