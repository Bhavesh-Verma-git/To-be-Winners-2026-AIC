"""
Toxicity / safety ensemble - 3 models, loaded ONCE, scored in PARALLEL. No LLM.

  * Detoxify('original')                     -> multi-label toxicity (torch)
  * unitary/toxic-bert                        -> HF text-classification
  * s-nlp/roberta_toxicity_classifier         -> HF text-classification

Each returns {prob, label}. `toxicity_max` = max prob across the three
(conservative / safety-biased, matching the original architecture note).
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from controlplane.config import settings

_TOX_LABELS = {"toxic", "toxicity", "hate", "offensive", "insult", "obscene", "threat",
               "identity_attack", "identity_hate", "severe_toxicity", "label_1", "1"}

_DETOXIFY_STEMS = {
    "original": "toxic_original",
    "unbiased": "toxic_debiased",
    "multilingual": "multilingual_debiased",
}


def _detoxify_cached(variant: str) -> bool:
    """True if the checkpoint is already downloaded, so runtime never blocks on a download."""
    if os.getenv("CP_DETOXIFY_ALLOW_DOWNLOAD", "").lower() in {"1", "true", "yes"}:
        return True
    if os.getenv("CP_DISABLE_DETOXIFY", "").lower() in {"1", "true", "yes"}:
        return False
    stem = _DETOXIFY_STEMS.get(variant, "toxic_original")
    hub = Path(os.path.expanduser("~/.cache/torch/hub/checkpoints"))
    if not hub.exists():
        return False
    return any(f.name.startswith(stem) and not f.name.endswith(".partial") for f in hub.iterdir())


class ToxicityEnsemble:
    def __init__(self) -> None:
        self._detoxify = None
        self._toxic_bert = None
        self._roberta = None
        self._lock = threading.Lock()
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            errors = {}
            if _detoxify_cached(settings.detoxify_variant):
                try:
                    from detoxify import Detoxify

                    self._detoxify = Detoxify(settings.detoxify_variant)
                except Exception as e:  # noqa: BLE001
                    errors["detoxify"] = str(e)
            else:
                errors["detoxify"] = "checkpoint not cached (set CP_DETOXIFY_ALLOW_DOWNLOAD=1 or pre-download)"
            try:
                from transformers import pipeline

                self._toxic_bert = pipeline(
                    "text-classification", model="unitary/toxic-bert", top_k=None, device=-1
                )
            except Exception as e:  # noqa: BLE001
                errors["toxic_bert"] = str(e)
            try:
                from transformers import pipeline

                self._roberta = pipeline(
                    "text-classification", model="s-nlp/roberta_toxicity_classifier", top_k=None, device=-1
                )
            except Exception as e:  # noqa: BLE001
                errors["roberta"] = str(e)
            self._errors = errors
            self._loaded = True

    # ---- per-model scorers -------------------------------------------------------------
    def _score_detoxify(self, text: str) -> Dict[str, Any]:
        if self._detoxify is None:
            return {"prob": 0.0, "label": "unavailable", "scores": {}}
        raw = {k: float(v) for k, v in self._detoxify.predict(text).items()}
        prob = float(raw.get("toxicity", max(raw.values()) if raw else 0.0))
        return {"prob": round(prob, 4), "label": "toxic" if prob >= 0.5 else "non-toxic", "scores": raw}

    @staticmethod
    def _pick(preds) -> Dict[str, Any]:
        # preds: list[list[{label,score}]] or list[{label,score}]
        rows = preds[0] if preds and isinstance(preds[0], list) else preds
        tox = 0.0
        best_label = "non-toxic"
        for r in rows or []:
            lbl = str(r.get("label", "")).lower()
            sc = float(r.get("score", 0.0))
            if lbl in _TOX_LABELS or "tox" in lbl or "hate" in lbl or "offens" in lbl:
                if sc > tox:
                    tox, best_label = sc, "toxic"
            if lbl in {"neutral", "non-toxic", "label_0", "0", "nothate", "clean"} and sc > (1 - tox):
                pass
        return {"prob": round(tox, 4), "label": "toxic" if tox >= 0.5 else "non-toxic"}

    def _score_toxic_bert(self, text: str) -> Dict[str, Any]:
        if self._toxic_bert is None:
            return {"prob": 0.0, "label": "unavailable"}
        return self._pick(self._toxic_bert(text, truncation=True))

    def _score_roberta(self, text: str) -> Dict[str, Any]:
        if self._roberta is None:
            return {"prob": 0.0, "label": "unavailable"}
        return self._pick(self._roberta(text, truncation=True))

    # ---- public -------------------------------------------------------------
    async def score(self, text: str) -> Dict[str, Any]:
        import asyncio

        self.load()
        t0 = time.perf_counter()
        dx, tb, rb = await asyncio.gather(
            asyncio.to_thread(self._score_detoxify, text),
            asyncio.to_thread(self._score_toxic_bert, text),
            asyncio.to_thread(self._score_roberta, text),
        )
        probs = [d["prob"] for d in (dx, tb, rb) if d.get("label") != "unavailable"]
        return {
            "detoxify": dx,
            "unitary": tb,
            "snlp": rb,
            "toxicity_max": round(max(probs), 4) if probs else 0.0,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    def score_sync(self, text: str) -> Dict[str, Any]:
        import asyncio

        return asyncio.run(self.score(text))


_ensemble: Optional[ToxicityEnsemble] = None


def get_toxicity_ensemble() -> ToxicityEnsemble:
    global _ensemble
    if _ensemble is None:
        _ensemble = ToxicityEnsemble()
    return _ensemble
