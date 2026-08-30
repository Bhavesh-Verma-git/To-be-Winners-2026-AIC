"""
XGBoost hallucination probability - NO LLM.

Uses the *already trained* model shipped in the repo:
    master_router/performance_branch/hallucination_classifier/model/xgb_hallucination_model.json
loaded as-is via `XGBClassifier.load_model` (no retraining).

Feature engineering keeps the exact 12-feature contract the model was trained on
(`feature_engineering.FeatureEngineer`) but is re-implemented here for LATENCY:

  * one shared `FeatureEngineer` instance (embedder + roberta-large-mnli + spaCy),
    warmed at startup;
  * the per-sentence NLI loop is capped to `settings.xgb_max_sentences` AND
    batched into a single pipeline call instead of N sequential calls
    (this is the dominant CPU cost);
  * `CP_NLI_MODEL` can still swap in a smaller NLI model - labels are normalised.

Result matches the training feature order exactly, then the loaded XGBoost model
predicts the hallucination probability directly.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List

import numpy as np

from controlplane.config import settings

_lock = threading.Lock()
_fe = None            # shared FeatureEngineer
_model = None         # XGBClassifier

_FEATURE_ORDER = [
    "overlap_score",
    "nli_entailment_score",
    "nli_contradiction_score",
    "nli_neutral_score",
    "sentence_level_max_contradiction",
    "entity_overlap_ratio",
    "new_entities_count",
    "response_length",
    "context_length",
    "length_ratio",
    "model_tier",
    "temperature",
]

_LABEL_MAP = {
    "entailment": "ENTAILMENT", "label_2": "ENTAILMENT",
    "neutral": "NEUTRAL", "label_1": "NEUTRAL",
    "contradiction": "CONTRADICTION", "label_0": "CONTRADICTION",
}


def _norm_label(lbl: str) -> str:
    l = str(lbl).strip().lower()
    return _LABEL_MAP.get(l, str(lbl).strip().upper())


def _load() -> None:
    global _fe, _model
    if _fe is not None and _model is not None:
        return
    with _lock:
        if _fe is None:
            import master_router.performance_branch.hallucination_classifier.feature_engineering as fe_mod

            target = settings.nli_model.strip()
            _orig_pipeline = fe_mod.pipeline
            if target and target != "roberta-large-mnli":
                # redirect the one pipeline() call in FeatureEngineer.__init__ so we don't
                # load roberta-large just to replace it
                def _patched(task, model=None, **kw):
                    if model == "roberta-large-mnli":
                        model = target
                    kw.pop("model", None)
                    return _orig_pipeline(task, model=model, **kw)

                fe_mod.pipeline = _patched
            try:
                fe = fe_mod.FeatureEngineer(device="cpu")
            finally:
                fe_mod.pipeline = _orig_pipeline
            _fe = fe
        if _model is None:
            from xgboost import XGBClassifier

            m = XGBClassifier()
            m.load_model(str(settings.paths["xgb_model"]))
            _model = m


def warmup() -> None:
    try:
        _load()
        for _ in range(2):  # 2nd call clears torch/thread JIT warmup cost
            score_hallucination("warm up context sentence here.", "warm up response sentence.", "warmup", 0.0)
    except Exception:
        pass


def _nli_batch(pairs: List[tuple[str, str]]) -> List[Dict[str, float]]:
    """One batched pipeline call for many (premise, hypothesis) pairs.

    Proper HF sentence-pair format ({"text": premise, "text_pair": hypothesis});
    the whole list goes in one call so the forward pass is batched instead of N
    sequential calls (the dominant CPU cost).
    """
    if not pairs:
        return []
    inputs = [{"text": p[:1400], "text_pair": h[:1400]} for p, h in pairs]
    try:
        raw = _fe.nli_model(inputs, truncation=True, max_length=512, batch_size=len(inputs))
    except Exception:
        try:
            raw = _fe.nli_model(inputs, truncation=True)
        except Exception:
            raw = [_fe.nli_model(f"{p[:1400]} </s></s> {h[:1400]}", truncation=True, max_length=512)
                   for p, h in pairs]
    out: List[Dict[str, float]] = []
    for row in raw:
        rows = row if isinstance(row, list) else [row]
        out.append({_norm_label(r["label"]): float(r["score"]) for r in rows})
    return out


def _features(context: str, output: str, model_name: str, temperature: float) -> Dict[str, float]:
    from sentence_transformers import util

    context, output = str(context), str(output)
    ctx_len = len(context.split())
    out_len = len(output.split())

    # embeddings overlap
    emb = _fe.embedder.encode([context, output], convert_to_tensor=True, show_progress_bar=False)
    overlap = float(util.cos_sim(emb[0], emb[1]).item())

    # entities
    ctx_ents = _fe._extract_entities(context)
    out_ents = _fe._extract_entities(output)
    new_ents = out_ents - ctx_ents
    ent_overlap = 1.0 if not out_ents else len(out_ents & ctx_ents) / (len(out_ents) + 1e-5)

    # NLI: whole-text + up to N sentences, all in ONE batched call.
    # CP_XGB_MAX_SENTS=0 -> whole-text only (fastest).
    n_sents = max(0, settings.xgb_max_sentences)
    sents: list[str] = []
    if n_sents:
        try:
            from nltk.tokenize import sent_tokenize

            sents = [s for s in sent_tokenize(output) if s.strip()][:n_sents]
        except Exception:
            sents = [s.strip() for s in output.split(".") if s.strip()][:n_sents]

    pairs = [(context, output)] + [(context, s) for s in sents]
    nli = _nli_batch(pairs)
    whole = nli[0] if nli else {}
    ent_s = whole.get("ENTAILMENT", 0.5)
    con_s = whole.get("CONTRADICTION", 0.5)
    neu_s = whole.get("NEUTRAL", 0.5)
    # whole-text contradiction is the floor when the per-sentence loop is disabled
    max_sent_contra = max((row.get("CONTRADICTION", 0.0) for row in nli[1:]), default=con_s)

    return {
        "overlap_score": overlap,
        "nli_entailment_score": ent_s,
        "nli_contradiction_score": con_s,
        "nli_neutral_score": neu_s,
        "sentence_level_max_contradiction": max_sent_contra,
        "entity_overlap_ratio": ent_overlap,
        "new_entities_count": float(len(new_ents)),
        "response_length": float(out_len),
        "context_length": float(ctx_len),
        "length_ratio": out_len / (ctx_len + 1e-5),
        "model_tier": float(_fe._get_model_tier(model_name)),
        "temperature": float(temperature),
    }


def score_hallucination(context: str, answer: str, model_name: str, temperature: float = 0.2) -> Dict[str, Any]:
    _load()
    feats = _features(context or "", answer or "", model_name or "unknown", temperature)
    X = np.array([[feats[f] for f in _FEATURE_ORDER]], dtype="float32")
    prob = float(_model.predict_proba(X)[0][1])
    label = int(_model.predict(X)[0])

    if prob > 0.8:
        risk = "CRITICAL"
    elif prob > 0.5:
        risk = "HIGH"
    elif prob > 0.2:
        risk = "MODERATE"
    else:
        risk = "LOW"

    return {
        "hallucination_probability": prob,
        "is_hallucination": bool(label == 1),
        "risk_level": risk,
        "features": feats,
    }
