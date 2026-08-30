"""
Build a BM25 index for the HR Policy KB (the repo ships FAISS only).

Output: rag_agents/hr_policy/faiss_index/bm25_parents.pkl
        {"bm25": BM25Okapi, "parent_ids": [...]}   over the parent sections.

Run once:  python -m controlplane.scripts.build_hr_bm25
"""

from __future__ import annotations

import json
import pickle
import re

from rank_bm25 import BM25Okapi

from controlplane.config import settings


def _tok(text: str):
    return [t for t in re.sub(r"[^\w\s-]", " ", (text or "").lower()).split() if len(t) > 1]


def main() -> None:
    p = settings.paths
    with open(p["hr_parents"], "r", encoding="utf-8") as f:
        parents = json.load(f)

    parent_ids = list(parents.keys())
    corpus = [
        _tok(f"{parents[pid].get('section_title', '')} {parents[pid].get('content', '')}")
        for pid in parent_ids
    ]
    bm25 = BM25Okapi(corpus)

    out = p["hr_bm25"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"bm25": bm25, "parent_ids": parent_ids}, f)

    print(f"[OK] HR Policy BM25 built over {len(parent_ids)} parent sections -> {out}")


if __name__ == "__main__":
    main()
