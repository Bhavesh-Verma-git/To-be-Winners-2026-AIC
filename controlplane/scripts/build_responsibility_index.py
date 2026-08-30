"""
Rebuild the Responsibility vector store with a real local embedder.

The shipped `Responsiblity Agent/data/chroma_db` was built with a hashing
`FallbackEmbedder` (384-dim random projections) - unusable for semantic search.
This script re-embeds `data/chunk_store.json` with all-MiniLM-L6-v2 into a fresh
Chroma collection at `data/chroma_db_local/` (collection `compliance_local_minilm`).

Optionally (if NEO4J_* is set and reachable) it also pushes the shipped
`data/graph_triples.json` into Neo4j using the same Cypher as the original
`graph_store._write_triples_to_neo4j`.

Run once:  python -m controlplane.scripts.build_responsibility_index
           python -m controlplane.scripts.build_responsibility_index --neo4j
"""

from __future__ import annotations

import argparse
import json
import shutil
import time

from controlplane.config import settings

_COLLECTION = "compliance_local_minilm"


def build_chroma(rebuild: bool = True) -> None:
    import numpy as np

    from controlplane.retrievers.registry import get_minilm

    p = settings.paths
    with open(p["resp_chunks"], "r", encoding="utf-8") as f:
        store = json.load(f)

    model = get_minilm()
    if model is None:
        raise RuntimeError("sentence-transformers / MiniLM not available")

    ids = list(store.keys())
    texts = [store[c].get("text", "") for c in ids]
    t0 = time.time()
    embs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    print(f"  embedded {len(ids)} chunks in {time.time() - t0:.1f}s")

    # always write the portable MiniLM matrix (no chromadb dependency)
    np.savez_compressed(p["resp_matrix"], matrix=np.asarray(embs, dtype="float32"), ids=np.array(ids, dtype=object))
    print(f"[OK] MiniLM matrix -> {p['resp_matrix']}")

    # additionally build a Chroma collection when chromadb is installed
    try:
        import chromadb
    except Exception:
        print("[skip] chromadb not installed - the .npz matrix will be used at runtime.")
        return

    target = p["resp_chroma_local"]
    if rebuild and target.exists():
        shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(target))
    try:
        client.delete_collection(_COLLECTION)
    except Exception:
        pass
    coll = client.create_collection(_COLLECTION, metadata={"hnsw:space": "cosine"})

    B = 500
    for i in range(0, len(ids), B):
        sl = slice(i, i + B)
        metas = []
        for c in ids[sl]:
            rec = store[c]
            metas.append(
                {
                    "law_or_article": rec.get("law_or_article", "") or "",
                    "doc_title": rec.get("doc_title", "") or "",
                    "heading_hierarchy": rec.get("heading_hierarchy", "") or "",
                    "pages": ",".join(map(str, rec.get("page_numbers", []) or [])),
                }
            )
        coll.add(ids=ids[sl], documents=texts[sl], embeddings=[e.tolist() for e in embs[sl]], metadatas=metas)

    print(f"[OK] Chroma collection '{_COLLECTION}' built: {coll.count()} vectors -> {target}")


def push_neo4j() -> None:
    if not settings.neo4j_uri:
        print("[skip] NEO4J_URI not set.")
        return
    from controlplane.responsibility.neo4j_util import get_driver

    p = settings.paths
    with open(p["resp_triples"], "r", encoding="utf-8") as f:
        triples = json.load(f)

    driver = get_driver(verify=True)
    if driver is None:
        print("[warn] Neo4j unreachable - skipped. The responsibility KB will use the "
              "cached graph_triples.json fallback (pipeline still works).")
        return
    cypher = """
    UNWIND $batch AS row
    MERGE (s:Entity {name: row.source})
      ON CREATE SET s.type=row.source_type, s.chunk_id=row.chunk_id,
                    s.doc_title=row.doc_title, s.heading_path=row.heading_path,
                    s.law_or_article=row.law_or_article
    MERGE (t:Entity {name: row.target})
      ON CREATE SET t.type=row.target_type, t.chunk_id=row.chunk_id,
                    t.doc_title=row.doc_title, t.heading_path=row.heading_path,
                    t.law_or_article=row.law_or_article
    MERGE (s)-[r:RELATED_TO {type: row.relationship, chunk_id: row.chunk_id}]->(t)
    """
    with driver.session(database=settings.neo4j_database) as s:
        for i in range(0, len(triples), 200):
            s.run(cypher, batch=triples[i : i + 200])
    driver.close()
    print(f"[OK] pushed {len(triples)} triples to Neo4j at {settings.neo4j_uri}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neo4j", action="store_true", help="also push graph triples to Neo4j")
    ap.add_argument("--keep", action="store_true", help="do not wipe an existing local chroma dir")
    args = ap.parse_args()

    build_chroma(rebuild=not args.keep)
    if args.neo4j:
        try:
            push_neo4j()
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Neo4j push failed ({exc}); the vector store + BM25 + cached "
                  f"graph triples are ready, so the pipeline still works.")


if __name__ == "__main__":
    main()
