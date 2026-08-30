"""
Robust, non-blocking Neo4j driver creation.

The Responsibility KG works fine WITHOUT Neo4j (it falls back to the cached
`graph_triples.json` token scorer), so connecting must never hang the pipeline.

  * tries the configured URI, then `+ssc` (trust self-signed cert), then plain bolt
  * every attempt is hard-capped (~3s) so a bad host can't stall warmup
  * `CP_NEO4J_DISABLE=1` skips it entirely
"""

from __future__ import annotations

import logging
import os
import re
from typing import List, Optional

from controlplane.config import settings

logger = logging.getLogger(__name__)


def _variants(uri: str) -> List[str]:
    uri = uri.strip()
    m = re.match(r"^[a-zA-Z0-9+]+://", uri)
    host = uri[m.end():] if m else uri
    scheme = m.group(0)[:-3] if m else "neo4j"
    out: List[str] = []

    def add(s: str):
        v = f"{s}://{host}"
        if v not in out:
            out.append(v)

    add(scheme)
    if scheme.endswith("+s"):
        add(scheme[:-2] + "+ssc")     # allow self-signed cert
    elif "+s" not in scheme:
        add("neo4j+ssc")
        add("bolt")
    return out[:3]


def get_driver(verify: bool = True):
    if not settings.neo4j_uri or os.getenv("CP_NEO4J_DISABLE", "").lower() in {"1", "true", "yes"}:
        return None
    try:
        from neo4j import GraphDatabase
    except Exception:
        return None

    auth = (settings.neo4j_user, settings.neo4j_password)
    last = None
    for uri in _variants(settings.neo4j_uri):
        drv = None
        try:
            drv = GraphDatabase.driver(
                uri, auth=auth,
                connection_timeout=3,
                connection_acquisition_timeout=4,
                max_transaction_retry_time=4,
            )
            if verify:
                drv.verify_connectivity()
            if uri != settings.neo4j_uri:
                logger.info("Neo4j connected via fallback scheme %s", uri.split("://")[0])
            return drv
        except Exception as exc:  # noqa: BLE001
            last = exc
            try:
                if drv is not None:
                    drv.close()
            except Exception:
                pass
    reason = "auth failed" if last and "Unauthorized" in str(last) else "unreachable"
    print(f"[controlplane] Neo4j {reason} - responsibility KG using the cached graph_triples.json "
          f"(3k+ triples). Set CP_NEO4J_DISABLE=1 to silence, or fix NEO4J_* in .env.")
    return None
