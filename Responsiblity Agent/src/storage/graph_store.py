import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple

try:
    from neo4j import GraphDatabase, Driver
except ImportError:
    GraphDatabase = None  # type: ignore
    Driver = None  # type: ignore

try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.schema import Document
    except ImportError:
        from dataclasses import dataclass, field
        @dataclass
        class Document:  # type: ignore
            page_content: str
            metadata: Dict[str, Any] = field(default_factory=dict)

from ..ingestion.chunk_store import Chunk, ChunkStore
from ..config import settings

logger = logging.getLogger(__name__)


def get_graph_transformer_llm():
    """
    Returns an initialized LLM specifically for LLMGraphTransformer.
    Priority is based on settings.GRAPH_LLM_PROVIDER.
    """
    provider = settings.GRAPH_LLM_PROVIDER.lower()
    
    def try_gemini():
        if settings.GOOGLE_API_KEY and settings.GOOGLE_API_KEY.strip():
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=settings.GEMINI_MODEL,
                    temperature=0.0,
                    google_api_key=settings.GOOGLE_API_KEY
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini LLM: {e}")
        return None

    def try_groq():
        if settings.GROQ_API_KEY and settings.GROQ_API_KEY.strip():
            try:
                from langchain_groq import ChatGroq
                return ChatGroq(
                    model=settings.GROQ_GRAPH_MODEL,
                    temperature=0.0,
                    groq_api_key=settings.GROQ_API_KEY,
                    max_retries=1
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Groq LLM for Graph Transformer: {e}")
        return None

    def try_openai():
        if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.strip():
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=settings.OPENAI_MODEL,
                    temperature=0.0,
                    openai_api_key=settings.OPENAI_API_KEY
                )
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI LLM: {e}")
        return None

    if provider == "gemini":
        return try_gemini() or try_groq() or try_openai()
    elif provider == "openai":
        return try_openai() or try_gemini() or try_groq()
    else:
        # Default to groq as preferred for graph
        return try_groq() or try_gemini() or try_openai()


class KnowledgeGraphManager:
    """
    Manages Knowledge Graph creation and querying with Neo4j.
    Uses LLMGraphTransformer (powered by Groq Qwen) to extract entities and relationships,
    tags each node and edge with chunk metadata (especially unique chunk_id),
    and retrieves top-k chunks via graph node matching.
    """

    def __init__(
        self,
        uri: str = settings.NEO4J_URI,
        username: str = settings.NEO4J_USERNAME,
        password: str = settings.NEO4J_PASSWORD,
        database: str = settings.NEO4J_DATABASE,
        cache_path: Path | str = settings.GRAPH_CACHE_PATH
    ):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.cache_path = Path(cache_path)
        self.driver: Optional[Driver] = None
        self.is_connected = False
        self._triples_cache: List[Dict[str, Any]] = []

        self._init_neo4j_driver()
        self._load_cache()

    def _init_neo4j_driver(self):
        """Attempts connection to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            self.driver.verify_connectivity()
            self.is_connected = True
            logger.info(f"Connected to Neo4j instance at {self.uri}")
        except Exception as e:
            self.is_connected = False
            self.driver = None
            logger.warning(f"Neo4j not reachable at {self.uri} ({e}). Graph will operate in local cached mode.")

    def close(self):
        if self.driver:
            self.driver.close()

    def _load_cache(self):
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._triples_cache = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load graph cache: {e}")

    def _save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._triples_cache, f, indent=2, ensure_ascii=False)

    def _extract_domain_entities_and_triples(self, chunk: Chunk) -> List[Dict[str, Any]]:
        """
        Domain-aware entity and relationship extractor for AI governance & compliance.
        Complements and supports LLMGraphTransformer.
        """
        triples = []
        text = chunk.text

        # Extract Laws/Articles/Standards
        primary_entity = chunk.law_or_article or (chunk.headings[-1] if chunk.headings else chunk.doc_title)
        
        # Identify key domain concepts
        concept_patterns = [
            ("Prohibited AI Practice", r"prohibited|subliminal|manipulat|social scoring|exploit vulnerability|biometric categorization|real-time remote biometric"),
            ("High-Risk AI System", r"high-risk|critical infrastructure|employment|law enforcement|migration|asylum|administration of justice"),
            ("Transparency Obligation", r"transparency|deepfake|synthetic|watermark|disclosure|generative ai"),
            ("Risk Management", r"risk management|mitigation|residual risk|risk tolerance|governance"),
            ("Data Governance", r"training data|validation data|testing data|bias|data quality"),
            ("Human Oversight", r"human oversight|human-in-the-loop|human autonomy|stop button"),
            ("Accuracy & Cybersecurity", r"accuracy|robustness|cybersecurity|resilience|vulnerability"),
            ("NIST Core Function", r"GOVERN|MAP|MEASURE|MANAGE"),
            ("Trustworthiness Characteristic", r"valid and reliable|safe|secure and resilient|accountable|explainable|privacy-enhanced|fair"),
            ("Systemic Risk", r"systemic risk|general-purpose ai model|10\^25 FLOPs"),
            ("Penalties & Fines", r"fines|penalties|infringement|sanctions|35 000 000|7 %")
        ]

        for concept_name, pattern in concept_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                triples.append({
                    "source": primary_entity,
                    "source_type": "ComplianceStandard",
                    "relationship": "GOVERNS_OR_DEFINES",
                    "target": concept_name,
                    "target_type": "RiskConcept",
                    "chunk_id": chunk.chunk_id,
                    "doc_title": chunk.doc_title,
                    "heading_path": chunk.heading_path,
                    "law_or_article": chunk.law_or_article,
                    "pages": chunk.page_numbers
                })

        # Add section hierarchy relationship
        if len(chunk.headings) > 1:
            for i in range(len(chunk.headings) - 1):
                triples.append({
                    "source": chunk.headings[i],
                    "source_type": "HeadingNode",
                    "relationship": "CONTAINS_SECTION",
                    "target": chunk.headings[i+1],
                    "target_type": "SubHeadingNode",
                    "chunk_id": chunk.chunk_id,
                    "doc_title": chunk.doc_title,
                    "heading_path": chunk.heading_path,
                    "law_or_article": chunk.law_or_article,
                    "pages": chunk.page_numbers
                })

        return triples

    def build_graph_from_chunks(self, chunks: List[Chunk], use_llm_transformer: bool = True):
        """
        Transforms chunks into entities and relationships using LLMGraphTransformer (Groq Qwen)
        and domain extraction, attaching unique chunk_id and metadata to every node and relation.
        """
        all_triples: List[Dict[str, Any]] = []

        llm = get_graph_transformer_llm() if use_llm_transformer else None
        llm_transformer = None
        if llm:
            try:
                from langchain_experimental.graph_transformers import LLMGraphTransformer
                llm_transformer = LLMGraphTransformer(llm=llm)
                logger.info(f"Initialized LLMGraphTransformer with model: {getattr(llm, 'model_name', getattr(llm, 'model', 'Groq Qwen'))}")
            except Exception as e:
                logger.warning(f"Could not load LLMGraphTransformer: {e}")

        logger.info(f"Extracting graph entities & relationships from {len(chunks)} chunks...")

        for idx, chunk in enumerate(chunks):
            # 1. Rule & Domain Extraction
            domain_triples = self._extract_domain_entities_and_triples(chunk)
            all_triples.extend(domain_triples)

            # 2. LLM Graph Transformer Extraction (with Groq Qwen)
            if llm_transformer and (idx < 2 or "Article 5" in chunk.law_or_article):
                try:
                    doc = Document(
                        page_content=chunk.text[:500],
                        metadata={
                            "chunk_id": chunk.chunk_id,
                            "law_or_article": chunk.law_or_article,
                            "doc_title": chunk.doc_title,
                            "heading_path": chunk.heading_path
                        }
                    )
                    graph_docs = llm_transformer.convert_to_graph_documents([doc])
                    for gdoc in graph_docs:
                        for rel in gdoc.relationships:
                            all_triples.append({
                                "source": rel.source.id,
                                "source_type": rel.source.type or "Entity",
                                "relationship": rel.type,
                                "target": rel.target.id,
                                "target_type": rel.target.type or "Entity",
                                "chunk_id": chunk.chunk_id,
                                "doc_title": chunk.doc_title,
                                "heading_path": chunk.heading_path,
                                "law_or_article": chunk.law_or_article,
                                "pages": chunk.page_numbers
                            })
                except Exception as e:
                    logger.warning(f"LLM graph extraction rate-limited or error for chunk {chunk.chunk_id}: {e}")
                    # Disable further LLM graph calls to prevent blocking on rate limits
                    llm_transformer = None

        self._triples_cache = all_triples
        self._save_cache()
        logger.info(f"Extracted {len(all_triples)} total graph triples.")

        # Persist to Neo4j if live
        if self.is_connected and self.driver:
            self._write_triples_to_neo4j(all_triples)

    def _write_triples_to_neo4j(self, triples: List[Dict[str, Any]]):
        """Ingests nodes and relationships into live Neo4j database."""
        logger.info("Writing graph triples to Neo4j database...")
        cypher_query = """
        UNWIND $batch AS row
        MERGE (s:Entity {name: row.source})
        ON CREATE SET 
            s.type = row.source_type,
            s.chunk_id = row.chunk_id,
            s.doc_title = row.doc_title,
            s.heading_path = row.heading_path,
            s.law_or_article = row.law_or_article

        MERGE (t:Entity {name: row.target})
        ON CREATE SET 
            t.type = row.target_type,
            t.chunk_id = row.chunk_id,
            t.doc_title = row.doc_title,
            t.heading_path = row.heading_path,
            t.law_or_article = row.law_or_article

        MERGE (s)-[r:RELATED_TO {type: row.relationship, chunk_id: row.chunk_id}]->(t)
        """
        try:
            with self.driver.session(database=self.database) as session:
                batch_size = 200
                for i in range(0, len(triples), batch_size):
                    batch = triples[i:i+batch_size]
                    session.run(cypher_query, batch=batch)
            logger.info("Successfully populated Neo4j knowledge graph.")
        except Exception as e:
            logger.error(f"Error inserting into Neo4j: {e}")

    def query_graph_for_chunks(
        self,
        query: str,
        k: int = 4,
        chunk_store: Optional[ChunkStore] = None
    ) -> List[Chunk]:
        """
        Retrieves top k nodes from Neo4j (or graph cache) matching the query,
        extracts their unique chunk_ids, and looks up the corresponding Chunks in ChunkStore.
        """
        matched_chunk_ids: List[str] = []

        # 1. Try Live Neo4j Cypher query
        if self.is_connected and self.driver:
            try:
                cypher = """
                MATCH (n:Entity)
                WHERE n.name =~ '(?i).*' + $search_term + '.*' 
                   OR n.law_or_article =~ '(?i).*' + $search_term + '.*'
                   OR n.heading_path =~ '(?i).*' + $search_term + '.*'
                RETURN DISTINCT n.chunk_id AS chunk_id, n.name AS name
                LIMIT $limit
                """
                tokens = re.findall(r"\w+", query)
                with self.driver.session(database=self.database) as session:
                    for token in tokens:
                        if len(token) > 3:
                            result = session.run(cypher, search_term=token, limit=k)
                            for record in result:
                                cid = record["chunk_id"]
                                if cid and cid not in matched_chunk_ids:
                                    matched_chunk_ids.append(cid)
                                if len(matched_chunk_ids) >= k:
                                    break
                        if len(matched_chunk_ids) >= k:
                            break
            except Exception as e:
                logger.warning(f"Neo4j query error: {e}")

        # 2. Graph Cache / Triple Search fallback or supplement
        if len(matched_chunk_ids) < k and self._triples_cache:
            tokens = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 2]
            scored_chunks: Dict[str, int] = {}

            for triple in self._triples_cache:
                s = triple["source"].lower()
                t = triple["target"].lower()
                rel = triple["relationship"].lower()
                h = triple.get("heading_path", "").lower()
                art = triple.get("law_or_article", "").lower()
                cid = triple.get("chunk_id", "")

                score = 0
                for token in tokens:
                    if token in s or token in t:
                        score += 3
                    if token in art:
                        score += 4
                    if token in h:
                        score += 2
                    if token in rel:
                        score += 1

                if score > 0 and cid:
                    scored_chunks[cid] = scored_chunks.get(cid, 0) + score

            sorted_cids = sorted(scored_chunks.keys(), key=lambda c: scored_chunks[c], reverse=True)
            for cid in sorted_cids:
                if cid not in matched_chunk_ids:
                    matched_chunk_ids.append(cid)
                if len(matched_chunk_ids) >= k:
                    break

        if chunk_store:
            return chunk_store.get_chunks(matched_chunk_ids[:k])
        return []
