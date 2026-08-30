import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

@dataclass
class Chunk:
    chunk_id: str
    text: str
    doc_title: str
    source_file: str
    heading_path: str
    headings: List[str] = field(default_factory=list)
    heading_hierarchy: str = ""  # formatted as h1: ... -> h2: ... -> h3: ...
    law_or_article: str = ""
    page_numbers: List[int] = field(default_factory=list)
    content_type: str = "text"  # 'text' or 'table'
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=data.get("chunk_id", ""),
            text=data.get("text", ""),
            doc_title=data.get("doc_title", ""),
            source_file=data.get("source_file", ""),
            heading_path=data.get("heading_path", ""),
            headings=data.get("headings", []),
            heading_hierarchy=data.get("heading_hierarchy", ""),
            law_or_article=data.get("law_or_article", ""),
            page_numbers=data.get("page_numbers", []),
            content_type=data.get("content_type", "text"),
            metadata=data.get("metadata", {})
        )

    def get_full_context_header(self) -> str:
        """Generates a structured header string for RAG grounding."""
        header_lines = [
            f"=== DOCUMENT: {self.doc_title} ===",
            f"Source File: {self.source_file} (Pages: {', '.join(map(str, self.page_numbers)) if self.page_numbers else 'N/A'})",
            f"Hierarchy: {self.heading_hierarchy or self.heading_path}",
        ]
        if self.law_or_article:
            header_lines.append(f"Law/Article/Section: {self.law_or_article}")
        header_lines.append(f"Content Type: {self.content_type.upper()}")
        header_lines.append(f"Chunk ID: {self.chunk_id}")
        return "\n".join(header_lines)


class ChunkStore:
    """Master Key-Value Store for all ingested chunks."""
    
    def __init__(self):
        self._chunks: Dict[str, Chunk] = {}

    def add_chunk(self, chunk: Chunk):
        self._chunks[chunk.chunk_id] = chunk

    def add_chunks(self, chunks: List[Chunk]):
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self._chunks.get(chunk_id)

    def get_chunks(self, chunk_ids: List[str]) -> List[Chunk]:
        results = []
        for cid in chunk_ids:
            chunk = self._chunks.get(cid)
            if chunk:
                results.append(chunk)
        return results

    def all_chunks(self) -> List[Chunk]:
        return list(self._chunks.values())

    def __len__(self) -> int:
        return len(self._chunks)

    def save_to_json(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {cid: chunk.to_dict() for cid, chunk in self._chunks.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_from_json(self, path: Path | str) -> "ChunkStore":
        path = Path(path)
        if not path.exists():
            return self
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._chunks = {cid: Chunk.from_dict(item) for cid, item in data.items()}
        return self
