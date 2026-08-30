import hashlib
from typing import List, Dict, Any, Callable, Optional

# Multi-path import fallback for RecursiveCharacterTextSplitter
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        class RecursiveCharacterTextSplitter:  # type: ignore
            """Pure Python recursive character text splitter fallback."""
            def __init__(
                self,
                chunk_size: int = 1000,
                chunk_overlap: int = 150,
                length_function: Callable[[str], int] = len,
                separators: Optional[List[str]] = None
            ):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                self.length_function = length_function
                self.separators = separators or [
                    "\n\nArticle ", "\n\nCHAPTER ", "\n\nSection ",
                    "\n\n", "\n", "; ", ". ", " ", ""
                ]

            def _split_text_with_separator(self, text: str, separator: str) -> List[str]:
                if separator:
                    splits = text.split(separator)
                else:
                    splits = list(text)
                return [s for s in splits if s != ""]

            def split_text(self, text: str) -> List[str]:
                if not text or not text.strip():
                    return []
                
                final_chunks: List[str] = []
                # Find appropriate separator
                separator = self.separators[-1]
                new_separators = []
                for i, sep in enumerate(self.separators):
                    if sep == "":
                        separator = ""
                        break
                    if sep in text:
                        separator = sep
                        new_separators = self.separators[i + 1:]
                        break

                splits = self._split_text_with_separator(text, separator)
                good_splits: List[str] = []
                
                for s in splits:
                    if self.length_function(s) < self.chunk_size:
                        good_splits.append(s)
                    else:
                        if good_splits:
                            merged = self._merge_splits(good_splits, separator)
                            final_chunks.extend(merged)
                            good_splits = []
                        if not new_separators:
                            final_chunks.append(s)
                        else:
                            other_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=self.chunk_size,
                                chunk_overlap=self.chunk_overlap,
                                length_function=self.length_function,
                                separators=new_separators
                            )
                            final_chunks.extend(other_splitter.split_text(s))
                
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                
                return final_chunks

            def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
                docs: List[str] = []
                current_doc: List[str] = []
                total = 0
                for d in splits:
                    _len = self.length_function(d)
                    if total + _len + (len(separator) if current_doc else 0) > self.chunk_size:
                        if total > 0:
                            doc = separator.join(current_doc)
                            if doc:
                                docs.append(doc)
                            # Keep overlap
                            while total > self.chunk_overlap or (total + _len + len(separator) > self.chunk_size and total > 0):
                                total -= self.length_function(current_doc[0]) + (len(separator) if len(current_doc) > 1 else 0)
                                current_doc = current_doc[1:]
                    current_doc.append(d)
                    total += _len + (len(separator) if len(current_doc) > 1 else 0)
                doc = separator.join(current_doc)
                if doc:
                    docs.append(doc)
                return docs

from .chunk_store import Chunk, ChunkStore
from .pdf_parser import ParsedSection, PDFDocumentParser
from .generic_parser import GenericPDFParser


class HierarchicalChunker:
    """
    Hierarchical chunker that applies RecursiveCharacterTextSplitter
    while strictly maintaining heading hierarchies (h1->h2->h3...),
    article/law identifiers, page numbers, and table structural integrity.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        table_chunk_size: int = 1200,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.table_chunk_size = table_chunk_size

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n\nArticle ",
                "\n\nCHAPTER ",
                "\n\nSection ",
                "\n\n(",
                "\n\n",
                "\n",
                ";\n",
                ". ",
                ", ",
                " ",
                ""
            ]
        )

        self.table_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.table_chunk_size,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", "; ", " ", ""]
        )

    def _generate_chunk_id(self, doc_slug: str, sec_idx: int, chunk_idx: int, text: str) -> str:
        """Generate a deterministic, unique chunk ID."""
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return f"{doc_slug}_s{sec_idx:03d}_c{chunk_idx:03d}_{content_hash}"

    def chunk_section(self, section: ParsedSection, sec_idx: int, doc_slug: str) -> List[Chunk]:
        """Splits a single parsed section into hierarchical chunks."""
        splitter = self.table_splitter if section.content_type == "table" else self.text_splitter
        raw_chunks = splitter.split_text(section.text)

        if not raw_chunks:
            return []

        chunks: List[Chunk] = []
        for c_idx, text_segment in enumerate(raw_chunks):
            chunk_id = self._generate_chunk_id(doc_slug, sec_idx, c_idx, text_segment)

            # Build enriched metadata dictionary
            metadata: Dict[str, Any] = {
                "chunk_id": chunk_id,
                "doc_title": section.doc_title,
                "source_file": section.source_file,
                "heading_hierarchy": section.heading_hierarchy,
                "heading_path": " > ".join(section.headings),
                "headings": section.headings,
                "law_or_article": section.law_or_article,
                "page_numbers": section.page_numbers,
                "content_type": section.content_type,
                "section_index": sec_idx,
                "chunk_index": c_idx,
                "char_length": len(text_segment),
                **section.metadata
            }

            chunk = Chunk(
                chunk_id=chunk_id,
                text=text_segment,
                doc_title=section.doc_title,
                source_file=section.source_file,
                heading_path=" > ".join(section.headings),
                headings=section.headings,
                heading_hierarchy=section.heading_hierarchy,
                law_or_article=section.law_or_article,
                page_numbers=section.page_numbers,
                content_type=section.content_type,
                metadata=metadata
            )
            chunks.append(chunk)

        return chunks

    def process_sections(self, sections: List[ParsedSection], doc_slug: str) -> List[Chunk]:
        """Processes all sections of a document into chunks."""
        all_chunks: List[Chunk] = []
        for sec_idx, section in enumerate(sections):
            section_chunks = self.chunk_section(section, sec_idx, doc_slug)
            all_chunks.extend(section_chunks)
        return all_chunks

    def ingest_documents(
        self,
        nist_pdf_path: str,
        eu_ai_act_pdf_path: str,
        chunk_store: ChunkStore
    ) -> List[Chunk]:
        """Parses both documents, creates chunks with hierarchy, and stores in ChunkStore."""
        parser = PDFDocumentParser()
        
        nist_sections = parser.parse_nist(nist_pdf_path)
        nist_chunks = self.process_sections(nist_sections, doc_slug="nist")

        eu_sections = parser.parse_eu_ai_act(eu_ai_act_pdf_path)
        eu_chunks = self.process_sections(eu_sections, doc_slug="eu_act")

        all_chunks = nist_chunks + eu_chunks
        chunk_store.add_chunks(all_chunks)

        return all_chunks

    def ingest_additional_documents(self, pdf_docs: List[tuple], chunk_store: ChunkStore) -> List[Chunk]:
        """
        Parses additional generic policy PDFs and appends them to the ChunkStore.
        pdf_docs is a list of tuples: (pdf_path, doc_title, doc_slug)
        """
        parser = GenericPDFParser()
        all_new_chunks = []

        for pdf_path, doc_title, doc_slug in pdf_docs:
            sections = parser.parse(pdf_path, doc_title)
            chunks = self.process_sections(sections, doc_slug=doc_slug)
            all_new_chunks.extend(chunks)

        chunk_store.add_chunks(all_new_chunks)
        return all_new_chunks
