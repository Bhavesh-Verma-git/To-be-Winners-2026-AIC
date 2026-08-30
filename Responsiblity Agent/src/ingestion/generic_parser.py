import re
from pathlib import Path
from typing import List
from .pdf_parser import ParsedSection

try:
    import pypdf
except ImportError:
    try:
        import PyPDF2 as pypdf  # type: ignore
    except ImportError:
        pypdf = None  # type: ignore

class GenericPDFParser:
    """A generic parser for standard policy/legal documents."""
    
    def __init__(self):
        self.chapter_pattern = re.compile(r"^(CHAPTER\s+[IVXLCDM]+.*)", re.IGNORECASE)
        self.section_pattern = re.compile(r"^(Section\s+\d+.*)", re.IGNORECASE)
        self.article_pattern = re.compile(r"^(Article\s+\d+.*)", re.IGNORECASE)

    def parse(self, pdf_path: str, doc_title: str) -> List[ParsedSection]:
        pdf_path_obj = Path(pdf_path)
        reader = pypdf.PdfReader(str(pdf_path_obj))
        source_file = pdf_path_obj.name

        sections: List[ParsedSection] = []
        
        current_h1 = doc_title
        current_h2 = "General Overview"
        current_h3 = ""
        current_art = ""
        current_pages = []
        current_lines = []

        def flush_section():
            nonlocal current_lines, current_pages, current_h1, current_h2, current_h3, current_art
            text = "\n".join(current_lines).strip()
            if text:
                headings = [h for h in [current_h1, current_h2, current_h3] if h]
                hierarchy_str = " -> ".join([f"h{i+1}: {h}" for i, h in enumerate(headings)])
                sections.append(ParsedSection(
                    doc_title=doc_title,
                    source_file=source_file,
                    headings=headings,
                    heading_hierarchy=hierarchy_str,
                    law_or_article=current_art,
                    page_numbers=sorted(list(set(current_pages))),
                    content_type="text",
                    text=text,
                    metadata={"doc_type": "GENERIC_POLICY"}
                ))
            current_lines = []
            current_pages = []

        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            raw_text = page.extract_text() or ""
            lines = raw_text.splitlines()

            for line in lines:
                line_str = line.strip()
                if not line_str:
                    continue

                chap_match = self.chapter_pattern.match(line_str)
                if chap_match:
                    flush_section()
                    current_h1 = chap_match.group(1)
                    current_h2 = ""
                    current_h3 = ""
                    current_art = current_h1
                    current_pages.append(page_num)
                    continue

                sec_match = self.section_pattern.match(line_str)
                if sec_match:
                    flush_section()
                    current_h2 = sec_match.group(1)
                    current_h3 = ""
                    current_art = current_h2
                    current_pages.append(page_num)
                    continue

                art_match = self.article_pattern.match(line_str)
                if art_match:
                    flush_section()
                    current_h3 = art_match.group(1)
                    current_art = current_h3
                    current_pages.append(page_num)
                    continue

                current_pages.append(page_num)
                current_lines.append(line_str)

        flush_section()
        return sections
