import re
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field

try:
    import pypdf
except ImportError:
    try:
        import PyPDF2 as pypdf  # type: ignore
    except ImportError:
        pypdf = None  # type: ignore

@dataclass
class ParsedSection:
    doc_title: str
    source_file: str
    headings: List[str]
    heading_hierarchy: str
    law_or_article: str
    page_numbers: List[int]
    content_type: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class PDFDocumentParser:
    """Specialized parser for NIST AI 100-1 and EU AI Act (Regulation EU 2024/1689)."""

    def __init__(self):
        pass

    def parse_nist(self, pdf_path: Path | str) -> List[ParsedSection]:
        pdf_path = Path(pdf_path)
        reader = pypdf.PdfReader(str(pdf_path))
        doc_title = "NIST AI Risk Management Framework (AI RMF 1.0)"
        source_file = pdf_path.name

        sections: List[ParsedSection] = []
        
        current_h1 = "NIST AI RMF 1.0"
        current_h2 = "General Overview"
        current_h3 = ""
        current_art = ""
        current_type = "text"
        current_pages = []
        current_lines = []

        def flush_section():
            nonlocal current_lines, current_pages, current_h1, current_h2, current_h3, current_art, current_type
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
                    content_type=current_type,
                    text=text,
                    metadata={"doc_type": "NIST_RMF", "core_function": current_h2 if current_h2 in ["GOVERN", "MAP", "MEASURE", "MANAGE"] else ""}
                ))
            current_lines = []
            current_pages = []

        h1_patterns = [
            (r"Executive Summary", "Executive Summary"),
            (r"Part 1:\s*Foundational Information", "Part 1: Foundational Information"),
            (r"Framing Risk", "Part 1: Framing Risk"),
            (r"Audience", "Part 1: Audience"),
            (r"AI Risks and Trustworthiness", "Part 1: AI Risks and Trustworthiness"),
            (r"Effectiveness of the AI RMF", "Part 1: Effectiveness of the AI RMF"),
            (r"Part 2:\s*Core and Profiles", "Part 2: Core and Profiles"),
            (r"AI RMF Core", "Part 2: AI RMF Core"),
            (r"AI RMF Profiles", "Part 2: AI RMF Profiles"),
            (r"Appendix A:", "Appendix A: AI Actor Tasks"),
            (r"Appendix B:", "Appendix B: AI Risks vs Traditional Software Risks"),
            (r"Appendix C:", "Appendix C: AI Risk Management & Human-AI Interaction"),
            (r"Appendix D:", "Appendix D: Attributes of the AI RMF"),
        ]

        h2_patterns = [
            (r"Understanding and Addressing Risks, Impacts, and Harms", "Understanding and Addressing Risks"),
            (r"Challenges for AI Risk Management", "Challenges for AI Risk Management"),
            (r"Valid and Reliable", "Trustworthiness: Valid and Reliable"),
            (r"Safe", "Trustworthiness: Safe"),
            (r"Secure and Resilient", "Trustworthiness: Secure and Resilient"),
            (r"Accountable and Transparent", "Trustworthiness: Accountable and Transparent"),
            (r"Explainable and Interpretable", "Trustworthiness: Explainable and Interpretable"),
            (r"Privacy-Enhanced", "Trustworthiness: Privacy-Enhanced"),
            (r"Fair\s*[–-]\s*with Harmful Bias Managed", "Trustworthiness: Fair with Harmful Bias Managed"),
            (r"\bGOVERN\b", "GOVERN"),
            (r"\bMAP\b", "MAP"),
            (r"\bMEASURE\b", "MEASURE"),
            (r"\bMANAGE\b", "MANAGE"),
        ]

        table_pattern = re.compile(r"Table\s+(\d+):\s*(.*)", re.IGNORECASE)
        subcategory_pattern = re.compile(r"\b(GOVERN|MAP|MEASURE|MANAGE)\s*([0-9]+\.[0-9]+)\s*:\s*(.*)", re.IGNORECASE)

        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            raw_text = page.extract_text() or ""
            lines = raw_text.splitlines()

            for line in lines:
                line_str = line.strip()
                if not line_str:
                    continue

                # Ignore recurring header / footer
                if "NIST AI 100-1" in line_str and ("AI RMF" in line_str or "January 2023" in line_str):
                    continue
                if line_str.isdigit() and len(line_str) <= 3:
                    continue

                # Check Table
                t_match = table_pattern.match(line_str)
                if t_match:
                    flush_section()
                    current_type = "table"
                    current_h3 = f"Table {t_match.group(1)}: {t_match.group(2)[:60]}"
                    current_art = f"NIST Table {t_match.group(1)}"
                    current_pages.append(page_num)
                    current_lines.append(line_str)
                    continue

                # Check Subcategory (e.g. GOVERN 1.1)
                sub_match = subcategory_pattern.match(line_str)
                if sub_match:
                    flush_section()
                    func_name = sub_match.group(1).upper()
                    sub_id = sub_match.group(2)
                    current_h2 = func_name
                    current_h3 = f"{func_name} {sub_id}"
                    current_art = f"NIST {func_name} {sub_id}"
                    current_type = "text"
                    current_pages.append(page_num)
                    current_lines.append(line_str)
                    continue

                # Check H1
                matched_h1 = None
                for pat, val in h1_patterns:
                    if re.search(pat, line_str, re.IGNORECASE) and len(line_str) < 80:
                        matched_h1 = val
                        break
                if matched_h1:
                    flush_section()
                    current_h1 = matched_h1
                    current_h2 = "Overview"
                    current_h3 = ""
                    current_art = matched_h1
                    current_type = "text"
                    current_pages.append(page_num)
                    current_lines.append(line_str)
                    continue

                # Check H2
                matched_h2 = None
                for pat, val in h2_patterns:
                    if re.search(pat, line_str, re.IGNORECASE) and len(line_str) < 80:
                        matched_h2 = val
                        break
                if matched_h2:
                    flush_section()
                    current_h2 = matched_h2
                    current_h3 = ""
                    current_art = f"NIST {matched_h2}"
                    current_type = "text"
                    current_pages.append(page_num)
                    current_lines.append(line_str)
                    continue

                current_pages.append(page_num)
                current_lines.append(line_str)

        flush_section()
        return sections

    def parse_eu_ai_act(self, pdf_path: Path | str) -> List[ParsedSection]:
        pdf_path = Path(pdf_path)
        reader = pypdf.PdfReader(str(pdf_path))
        doc_title = "Regulation (EU) 2024/1689 (EU Artificial Intelligence Act)"
        source_file = pdf_path.name

        sections: List[ParsedSection] = []

        current_h1 = "Preamble & Recitals"
        current_h2 = "Recitals"
        current_h3 = ""
        current_art = ""
        current_type = "text"
        current_pages = []
        current_lines = []

        def flush_section():
            nonlocal current_lines, current_pages, current_h1, current_h2, current_h3, current_art, current_type
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
                    content_type=current_type,
                    text=text,
                    metadata={"doc_type": "EU_AI_ACT", "article": current_art}
                ))
            current_lines = []
            current_pages = []

        chapter_pattern = re.compile(r"^CHAPTER\s+([IVXLCDM]+)\s*$", re.IGNORECASE)
        section_pattern = re.compile(r"^Section\s+(\d+)\s*$", re.IGNORECASE)
        article_pattern = re.compile(r"^Article\s+(\d+)\s*$", re.IGNORECASE)
        annex_pattern = re.compile(r"^ANNEX\s+([IVXLCDM]+)\s*$", re.IGNORECASE)
        recital_pattern = re.compile(r"^\((\d{1,3})\)\s+(.*)")

        in_annex = False

        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            raw_text = page.extract_text() or ""
            lines = raw_text.splitlines()

            i = 0
            while i < len(lines):
                line_str = lines[i].strip()
                if not line_str:
                    i += 1
                    continue

                # Check Annex
                annex_match = annex_pattern.match(line_str)
                if annex_match:
                    flush_section()
                    in_annex = True
                    annex_num = annex_match.group(1).upper()
                    annex_title = ""
                    if i + 1 < len(lines):
                        annex_title = lines[i+1].strip()
                        i += 1
                    current_h1 = f"ANNEX {annex_num}: {annex_title}"
                    current_h2 = f"ANNEX {annex_num}"
                    current_h3 = ""
                    current_art = f"Annex {annex_num}"
                    current_type = "table" if "list" in annex_title.lower() or "criteria" in annex_title.lower() else "text"
                    current_pages.append(page_num)
                    current_lines.append(f"ANNEX {annex_num}\n{annex_title}")
                    i += 1
                    continue

                # Check Chapter
                chap_match = chapter_pattern.match(line_str)
                if chap_match and not in_annex:
                    flush_section()
                    chap_num = chap_match.group(1).upper()
                    chap_title = ""
                    if i + 1 < len(lines):
                        chap_title = lines[i+1].strip()
                        i += 1
                    current_h1 = f"CHAPTER {chap_num}: {chap_title}"
                    current_h2 = f"CHAPTER {chap_num}"
                    current_h3 = ""
                    current_art = f"Chapter {chap_num}"
                    current_type = "text"
                    current_pages.append(page_num)
                    current_lines.append(f"CHAPTER {chap_num}\n{chap_title}")
                    i += 1
                    continue

                # Check Section
                sec_match = section_pattern.match(line_str)
                if sec_match and not in_annex:
                    flush_section()
                    sec_num = sec_match.group(1)
                    sec_title = ""
                    if i + 1 < len(lines):
                        sec_title = lines[i+1].strip()
                        i += 1
                    current_h2 = f"Section {sec_num}: {sec_title}"
                    current_h3 = ""
                    current_pages.append(page_num)
                    current_lines.append(f"Section {sec_num}\n{sec_title}")
                    i += 1
                    continue

                # Check Article
                art_match = article_pattern.match(line_str)
                if art_match:
                    flush_section()
                    art_num = art_match.group(1)
                    art_title = ""
                    if i + 1 < len(lines):
                        art_title = lines[i+1].strip()
                        i += 1
                    current_h3 = f"Article {art_num}: {art_title}"
                    current_art = f"Article {art_num}"
                    current_type = "text"
                    current_pages.append(page_num)
                    current_lines.append(f"Article {art_num}\n{art_title}")
                    i += 1
                    continue

                # Check Recital in Preamble (pages 1 to 44)
                if page_num <= 44:
                    rec_match = recital_pattern.match(line_str)
                    if rec_match:
                        rec_num = rec_match.group(1)
                        if int(rec_num) > 0 and int(rec_num) <= 180:
                            flush_section()
                            current_h1 = "Preamble & Recitals"
                            current_h2 = f"Recital ({rec_num})"
                            current_h3 = ""
                            current_art = f"Recital ({rec_num})"
                            current_type = "text"
                            current_pages.append(page_num)
                            current_lines.append(line_str)
                            i += 1
                            continue

                current_pages.append(page_num)
                current_lines.append(line_str)
                i += 1

        flush_section()
        return sections
