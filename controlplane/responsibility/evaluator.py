"""
Responsibility evaluator.

Pure-logic gate first (no LLM on the safe path):
  * unsafe    if toxicity_max >= CP_TOX_HARD
              OR (toxicity_max >= CP_TOX_SOFT AND a high-relevance prohibiting
                  clause was retrieved)
  * uncertain if toxicity_max in the soft band, or a strong clause hit alone
  * safe      otherwise

Only when unsafe/uncertain do we spend ONE `responsibility` LLM call to produce
the structured, clause-cited violation report (reusing the Responsibility Agent's
system prompt). Everything cites real retrieved chunks and real model outputs -
nothing is fabricated.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List

from controlplane.config import settings
from controlplane.llm import complete

try:
    import sys
    from pathlib import Path

    _RESP_SRC = Path(__file__).resolve().parents[2] / "Responsiblity Agent"
    if str(_RESP_SRC) not in sys.path:
        sys.path.insert(0, str(_RESP_SRC))
    from src.agent.prompts import RESPONSIBILITY_AGENT_SYSTEM_PROMPT  # type: ignore
except Exception:  # pragma: no cover
    RESPONSIBILITY_AGENT_SYSTEM_PROMPT = (
        "You are an AI Responsibility & Content-Safety officer. Decide if the STATEMENT is "
        "harmful/discriminatory or violates the EU AI Act / NIST AI RMF, citing the retrieved "
        "evidence. Start with '**STATUS**: COMPLIANT' or '**STATUS**: UNETHICAL / NON-COMPLIANT'."
    )

# genuinely risky AI actions the ANSWER might advocate (toxicity models can't catch these)
_RISKY_ANSWER = re.compile(
    r"(subliminal|manipulat\w+|social scor\w+|social credit|mass surveillance|"
    r"facial recognition to (track|identify|penal)|emotion recognition|deceiv\w+ users|"
    r"exploit\w* (vulnerab|children)|without (their )?consent|"
    r"(deny|reject|disqualif\w+|penali[sz]e|filter out) \w*.{0,30}(based on|because of).{0,20}(race|gender|religion|ethnic|zip code|demographic|age|nationalit)|"
    r"predict\w* criminal\w*|automatically (reject|penali[sz]e|disqualif\w+|deduct) )",
    re.IGNORECASE,
)
_PROHIBIT_MARKERS = re.compile(
    r"prohibit|shall not|is not permitted|unlawful|subliminal|social scoring|"
    r"biometric categoris|manipulat|discriminat", re.IGNORECASE,
)

# Hate-speech / dehumanising terms that should be caught directly from the query.
# Matches slurs, explicit dehumanisation, and calls for exclusion based on identity.
_HATE_QUERY = re.compile(
    r"subhuman|untermensch|vermin|parasite|infestation|cockroach|savage|"
    r"inferior race|master race|racial inferior|ethnic cleansi|genocide|"
    r"slur|hate speech|dehumaniz|dehumani[sz]|white suprema|neo.?nazi|"
    r"n[i1][g9][g9][e3]r|f[a@]gg[o0]t|sp[i1]c|ch[i1]nk|k[i1]ke|w[e3]tb[a@]ck|"
    r"(should be|must be|need to be)\s+(eliminat|extermina|purged|expel|remov|banned|killed)\s*"
    r".{0,30}(race|ethnic|religion|group|minorit|jew|muslim|black|white|asian|gay|trans)",
    re.IGNORECASE,
)

# Hate-speech markers found in the *retrieved chunks* themselves (the main RAG results).
# If the toxicity_kb retrieves content like "Islam is inherently violent", those chunks
# are themselves evidence that the query surfaced prohibited hate-speech material.
_HATE_CHUNK = re.compile(
    r"inherently violent|subhuman|inferior|vermin|parasite|cockroach|savage|"
    r"should be killed|should be banned|ethnic cleansing|genocide|extermination|"
    r"hate speech|racial slur|dehumanising|dehumanizing|white supremacy|"
    r"(all|most|many)\s+(muslim|jew|black|asian|gay|trans|women|men)\s+(are|is|should)",
    re.IGNORECASE,
)


def _chunk_hate_signal(main_rrf_chunks: List[Dict[str, Any]]) -> tuple[float, List[str]]:
    """Check if the MAIN RAG retrieved chunks contain hate-speech / discriminatory content.

    When the toxicity_kb route retrieves hate-speech examples (e.g. Islamophobia,
    anti-semitism), those chunks are direct evidence that the query surfaced prohibited
    material under EU AI Act Article 5 and NIST AI RMF fairness principles.
    Returns (strength 0..1, default_rules).
    """
    if not main_rrf_chunks:
        return 0.0, []
    hate_chunks = [c for c in main_rrf_chunks[:5] if _HATE_CHUNK.search(c.get("text", ""))]
    # Also check metadata: target= fields indicate protected-group targeting
    targeted = [
        c for c in main_rrf_chunks[:5]
        if (c.get("metadata", {}) or {}).get("target") not in {None, "", "none", "not_hate"}
    ]
    hit_count = len(set([id(c) for c in hate_chunks + targeted]))
    if hit_count == 0:
        return 0.0, []
    strength = min(1.0, 0.4 + 0.15 * hit_count)
    return strength, ["EU AI Act – Article 5 (prohibited practices)",
                      "NIST AI RMF – GOVERN 1.1 (fairness)",
                      "UN Declaration of Human Rights – Article 1"]


def _clause_signal(rrf_chunks: List[Dict[str, Any]], answer: str, query: str = "") -> tuple[float, List[str]]:
    """How strong is the 'this answer may violate a retrieved rule' signal.

    Legal boilerplate is everywhere in the corpus, so a chunk merely containing
    'prohibit' is NOT enough - the ANSWER itself must describe a risky AI action
    AND a strongly-ranked prohibiting clause must have been retrieved.

    Exception: if the QUERY itself contains hate-speech / dehumanising terms,
    we immediately return strength=1.0 regardless of the answer text.
    """
    # Fast-path: hate-speech / slur in the query itself → always unsafe
    if query and _HATE_QUERY.search(query):
        rules = sorted({
            ((c.get("metadata", {}) or {}).get("law_or_article") or c.get("title"))
            for c in rrf_chunks[:5]
            if ((c.get("metadata", {}) or {}).get("law_or_article") or c.get("title"))
        })
        return 1.0, rules or ["EU AI Act – prohibited practices", "NIST AI RMF – fairness"]

    if not rrf_chunks:
        return 0.0, []
    risky = bool(_RISKY_ANSWER.search(answer or ""))
    top = rrf_chunks[: min(3, len(rrf_chunks))]
    prohibiting = [c for c in top if _PROHIBIT_MARKERS.search(c.get("text", ""))]
    top_score = rrf_chunks[0].get("score", 0.0)
    rules = sorted({
        ((c.get("metadata", {}) or {}).get("law_or_article") or c.get("title"))
        for c in prohibiting if ((c.get("metadata", {}) or {}).get("law_or_article") or c.get("title"))
    })
    if not risky:
        return 0.0, rules
    strength = min(1.0, 0.5 * bool(prohibiting) + 0.3 * (top_score > 0.03) + 0.2 * (len(prohibiting) >= 2))
    return strength, rules


def _extract_status(text: str) -> bool:
    block = text[:1500]
    m = re.search(r"\*\*STATUS\*\*\s*:?\s*(.+)", block, re.IGNORECASE)
    seg = (m.group(1) if m else block).upper()
    if "UNETHICAL" in seg or "NON-COMPLIANT" in seg or "FLAGGED" in seg or "VIOLAT" in seg:
        return True
    return False


def _llm_report(query: str, answer: str, rrf_chunks: List[Dict[str, Any]], toxicity: Dict[str, Any]) -> Dict[str, Any]:
    evidence = "\n\n".join(
        f"[EVIDENCE {i+1}] {c.get('title', '')}\n{c.get('text', '')[:900]}"
        for i, c in enumerate(rrf_chunks[:5])
    ) or "(no evidence retrieved)"
    tox_line = ", ".join(
        f"{k}={v.get('prob')}({v.get('label')})" for k, v in toxicity.items()
        if isinstance(v, dict) and "prob" in v
    )
    user = (
        f"### ORIGINAL USER QUERY (this is what was typed by the user):\n\"\"\"\n{query}\n\"\"\"\n\n"
        f"### AI-GENERATED ANSWER TO EVALUATE:\n\"\"\"\n{answer}\n\"\"\"\n\n"
        f"### TOXICITY MODEL OUTPUTS (scored on both query and answer):\n{tox_line}\n\n"
        f"### RETRIEVED COMPLIANCE EVIDENCE:\n{evidence}\n\n"
        "IMPORTANT: Evaluate whether the USER QUERY itself contains hate speech, "
        "dehumanising language, slurs, or prohibited content — not just the AI answer. "
        "If the query contains a slur or dehumanising term (e.g. 'subhuman'), that "
        "constitutes a hate-speech instance regardless of how the AI answered it. "
        "Flag it as UNETHICAL / NON-COMPLIANT and cite the relevant law. Follow the strict output format."
    )
    try:
        res = complete(
            "responsibility",
            [
                {"role": "system", "content": RESPONSIBILITY_AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=900,
        )
        return {"report": res.text, "is_violation": _extract_status(res.text), "llm_call": res.as_call_record()}
    except Exception as e:  # noqa: BLE001
        return {"report": f"(responsibility LLM unavailable: {e})", "is_violation": False, "llm_call": None}


def evaluate_responsibility(state: Dict[str, Any]) -> Dict[str, Any]:
    answer = state.get("answer", "") or ""
    query = state.get("guarded_query") or state.get("original_query", "")
    toxicity = state.get("toxicity", {}) or {}
    tox_max = float(state.get("toxicity_max") or toxicity.get("toxicity_max") or 0.0)
    rrf = state.get("resp_rrf_chunks", []) or []          # responsibility KB chunks (laws)
    main_rrf = state.get("rrf_chunks", []) or []           # main RAG chunks (may contain hate content)
    selected_kb = state.get("selected_kb") or ""

    # 1. Check hate-speech in the query itself / responsibility law chunks
    clause_strength, rules = _clause_signal(rrf, answer, query=query)

    # 2. Check if the MAIN retrieved chunks contain hate-speech material.
    #    This catches cases where the LLM refused to answer ("I can't help") but
    #    the retrieved documents are themselves hate-speech / discriminatory.
    chunk_strength, chunk_rules = _chunk_hate_signal(main_rrf)
    if chunk_strength > clause_strength:
        clause_strength = chunk_strength
        rules = rules or chunk_rules

    t0 = time.perf_counter()

    hard = tox_max >= settings.tox_hard_threshold
    soft = tox_max >= settings.tox_soft_threshold
    is_tox_route = (selected_kb == "toxicity_kb")

    if hard or (clause_strength >= 0.8):
        status = "unsafe"
    elif soft or clause_strength >= 0.5:
        status = "unsafe" if is_tox_route else "uncertain"
    else:
        status = "safe"

    out: Dict[str, Any] = {
        "resp_status": status,
        "resp_reasoning": (
            f"toxicity_max={tox_max:.2f} (hard={settings.tox_hard_threshold}, soft={settings.tox_soft_threshold}), "
            f"clause_signal={clause_strength:.2f}, chunk_signal={chunk_strength:.2f}, rules={rules[:5]}"
        ),
        "violated_rules": [],
        "resp_report": None,
        "evidence_chunks": rrf[:5],
    }

    if status in {"unsafe", "uncertain"}:
        # Combine responsibility law chunks + the main RAG chunks as evidence for the LLM report
        combined_evidence = rrf[:3] + [
            {"title": f"Retrieved content [{i+1}]", "text": c.get("text", ""),
             "metadata": c.get("metadata", {})}
            for i, c in enumerate(main_rrf[:3])
        ]
        report = _llm_report(query, answer, combined_evidence, toxicity)
        out["resp_report"] = report["report"]
        # clause_strength==1.0 means we KNOW it's hate-speech — never let the LLM flip it back.
        if clause_strength >= 1.0 or hard or report["is_violation"]:
            out["resp_status"] = "unsafe"
            out["violated_rules"] = rules or _rules_from_report(report["report"])
        else:
            out["resp_status"] = "safe"
        if report.get("llm_call"):
            out["_resp_llm_call"] = report["llm_call"]

    out["resp_eval_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


def _rules_from_report(text: str) -> List[str]:
    found = re.findall(r"(EU AI Act[^\n,;.]*|Article\s+\d+[a-z()0-9 ]*|NIST[^\n,;.]*|GDPR[^\n,;.]*|Recital\s*\(?\d+\)?)", text)
    return sorted({f.strip() for f in found})[:8]
