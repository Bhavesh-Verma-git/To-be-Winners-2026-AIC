"""
"Mulling" ticker for Tab 1 - a stream of short status phrases that reveals
word-by-word while the pipeline runs, the way Claude shows it is working.

Zero added latency: `MullFeed.pump()` is called on the events the runner is
already emitting (node changes, tokens). It just reveals a few more words of a
pre-written script each call - it never sleeps or blocks.
"""

from __future__ import annotations

import random
from typing import List

from controlplane.state import Stage

# phrases keyed by the stage that is starting; ~150 words total across a run
_SCRIPT = {
    Stage.START: [
        "Spinning up the governance pipeline",
        "Receiving your query and starting the clock",
    ],
    Stage.GUARDRAILS: [
        "Screening the input for prompt-injection and jailbreak patterns",
        "Scanning for PII and masking anything sensitive",
        "Input looks clean, moving on",
    ],
    Stage.CACHE: [
        "Embedding the query and checking the semantic cache",
        "Comparing against previously answered questions",
        "No close enough match, running the full pipeline",
    ],
    Stage.ROUTER: [
        "Handing the query to the routing agent",
        "Deciding which knowledge base should answer this",
        "Weighing customer-support, HR, Azure, decision-support and content-safety",
        "Route locked in",
    ],
    Stage.RETRIEVAL: [
        "Running vector search over the index",
        "Running BM25 keyword search in parallel",
        "Fusing both rankings with reciprocal rank fusion",
        "Selecting the top five grounding chunks",
    ],
    Stage.ANSWER: [
        "Grounding the answer strictly in the retrieved context",
        "Streaming the response token by token",
    ],
    Stage.PERFORMANCE: [
        "Fanning out the performance and responsibility branches together",
        "RAGAS scoring faithfulness, relevancy and coverage",
        "XGBoost estimating hallucination probability from twelve features",
        "Checking for entity drift against the source chunks",
    ],
    Stage.RESPONSIBILITY: [
        "Retrieving the EU AI Act, NIST and hate-speech clauses that apply",
        "Detoxify, toxic-bert and s-nlp RoBERTa scoring the query and the draft",
        "Cross-checking the answer against compliance evidence",
    ],
    Stage.AGGREGATE: [
        "Both branches are back, joining the results",
        "Applying the safety-biased decision rules",
        "Deciding: deliver, self-reflect and retry, ask a human, or block",
    ],
    Stage.HALLUCINATION_RETRY: [
        "Answer was not well grounded, self-reflecting",
        "The agent is rewriting the query for a better retrieval",
        "Running the pipeline once more with the improved query",
    ],
    Stage.HITL: [
        "The context is not enough to answer this safely",
        "Pausing to ask you for the missing detail",
    ],
    Stage.FINALIZE: [
        "Formatting the final response and verdict",
        "Writing the trace to LangSmith",
    ],
}

_FILLER = [
    "still working", "almost there", "tightening the grounding",
    "keeping every step under the ten-second budget",
]


class MullFeed:
    """Accumulates a word queue from stage phrases and reveals it gradually."""

    def __init__(self, words_per_pump: int = 3) -> None:
        self._queue: List[str] = []
        self._shown: List[str] = []
        self._seen_stages: set[str] = set()
        self._wpp = words_per_pump
        self._since_new = 0

    def add_stage(self, stage: str) -> None:
        if stage in self._seen_stages:
            return
        self._seen_stages.add(stage)
        for phrase in _SCRIPT.get(stage, []):
            self._queue.extend(phrase.split())
            self._queue.append("·")
        self._since_new = 0

    def pump(self) -> str:
        """Reveal a few more words. Returns the text to display."""
        self._since_new += 1
        if not self._queue and self._since_new % 6 == 0:
            self._queue.extend((random.choice(_FILLER) + " ·").split())
        for _ in range(self._wpp):
            if not self._queue:
                break
            self._shown.append(self._queue.pop(0))
        # keep only the tail so it doesn't grow without bound
        if len(self._shown) > 80:
            self._shown = self._shown[-80:]
        return " ".join(self._shown)

    def drain(self) -> str:
        self._shown.extend(self._queue)
        self._queue.clear()
        return " ".join(self._shown)


def mull_html(text: str, done: bool = False) -> str:
    if not text:
        return ""
    dot = "" if done else '<span class="cp-mull-dot">●</span> '
    return (
        f'<div class="cp-mull">{dot}{text}</div>'
    )
