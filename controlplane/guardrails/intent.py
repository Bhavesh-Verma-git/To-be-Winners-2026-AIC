"""
Query-intent classifiers shared by the router, the performance evaluator and the
responsibility evaluator. Pure regex - no LLM, no latency.

  harmful_generation_request(q)  -> True if the query ASKS the assistant to
      produce / justify / argue hateful, discriminatory or dehumanising content
      about a protected group. These must be BLOCKED (finalize_harmful), never
      sent to the hallucination-retry or human-in-the-loop paths.

  content_safety_question(q)     -> True if the query is an *analytical* question
      about whether some phrase is toxic / hate speech / a slur. These are
      answered (the content-safety KB), not blocked.
"""

from __future__ import annotations

import re

_PROTECTED = (
    r"(wom[ae]n|m[ae]n|girls?|boys?|muslims?|jews?|jewish|christians?|hindus?|buddhists?|sikhs?|"
    r"black|white|asian|african|arab|latino|latina|hispanic|mexican|indian|chinese|"
    r"immigrants?|migrants?|refugees?|foreigners?|gay|lesbian|bisexual|trans(gender)?|queer|lgbt\w*|"
    r"disabled|elderly|old people|young people|poor people|homeless|"
    r"a (certain|particular) (group|religion|race|ethnicity|nationality|community)|"
    r"one (ethnic|racial|religious) group|another (ethnic|racial|religious) group|"
    r"people from [a-z ]+|[a-z]+ people|[a-z]+ nationals?|[a-z]+ community)"
)
_NEGATIVE = (
    r"(inferior|superior|subhuman|less (evolved|human|intelligent|capable|worthy)|"
    r"a burden|burden on|drain on|parasites?|vermin|criminals?|terrorists?|violent|dangerous|"
    r"stupid|lazy|dirty|evil|greedy|untrustworthy|dishonest|incapable|worthless|"
    r"not (as )?(good|smart|capable|intelligent|welcome)|don'?t belong|should (not|n't) (be here|exist)|"
    r"deserve (less|worse|to be)|shouldn'?t (have|be allowed)|are the problem|ruin\w*)"
)
_ASK_VERB = (
    r"(write|compose|draft|give me|provide|list|generate|produce|create|make|"
    r"argue|arguing|make the case|justify|defend|prove|explain why|tell me why|"
    r"reasons? (why|that)|arguments? (that|for|why)|a (paragraph|essay|speech|rant|post))"
)

_SLUR = re.compile(
    r"\b(n[i1]gg[e3]rs?|f[a@]gg?[o0]ts?|k[i1]kes?|ch[i1]nks?|sp[i1]cs?|wetbacks?|"
    r"tr[a@]nn?(y|ies)|retards?|paki[s]?)\b",
    re.IGNORECASE,
)

_HARMFUL_REQUEST = re.compile(
    _ASK_VERB + r"\b[^.?!]{0,70}\b" + _PROTECTED + r"\b[^.?!]{0,50}\b" + _NEGATIVE,
    re.IGNORECASE,
)
_HARMFUL_REQUEST_REV = re.compile(  # "... why <group> are inferior" without a lead verb
    r"\b(why|how come)\b[^.?!]{0,25}\b" + _PROTECTED + r"\b[^.?!]{0,40}\b" + _NEGATIVE,
    re.IGNORECASE,
)
# A request to PRODUCE humour / a rhyme / a "roast" targeting a demographic group -
# jokes about a protected group trade on stereotypes. (Not hardcoded to any group;
# any word matching _PROTECTED counts.)
_HARMFUL_HUMOUR = re.compile(
    r"\b(joke|jokes|meme|memes|limerick|rhyme|poem|roast|skit|gag|one-?liner|pun)\b"
    r"[^.?!]{0,45}\b" + _PROTECTED
    + r"|\b" + _PROTECTED + r"\b[^.?!]{0,25}\b(joke|jokes|meme|memes|limerick|rhyme|roast)\b",
    re.IGNORECASE,
)

# An *analytical* content-safety question is about ONE specific statement / phrase
# being classified. It must name a concrete referent (a quote, "this statement",
# "the phrase", "calling X a Y", "the word/term", ...) - NOT ask what toxic views /
# stereotypes about a group ARE (that is a request to surface toxic content).
_REFERENT = (
    r"(this|that|the following|the above)\s*"
    r"(statement|phrase|sentence|comment|remark|line|text|joke|word|term|expression|question)"
    r"|\bcalling\s+(someone|somebody|him|her|them|people|a group|a person|an?)\b"
    r"|\bthe (word|term|phrase|expression|statement)\b"
    r"|\bsaying (that|\")|to say (that|\")"
    r"|\"[^\"]{2,}\"|'[^']{2,}'"          # an explicitly quoted phrase
    r"|\bwhether (this|that|the|it|saying)\b"
)
_SAFETY_Q = re.compile(
    r"(" + _REFERENT + r")[^.?!]{0,80}\b(hate speech|a slur|slurs?|offensive|toxic|"
    r"racist|sexist|bigoted|discriminatory|dehumani[sz]\w+|acceptable|okay to say|"
    r"inappropriate|harmful|a stereotype)\b"
    r"|\b(hate speech|a slur|slurs?|offensive|toxic|racist|sexist|bigoted|discriminatory|"
    r"dehumani[sz]\w+|harmful|a stereotype)\b[^.?!]{0,60}(" + _REFERENT + r")"
    r"|\bwhat (target group|framing|category)\b[^.?!]{0,40}(" + _REFERENT + r"|fall under|belong)"
    r"|\bclassif\w+ (this|the following)\b"
    r"|\banaly[sz]e\b[^.?!]{0,30}\b(statement|phrase|comment|text|joke)\b",
    re.IGNORECASE,
)


_EDU_FRAME = (
    r"(recogni[sz]e|identif\w+|detect\w*|spot|notice|warning signs?|signs? of|red flags?|"
    r"counter|combat|respond to|report\w*|prevent\w*|reduce|mitigat\w+|"
    r"protect\w* (against|from|yourself)|guard against|defend against|stand up to|"
    r"moderat\w+|content moderation|handle|address\w*|de-?escalat\w+|"
    r"raise awareness|awareness (of|about|campaign)|educat\w+|train\w*|training|course|class|"
    r"workshop|lesson|seminar|teach\w*|curriculum|syllabus|study|studying|research\w*|academ\w+|"
    r"policy (on|for|about)|guidelines? (for|on|to|about))"
)
_HARM_NOUN = (
    r"(hate speech|hateful (content|language|speech)|harass\w+|bully\w*|bullying|abuse|abusive|"
    r"discriminat\w+|toxic\w*|toxicity|extremis\w+|radicali[sz]\w+|misinformation|"
    r"disinformation|online harm|trolling|cyberbullying|prejudice|bigotry)"
)
_EDU_DEFENSIVE = re.compile(
    _EDU_FRAME + r"\b[^.?!]{0,70}\b" + _HARM_NOUN
    + r"|" + _HARM_NOUN + r"\b[^.?!]{0,70}\b" + _EDU_FRAME,
    re.IGNORECASE,
)


def harmful_generation_request(query: str) -> bool:
    q = query or ""
    if content_safety_question(q) or defensive_or_educational(q):
        return False
    return bool(_HARMFUL_REQUEST.search(q) or _HARMFUL_REQUEST_REV.search(q)
                or _HARMFUL_HUMOUR.search(q) or _SLUR.search(q))


def content_safety_question(query: str) -> bool:
    return bool(_SAFETY_Q.search(query or ""))


def defensive_or_educational(query: str) -> bool:
    """Query is about *countering / studying / moderating* harmful content, not
    producing it - must not be blocked as harmful."""
    return bool(_EDU_DEFENSIVE.search(query or ""))
