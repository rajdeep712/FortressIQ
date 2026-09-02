"""Step 1: Query preprocessing.

Turns a raw user message into a query suitable for conversation search,
document retrieval (later), reranking and generation -- WITHOUT invoking an
LLM. We only normalize formatting and produce lightweight signals that the
router (step 3) consumes:

  * normalized_query            -- cleaned, ready-to-embed text
  * has_conversation_reference  -- anaphora / "we discussed..." style signal
  * explicit_document_reference -- "according to the pdf / the doc says..."

These signals never perform retrieval themselves; they only steer the router.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict


class QueryContext(BaseModel):
    original_query: str
    normalized_query: str


class QueryAnalysis(QueryContext):
    """Full analysis of a raw user message (preprocessing output)."""

    original_query: str = ""
    normalized_query: str = ""
    has_conversation_reference: bool = False
    explicit_document_reference: bool = False

    model_config = ConfigDict(extra="forbid")


# ---- Normalization helpers -------------------------------------------------

# All unicode whitespace (including NBSP / narrow spaces) -> single ASCII space.
_WS_RUN = re.compile(r"[\s\u00a0\u1680\u2000-\u200b\u202f\u205f\u3000]+")

# Whitespace before a sentence/clause punctuation mark.
_WS_BEFORE_PUNCT = re.compile(r"\s+([.,!?;:])")

# Repeated question / exclamation / period marks -> single occurrence.
_MULTI_QUESTION = re.compile(r"\?{2,}")
_MULTI_EXCLAM = re.compile(r"!{2,}")
_MULTI_PERIOD = re.compile(r"\.{2,}")

# Accidental duplicated function words ("the the", "and and", "of of").
_DUP_WORD = re.compile(
    r"\b(and|or|to|of|in|on|for|the|a|an|with|by|as|at)\s+\1\b",
    flags=re.IGNORECASE,
)

# Trailing punctuation that is safe to strip after all cleanup.
_TRAILING_JUNK = " \t\x00"

_SENSE_NEUTRAL_PUNCT_RE = re.compile(r"[\u200b\u200c\u200d]")


def _normalize(raw: str | None) -> str:
    """Return a cleaned, re-joinable version of a raw query.

    Preserves casing and terminal intent (keeps a trailing '?'), collapses
    whitespace, normalizes formatting and obvious typos-like repetition.
    Empty / None / whitespace-only input yields an empty string.
    """
    if raw is None:
        return ""

    s = _SENSE_NEUTRAL_PUNCT_RE.sub("", raw)
    s = s.strip()
    if not s:
        return ""

    s = _WS_RUN.sub(" ", s)
    s = _WS_BEFORE_PUNCT.sub(r"\1", s)
    # Re-insert a single space after a clause punct if it ran up against text.
    s = re.sub(r"([.,!?;:])(?=[^\s.,!?;:])", r"\1 ", s)
    s = _MULTI_QUESTION.sub("?", s)
    s = _MULTI_EXCLAM.sub("!", s)
    s = _MULTI_PERIOD.sub(".", s)
    s = _DUP_WORD.sub(r"\1", s)

    return s.strip(_TRAILING_JUNK)


# ---- Conversational-reference detection ------------------------------------

# Short anaphora pronouns. These are intentionally broad: the flag is a
# *signal* for the router, not a decision, so slightly high recall is fine.
_ANAPHORA_WORDS = (
    "that",
    "this",
    "it",
    "these",
    "those",
    "them",
    "there",
    "then",
)

_ANAPHORA_WORD_RE = re.compile(
    r"(?<![A-Za-z0-9_])(" + "|".join(
        re.escape(w) for w in _ANAPHORA_WORDS
    ) + r")(?![A-Za-z0-9_])",
    flags=re.IGNORECASE,
)

# Multi-word / idiomatic references to prior context.
_CONVERSATION_PHRASES = [
    r"the\s+above",
    r"the\s+following",
    r"aforementioned",
    r"as\s+(we|i|you)\s+(discussed|said|mentioned|talked|covered)",
    r"as\s+discussed",
    r"as\s+previously\s+(said|mentioned|stated)",
    r"(we|you|i)\s+(discussed|talked|said|mentioned|covered)\b",
    r"you\s+(said|mentioned|recommended|suggested|told)\b",
    r"what\s+did\s+(you|we|i)\s+recommend",
    r"our\s+(previous|earlier)\s+(decision|discussion|conversation)",
    r"earlier\s+discussion",
    r"continue",
    r"explain\s+that",
    r"go\s+(on|ahead)",
    r"elaborate",
    r"expand",
    r"in\s+more\s+detail",
    r"why\s+did\s+(we|you|i)\b",
    r"what\s+was\s+our\s+reason",
    r"the\s+decision",
]

_CONVERSATION_PHRASE_RE = re.compile(
    "|".join(f"(?:{p})" for p in _CONVERSATION_PHRASES),
    flags=re.IGNORECASE,
)


def _detect_conversation_reference(normalized: str) -> bool:
    if not normalized:
        return False
    if _ANAPHORA_WORD_RE.search(normalized):
        return True
    return bool(_CONVERSATION_PHRASE_RE.search(normalized))


# ---- Explicit document-reference detection ----------------------------------

# Document-ish nouns that, when appearing in a query, usually mean the user
# wants an answer grounded in an uploaded source rather than in chat history.
_DOC_NOUNS = (
    "document",
    "documents",
    "docs",
    "pdf",
    "pdfs",
    "file",
    "files",
    "paper",
    "papers",
    "report",
    "reports",
    "memo",
    "memos",
    "whitepaper",
    "chapter",
    "section",
    "appendix",
    "attachment",
    "upload",
    "uploaded",
    "docs",
)

_DOC_NOUN_RE = re.compile(
    r"(?<![A-Za-z0-9_])(" + "|".join(
        re.escape(n) for n in _DOC_NOUNS
    ) + r")(?![A-Za-z0-9_])",
    flags=re.IGNORECASE,
)

# Phrases that bind the query to the documents.
_DOC_PHRASES = [
    r"according\s+to\b",
    r"as\s+per\b",
    r"in\s+the\s+(?:uploaded\s+)?(?:attached\s+)?\b",
    r"from\s+the\s+(?:uploaded\s+)?\b",
    r"of\s+the\s+(?:uploaded\s+)?\b",
    r"referring\s+to\b",
    r"refer(?:s|ring)?\s+to\s+the\b",
    r"(?:what|where|how|who|why|which)\s+does\s+(?:the|this|that|our)\b",
    r"does\s+(?:the|this|that|our)\b.*\b(say|state|mention|recommend|suggest|covers|contain|describe)\b",
    r"on\s+page\s+\d+",
    r"pages?\s+\d+",
    r"in\s+section\b",
]

_DOC_PHRASE_RE = re.compile(
    "|".join(f"(?:{p})" for p in _DOC_PHRASES),
    flags=re.IGNORECASE,
)

# A doc noun directly followed by a reporting verb: "the document says...".
_DOC_VERB_RE = re.compile(
    r"\b(document|pdf|file|paper|report|memo|chapter|section|appendix)\b"
    r"\s+(says?|states?|mentions?|recommends?|suggests?|covers?|contains?|describes?|shows?|notes?)\b",
    flags=re.IGNORECASE,
)


def _detect_document_reference(normalized: str) -> bool:
    if not normalized:
        return False
    # A doc noun + a binding phrase, OR a doc noun + reporting verb.
    if _DOC_NOUN_RE.search(normalized) and _DOC_PHRASE_RE.search(normalized):
        return True
    if _DOC_VERB_RE.search(normalized):
        return True
    return False


# ---- Public API -------------------------------------------------------------


def analyze_query(raw_query: str | None) -> QueryAnalysis:
    """Analyze a raw user message (step 1).

    Robust to None / empty / whitespace-only input: such a query yields an
    empty normalized query and both flags False (the router must then fall
    back to document retrieval or reject).
    """
    original = raw_query or ""
    normalized = _normalize(raw_query)

    return QueryAnalysis(
        original_query=original,
        normalized_query=normalized,
        has_conversation_reference=_detect_conversation_reference(normalized),
        explicit_document_reference=_detect_document_reference(normalized),
    )
