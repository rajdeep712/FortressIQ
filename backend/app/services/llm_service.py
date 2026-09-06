"""Groq-hosted LLM for grounded answer generation.

Wraps ``langchain_groq.ChatGroq`` (OpenAI-compatible) with a streaming
interface. The system prompt forces grounded, cited answers: the model
must only use the supplied context and emit inline ``[n]`` markers plus a
trailing JSON citation block that the graph parses into UI locators.
"""

from __future__ import annotations

import json
import logging
import re
from typing import AsyncIterator, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise research assistant grounded ONLY in the provided context excerpts.

Rules:
1. Answer using ONLY the context below. Do not use outside knowledge.
2. Cite every factual claim with an inline marker like [1], [2], ... matching the context numbering.
3. If the context is insufficient, say so clearly instead of guessing.
4. Prefer markdown: short headings and concise bullets.
5. At the very end of your answer, output a JSON block on its own, of the form:

START_CITATIONS
{"citations": [{"id":1,"doc_id":"...","page":1,"chunk_title":"...","snippet":"...","confidence":0.9,"score":0.85,"bounding_box":{"top":0,"left":0,"width":100,"height":100}}]}
END_CITATIONS

Keep the JSON valid and complete. Use the exact START_CITATIONS / END_CITATIONS delimiters.
"""


class LLMService:

    def __init__(self, chat: ChatGroq | None = None):
        self._chat = chat
        self._model = settings.groq_chat_model

    @property
    def model(self) -> str:
        return self._model

    def _client(self) -> ChatGroq:
        if self._chat is not None:
            return self._chat
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not configured. Set it in backend/.env "
                "to enable answer generation."
            )
        self._chat = ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name=self._model,
            max_tokens=settings.groq_max_tokens,
            temperature=settings.groq_temperature,
            timeout=settings.groq_request_timeout_seconds,
        )
        return self._chat

    async def stream(
        self,
        system: str,
        user_prompt: str,
    ) -> AsyncIterator[str]:
        """Yield text chunks as the LLM streams its answer."""
        client = self._client()
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=user_prompt),
        ]
        async for event in client.astream(messages):
            content = getattr(event, "content", None)
            if not content:
                continue
            # content may be a plain string or a list of content blocks.
            if isinstance(content, str):
                chunk = content
            elif isinstance(content, list):
                chunk = "".join(
                    str(block.get("text", ""))
                    for block in content
                    if isinstance(block, dict)
                )
            else:
                chunk = str(content)
            if chunk:
                yield chunk


def parse_citations(full_answer: str) -> tuple[str, list[dict]]:
    """Split a generated answer into (clean_text, citations_list).

    The JSON citation block is stripped from the returned text. If parsing
    fails we return the raw text with an empty citation list (the caller
    can fall back to document hits).
    """
    citations: list[dict] = []
    text = full_answer

    m = re.search(
        r"START_CITATIONS\s*(\{.*?\})\s*END_CITATIONS",
        full_answer,
        flags=re.DOTALL,
    )
    if m:
        raw = m.group(1)
        # Trim trailing commas inside the JSON (common LLM slip).
        raw = re.sub(r",\s*}", "}", raw)
        try:
            data = json.loads(raw)
            citations = data.get("citations", []) or []
        except json.JSONDecodeError:
            logger.warning("Could not parse citation JSON from LLM output")
        # Remove the whole START..END block from the visible text.
        text = (
            full_answer[: m.start()].rstrip()
            + "\n\n"
            + full_answer[m.end():].strip()
        ).strip()

    return text, citations
