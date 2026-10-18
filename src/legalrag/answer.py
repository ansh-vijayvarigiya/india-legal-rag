"""Answer generation over retrieved passages.

The model is handed numbered passages and told to answer from those alone. That
instruction, not the model's judgement, is what keeps an answer inside the
corpus. Asked "what is the bank rate?" with no context, a model will describe a
commercial bank's lending rate, which is not the RBI's Bank Rate — a confident
answer to a different question.

`answer` is separated from the chat client so the prompt, the refusal path and
the citation parsing can be tested without a network call.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx

from legalrag.models import Retrieved

INSTRUCTIONS = """You answer questions about Indian regulatory circulars.

Rules:
- Use only the numbered passages below. For this answer they are the whole of
  your knowledge.
- Do not use anything you know from training, and do not fill gaps with
  plausible detail.
- If the passages do not contain the answer, reply exactly "I could not find
  this in the circulars provided" and stop. Do not guess.
- Cite the passage numbers you relied on, as [1] or [2], next to the sentence
  they support.
- Write a short explanation, not a one-line answer. Give the direct answer
  first, then the specifics behind it: figures, rates, dates, thresholds and
  conditions. Use a short bullet list when there are several.
- Expand every acronym the first time you use it, in brackets, because the
  reader may not work in this field.
- Keep it under about 120 words. Say what the passages state, and say so plainly
  when they cover the question only partly.
- You are not a lawyer. Give no legal advice and no recommendation about what
  the reader should do.
"""

NOTHING_RETRIEVED = (
    "I could not find this in the circulars provided. Nothing in the corpus "
    "matched the question closely enough to answer from."
)

_MARKER = re.compile(r"\[(\d+)[^\]]*\]")


def _normalise_markers(text: str) -> str:
    """Make citation markers uniform: `【1】` and `[1†L2-L4]` both become `[1]`.

    Models vary in how they mark a citation, and the variation is invisible in
    the reply text but breaks the mapping from marker to chunk.
    """
    return _MARKER.sub(r"[\1]", text.replace("【", "[").replace("】", "]"))

# Free tiers meter tokens per minute, and every retrieved passage is sent whole.
# Five is set by the corpus rather than by taste: multi-clause questions need a
# second chunk and those sit as deep as rank 4, so a shorter prompt refuses
# questions it could have answered. Three would halve the cost and cost answers.
MAX_PASSAGES = 5
# Room for an explanation of about a hundred words, a short bullet list, and the
# model's own thinking. The finish_reason check catches it if this is too small.
# This is the default for every client. `legalrag.judge` raises it, because a
# judge's hidden reasoning runs longer than the JSON verdict it finally writes.
MAX_REPLY_TOKENS = 700


class LLMError(RuntimeError):
    """The model could not be reached, or replied in a shape we cannot use."""


def is_rate_limit(error: Exception) -> bool:
    """Free plans meter tokens per minute and say so in their own words."""
    text = str(error).lower()
    return "429" in text or "rate limit" in text


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    used_chunk_ids: tuple[str, ...]


class ChatClient(Protocol):
    def complete(self, messages: list[dict]) -> str: ...


def render_context(hits: Sequence[Retrieved]) -> str:
    """Numbered passages. Numbers are positions, so a marker maps to a chunk."""
    return "\n\n".join(
        f"[{position}] {hit.chunk.chunk_id}\n{hit.chunk.text}"
        for position, hit in enumerate(hits, start=1)
    )


def build_messages(question: str, hits: Sequence[Retrieved]) -> list[dict]:
    user = f"Passages:\n\n{render_context(hits)}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": user},
    ]


def _cited_chunk_ids(text: str, hits: Sequence[Retrieved]) -> tuple[str, ...]:
    """Chunk ids for the passage numbers the reply cited, ignoring nonsense."""
    positions = {
        int(match.group(1)) for match in _MARKER.finditer(text)
    }
    return tuple(
        hits[position - 1].chunk.chunk_id
        for position in sorted(positions)
        if 1 <= position <= len(hits)
    )


def answer(
    question: str, hits: Sequence[Retrieved], client: ChatClient
) -> Answer:
    """Answer `question` from the top `MAX_PASSAGES` of `hits`.

    Retrieving nothing costs no model call.
    """
    selected = list(hits)[:MAX_PASSAGES]
    if not selected:
        return Answer(NOTHING_RETRIEVED, ())
    text = _normalise_markers(client.complete(build_messages(question, selected)).strip())
    return Answer(text, _cited_chunk_ids(text, selected))


def _error_text(response: httpx.Response) -> str:
    """The provider's message. Never the request, which carries the key."""
    try:
        return response.json()["error"]["message"]
    except Exception:
        return response.text[:200]


def _reply_text(response: httpx.Response) -> str:
    try:
        choice = response.json()["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, ValueError) as error:
        raise LLMError(f"unexpected reply shape: {error}") from error
    if choice.get("finish_reason") == "length":
        # A truncated sentence reads exactly like a finished one on the page.
        raise LLMError(
            "the model hit the reply limit and was cut off; "
            "raise the token limit or send fewer passages"
        )
    if not content or not content.strip():
        raise LLMError("the model returned an empty reply")
    return content


class OpenAIChatClient:
    """Minimal client for the OpenAI chat-completions shape.

    Groq, Google AI Studio and others expose this same interface, so the
    provider is configuration rather than code.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        http_client: httpx.Client | None = None,
        timeout: float = 60.0,
        max_tokens: int = MAX_REPLY_TOKENS,
        reasoning_effort: str = "low",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self._client = http_client or httpx.Client(timeout=timeout)

    def complete(self, messages: list[dict]) -> str:
        response = self._client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": self.max_tokens,
                # gpt-oss spends reply tokens on hidden reasoning. Left at the
                # default, the thinking eats the answer's budget and the reply
                # arrives cut off. Qwen accepts `none` here and the judge uses
                # it: on the free tier hidden reasoning counts against an output
                # token meter of a thousand a minute.
                "reasoning_effort": self.reasoning_effort,
            },
        )
        if response.status_code != 200:
            raise LLMError(
                f"the model returned {response.status_code}: {_error_text(response)}"
            )
        return _reply_text(response)
