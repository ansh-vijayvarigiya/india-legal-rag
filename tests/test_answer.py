"""Answer generation.

Nothing here touches the network: the chat client is either faked outright or
given an `httpx.MockTransport`.
"""

import json

import httpx
import pytest

from legalrag.answer import (
    LLMError,
    MAX_PASSAGES,
    MAX_REPLY_TOKENS,
    OpenAIChatClient,
    answer,
    build_messages,
    is_rate_limit,
    render_context,
)
from legalrag.models import Chunk, Retrieved


def _hit(chunk_id: str, text: str, rank: int = 1) -> Retrieved:
    return Retrieved(
        chunk=Chunk(chunk_id, "rbi:1", rank - 1, text, text, {}),
        score=1.0,
        rank=rank,
    )


class FakeClient:
    def __init__(self, reply: str = "An answer [1].") -> None:
        self.reply = reply
        self.calls: list[list[dict]] = []

    def complete(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return self.reply


def test_context_numbers_each_passage_and_shows_its_chunk_id():
    context = render_context(
        [_hit("rbi:1#0000", "alpha"), _hit("rbi:1#0001", "beta", rank=2)]
    )
    assert "[1] rbi:1#0000" in context
    assert "[2] rbi:1#0001" in context
    assert "alpha" in context
    assert "beta" in context


def test_instructions_forbid_outside_knowledge_and_require_refusal():
    """The model must not answer from its own training. Pinned so it cannot be softened."""
    messages = build_messages("who won the 2019 election?", [_hit("rbi:1#0000", "text")])
    system = messages[0]["content"].lower()
    assert "only" in system
    assert "do not use" in system
    assert "could not find" in system
    assert "legal advice" in system
    assert "do not guess" in system


def test_the_prompt_asks_for_an_explanation_rather_than_a_one_liner():
    """A one-line verdict teaches a reader who does not know the domain nothing."""
    system = build_messages("q", [_hit("rbi:1#0000", "t")])[0]["content"].lower()
    assert "not a one-line answer" in system
    assert "expand every acronym" in system
    assert "words" in system


def test_the_question_and_the_passages_are_both_sent():
    messages = build_messages("what is the rate?", [_hit("rbi:1#0000", "the rate is 4%")])
    assert messages[0]["role"] == "system"
    user = messages[1]["content"]
    assert "what is the rate?" in user
    assert "the rate is 4%" in user


def test_no_passages_refuses_without_calling_the_model():
    client = FakeClient()
    result = answer("what is the GST rate?", [], client)
    assert client.calls == []
    assert "could not find" in result.text.lower()
    assert result.used_chunk_ids == ()


def test_answer_returns_the_text_and_the_chunk_ids_the_model_cited():
    client = FakeClient("The rate is 4% [2].")
    result = answer(
        "rate?", [_hit("rbi:1#0000", "a"), _hit("rbi:1#0001", "b", rank=2)], client
    )
    assert result.text == "The rate is 4% [2]."
    assert result.used_chunk_ids == ("rbi:1#0001",)


def test_a_full_width_citation_is_recognised_and_tidied():
    """gpt-oss writes 【1】. The writer of the brackets is not the writer of this regex."""
    client = FakeClient("The rate is 4% 【1】.")
    result = answer("rate?", [_hit("rbi:1#0000", "a")], client)
    assert result.used_chunk_ids == ("rbi:1#0000",)
    assert result.text == "The rate is 4% [1]."


def test_a_citation_with_a_location_suffix_is_recognised_and_tidied():
    client = FakeClient("Only for hedging [1†L2-L4].")
    result = answer("q", [_hit("rbi:1#0000", "a")], client)
    assert result.used_chunk_ids == ("rbi:1#0000",)
    assert result.text == "Only for hedging [1]."


def test_a_cited_number_that_does_not_exist_is_ignored():
    client = FakeClient("The rate is 4% [7].")
    result = answer("rate?", [_hit("rbi:1#0000", "a")], client)
    assert result.used_chunk_ids == ()


def test_a_repeated_citation_is_recorded_once():
    client = FakeClient("It is 4% [1], and again [1].")
    result = answer("rate?", [_hit("rbi:1#0000", "a")], client)
    assert result.used_chunk_ids == ("rbi:1#0000",)


def _client(reply: httpx.Response) -> OpenAIChatClient:
    return OpenAIChatClient(
        "https://llm.test/v1",
        "secret-key",
        "test-model",
        http_client=httpx.Client(transport=httpx.MockTransport(lambda request: reply)),
    )


def test_rate_limits_are_told_apart_from_other_failures():
    assert is_rate_limit(LLMError("the model returned 429: Rate limit reached"))
    assert not is_rate_limit(LLMError("the model returned 401: invalid api key"))


def test_only_the_top_passages_are_sent():
    """Passage count is a token budget: the free tier allows ~8000 tokens a minute."""
    client = FakeClient()
    hits = [
        _hit(f"rbi:1#{i:04d}", f"passage {i}", rank=i + 1)
        for i in range(MAX_PASSAGES + 1)
    ]
    answer("q", hits, client)
    sent = client.calls[0][1]["content"]
    assert f"passage {MAX_PASSAGES - 1}" in sent
    assert f"passage {MAX_PASSAGES}" not in sent


def test_enough_passages_are_sent_to_reach_a_second_chunk():
    """q014's second labelled chunk sits at rank 4, so a shorter prompt answers
    that question with a refusal it did not have to give."""
    assert MAX_PASSAGES >= 5


def test_a_caller_can_turn_reasoning_off():
    """Qwen 3.8 accepts `none`, which emits no reasoning tokens at all. The
    answerer cannot use it, because gpt-oss does not accept `none`."""
    assert _sent_body(reasoning_effort="none")["reasoning_effort"] == "none"


def test_the_request_asks_for_light_reasoning():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
        )

    client = OpenAIChatClient(
        "https://llm.test/v1",
        "k",
        "m",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    client.complete([{"role": "user", "content": "hi"}])
    assert seen["reasoning_effort"] == "low"


def _sent_body(**kwargs) -> dict:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
        )

    client = OpenAIChatClient(
        "https://llm.test/v1",
        "k",
        "m",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kwargs,
    )
    client.complete([{"role": "user", "content": "hi"}])
    return seen


def test_the_token_budget_defaults_to_the_answer_budget():
    assert _sent_body()["max_tokens"] == MAX_REPLY_TOKENS


def test_a_caller_can_raise_the_token_budget():
    """The judge's hidden reasoning eats the budget before its JSON arrives, so
    it needs more room than an answer does."""
    assert _sent_body(max_tokens=1500)["max_tokens"] == 1500


def test_a_reply_cut_off_at_the_token_limit_is_an_error_not_an_answer():
    """A half sentence reads as a real answer. It must not reach the page."""
    client = _client(
        httpx.Response(
            200,
            json={"choices": [{"message": {"content": "The rate is 4"}, "finish_reason": "length"}]},
        )
    )
    with pytest.raises(LLMError, match="cut off"):
        client.complete([{"role": "user", "content": "hi"}])


def test_chat_client_posts_the_model_and_returns_the_reply_text():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})

    client = OpenAIChatClient(
        "https://llm.test/v1",
        "secret-key",
        "test-model",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert client.complete([{"role": "user", "content": "hi"}]) == "hello"
    assert seen["url"] == "https://llm.test/v1/chat/completions"
    assert seen["auth"] == "Bearer secret-key"
    assert seen["body"]["model"] == "test-model"
    assert seen["body"]["messages"] == [{"role": "user", "content": "hi"}]


def test_chat_client_reports_an_error_status_without_leaking_the_key():
    client = _client(
        httpx.Response(401, json={"error": {"message": "invalid api key"}})
    )
    with pytest.raises(LLMError) as raised:
        client.complete([{"role": "user", "content": "hi"}])
    assert "401" in str(raised.value)
    assert "invalid api key" in str(raised.value)
    assert "secret-key" not in str(raised.value)


def test_chat_client_rejects_an_empty_reply():
    client = _client(httpx.Response(200, json={"choices": [{"message": {"content": ""}}]}))
    with pytest.raises(LLMError, match="empty"):
        client.complete([{"role": "user", "content": "hi"}])
