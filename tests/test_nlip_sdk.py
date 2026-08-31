import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nlip_examples.nlip_integration_examples._config import CONFIG
from nlip_sdk import NLIPClient, NLIPMessage, NLIPServer, NLIPSession, load_config
from nlip_sdk.nlip_client import client as client_module


def test_config_loader_rejects_empty_endpoints(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('[entities.claude]\nendpoint = ""\n')

    with pytest.raises(ValueError, match="non-empty URL string"):
        load_config(config_path)


def test_message_server_and_config() -> None:
    with pytest.raises(ValidationError):
        NLIPMessage(
            format="text",
            subformat="English",
            content="invalid empty submessages",
            submessages=[],
        )

    request = NLIPMessage.model_validate(
        {
            "MessageType": "CONTROL",
            "Format": "TEXT",
            "Subformat": "English",
            "Content": "hello",
            "Submessages": [
                {
                    "Format": "TOKEN",
                    "Subformat": "conversation_client",
                    "Content": "conversation-1",
                }
            ],
        }
    )

    class EchoServer(NLIPServer):
        started = False
        stopped = False
        session_ids: list[str] = []
        history_sizes: list[int] = []
        ended_ids: list[str] = []

        def __init__(self) -> None:
            super().__init__("echo", max_sessions=1)

        async def startup(self) -> None:
            self.started = True

        async def shutdown(self) -> None:
            self.stopped = True

        async def session_ended(self, session: NLIPSession) -> None:
            self.ended_ids.append(session.id)

        async def handle(
            self, message: NLIPMessage, session: NLIPSession
        ) -> NLIPMessage:
            assert self.started
            self.session_ids.append(session.id)
            self.history_sizes.append(len(session.history))
            return NLIPMessage.text(message.extract_text() or "")

    server = EchoServer()
    app = server.create_nlip_app()
    with TestClient(app) as client:
        response = client.post("/nlip", json=request.to_dict())
        response.raise_for_status()
        first = NLIPMessage.model_validate(response.json())

        followup = NLIPMessage.text("again")
        for token in first.token_submessages():
            followup.add_submessage(token.model_copy(deep=True))
        second_response = client.post("/nlip", json=followup.to_dict())
        second_response.raise_for_status()

        fresh_response = client.post("/nlip", json=NLIPMessage.text("fresh").to_dict())
        fresh_response.raise_for_status()

    assert server.stopped
    assert first.is_control()
    assert first.extract_text() == "hello"
    assert first.extract_conversation_token() == "conversation-1"
    assert any(
        token.subformat == "conversation_echo"
        for token in first.token_submessages()
    )
    assert server.session_ids[0] == server.session_ids[1]
    assert server.session_ids[2] != server.session_ids[0]
    assert server.history_sizes == [0, 1, 0]
    assert server.ended_ids == [server.session_ids[0], server.session_ids[2]]

    assert set(CONFIG["entities"]) == {"claude", "langchain"}


def test_one_client_routes_to_multiple_destinations_and_isolates_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[tuple[str, dict]] = []

    class FakeHTTPClient:
        def __init__(self, **_: object) -> None:
            pass

        async def __aenter__(self) -> "FakeHTTPClient":
            return self

        async def __aexit__(self, *_: object) -> None:
            pass

        async def post(self, url: str, json: dict) -> httpx.Response:
            requests.append((url, json))
            incoming = NLIPMessage.model_validate(json)
            response = NLIPMessage.text(f"reply from {url}")
            response.echo_tokens_from(incoming)
            client_token = next(
                token
                for token in incoming.token_submessages()
                if token.subformat == "conversation_client"
            )
            peer = "claude" if "8011" in url else "langchain"
            response.add_token(
                f"{peer}-token-for-{client_token.content}",
                f"conversation_{peer}",
            )
            return httpx.Response(
                200,
                json=response.to_dict(),
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(client_module.httpx, "AsyncClient", FakeHTTPClient)
    client = NLIPClient.from_config(CONFIG, max_history=2)

    assert client.destinations == ("claude", "langchain")
    assert client.tool_input_schema["properties"]["destination"]["enum"] == [
        "claude",
        "langchain",
    ]
    assert "claude, langchain" in client.tool_description

    scoped = client.session("claude", scope="parent-1")
    assert client.session("claude", scope="parent-1") is scoped
    assert client.session("claude", scope="parent-2") is not scoped
    client.clear_scope("parent-1")
    assert client.session("claude", scope="parent-1") is not scoped

    first_session = client.create_session("claude")
    second_session = client.create_session("claude")

    async def exercise() -> None:
        await client.send_text("claude", "first")
        await client.send_text("langchain", "second")
        await client.send_text("claude", "third")
        await first_session.send_text("independent one")
        await second_session.send_text("independent two")

    asyncio.run(exercise())

    assert requests[0][0] == "http://127.0.0.1:8011/nlip"
    assert requests[1][0] == "http://127.0.0.1:8012/nlip"
    first_tokens = {
        str(token.content)
        for token in NLIPMessage.model_validate(requests[0][1]).token_submessages()
    }
    second_tokens = {
        str(token.content)
        for token in NLIPMessage.model_validate(requests[1][1]).token_submessages()
    }
    third_tokens = {
        str(token.content)
        for token in NLIPMessage.model_validate(requests[2][1]).token_submessages()
    }
    assert first_tokens.isdisjoint(second_tokens)
    assert f"claude-token-for-{client.session('claude').id}" in third_tokens
    assert len(client.session("claude").history) == 2

    assert first_session.id != second_session.id
    assert (
        NLIPMessage.model_validate(requests[-2][1]).extract_conversation_token()
        == first_session.id
    )
    assert (
        NLIPMessage.model_validate(requests[-1][1]).extract_conversation_token()
        == second_session.id
    )
