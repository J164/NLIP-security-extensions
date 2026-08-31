import asyncio

import httpx
import pytest

from nlip_examples.nlip_integration_examples._config import CONFIG
from nlip_sdk import NLIPClient, NLIPMessage, NLIPServer, NLIPSession
from nlip_sdk.nlip_security_extensions import (
    EnforcementPoint,
    SecurityBlockedError,
    SecurityDecision,
    SecurityEvent,
    SecurityExtension,
    SecurityExtensionManager,
)


def test_security_extension_loading_order_replace_and_block(capsys) -> None:
    configured = SecurityExtensionManager.from_config(CONFIG, entity="claude")
    assert configured.extension_names(EnforcementPoint.INGRESS) == ("audit_example",)
    assert configured.extension_names(EnforcementPoint.EGRESS) == ("audit_example",)

    session = NLIPSession("test-session", peer="peer")
    asyncio.run(
        configured.enforce_ingress(
            NLIPMessage.text("observed"), session, component="server"
        )
    )
    audit_output = capsys.readouterr().out
    assert '"point": "ingress"' in audit_output
    assert '"content": "observed"' in audit_output
    assert '"history"' not in audit_output

    seen: list[str] = []

    class Replace(SecurityExtension):
        async def evaluate(self, event: SecurityEvent) -> SecurityDecision:
            seen.append(str(event.message.content))
            return SecurityDecision.replace(NLIPMessage.text("replacement"))

    class Block(SecurityExtension):
        async def evaluate(self, event: SecurityEvent) -> SecurityDecision:
            seen.append(str(event.message.content))
            return SecurityDecision.block("test policy")

    manager = SecurityExtensionManager("test")
    manager.register("replace", Replace(), [EnforcementPoint.INGRESS])
    manager.register("block", Block(), [EnforcementPoint.INGRESS])

    with pytest.raises(SecurityBlockedError, match="test policy"):
        asyncio.run(
            manager.enforce_ingress(
                NLIPMessage.text("original"), session, component="client"
            )
        )
    assert seen == ["original", "replacement"]


def test_client_and_server_map_transport_to_ingress_and_egress() -> None:
    seen: list[tuple[str, str, str]] = []

    class Record(SecurityExtension):
        async def evaluate(self, event: SecurityEvent) -> SecurityDecision:
            seen.append((event.local_entity, event.component, event.point.value))
            return SecurityDecision.allow()

    def recording_manager(entity: str) -> SecurityExtensionManager:
        manager = SecurityExtensionManager(entity)
        extension = Record()
        manager.register("record", extension, list(EnforcementPoint))
        return manager

    class EchoServer(NLIPServer):
        async def handle(
            self, message: NLIPMessage, session: NLIPSession
        ) -> NLIPMessage:
            return NLIPMessage.text(message.extract_text() or "")

    server = EchoServer("server", security=recording_manager("server"))
    asyncio.run(server.process_exchange(NLIPMessage.text("hello")))

    client = NLIPClient(
        {"server": "http://server.test/nlip"},
        identity="client",
        security=recording_manager("client"),
    )

    async def fake_post(destination: str, message: NLIPMessage) -> httpx.Response:
        response = NLIPMessage.text("hello")
        response.echo_tokens_from(message)
        return httpx.Response(
            200,
            json=response.to_dict(),
            request=httpx.Request("POST", client.endpoints[destination]),
        )

    client._post = fake_post  # type: ignore[method-assign]
    asyncio.run(client.send_text("server", "hello"))

    assert seen == [
        ("server", "server", "ingress"),
        ("server", "server", "egress"),
        ("client", "client", "egress"),
        ("client", "client", "ingress"),
    ]


def test_server_serializes_complete_turns_within_one_session() -> None:
    class SerialServer(NLIPServer):
        active = 0
        max_active = 0
        last_session: NLIPSession | None = None

        async def handle(
            self, message: NLIPMessage, session: NLIPSession
        ) -> NLIPMessage:
            self.last_session = session
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            return NLIPMessage.text(str(message.content))

    server = SerialServer("serial")

    async def exercise() -> None:
        first = await server.process_exchange(NLIPMessage.text("first"))
        requests = [NLIPMessage.text("second"), NLIPMessage.text("third")]
        for request in requests:
            request.echo_tokens_from(first)
        await asyncio.gather(
            *(server.process_exchange(request) for request in requests)
        )

    asyncio.run(exercise())
    assert server.max_active == 1
    assert server.last_session is not None
    assert len(server.last_session.history) == 3
