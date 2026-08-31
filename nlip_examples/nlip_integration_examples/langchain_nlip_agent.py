"""LangChainTestAgent wrapped with NLIP client and server capabilities."""

from __future__ import annotations

from langchain.tools import tool
from nlip_examples.nlip_agnostic_agents.langchain_agent import LangChainTestAgent
from nlip_examples.nlip_integration_examples._config import CONFIG
from nlip_sdk import (
    NLIPClient,
    NLIPMessage,
    NLIPServer,
    NLIPSession,
)
from nlip_sdk.nlip_security_extensions import SecurityExtensionManager

ENTITY_NAME = "langchain"


class LangChainNLIPAgent(NLIPServer):
    """Expose LangChain as a server and let it call configured entities via NLIP."""

    def __init__(self) -> None:
        # One manager sees both server traffic and this agent's outbound client traffic.
        security = SecurityExtensionManager.from_config(CONFIG, entity=ENTITY_NAME)
        super().__init__(ENTITY_NAME, security=security)
        # Excluding this entity prevents the model from recursively calling itself.
        self.nlip_client = NLIPClient.from_config(
            CONFIG,
            exclude={ENTITY_NAME},
            identity=ENTITY_NAME,
            security=security,
        )
        # Each inbound NLIP session owns one stateful LangChain agent instance.
        self._agent_instances: dict[str, LangChainTestAgent] = {}

    async def session_started(self, server_session: NLIPSession) -> None:
        # LangChain can call a normal async function, but its decorator attaches the
        # config-derived description and destination enum to the model-visible schema.
        @tool(
            NLIPClient.TOOL_NAME,
            description=self.nlip_client.tool_description,
            args_schema=self.nlip_client.tool_input_schema,
        )
        async def send_nlip_message_tool(destination: str, message: str) -> str:
            destination = destination.casefold()
            return await self.nlip_client.send_nlip_message(
                destination, message, scope=server_session.id
            )

        # Transport and destination selection stay in NLIPClient, not in this tool layer.
        agent_instance = LangChainTestAgent(
            system_prompt=(
                "You are the LangChain agent. When asked to contact another entity, call "
                "send_nlip_message and use the returned response in your reply."
            ),
            tools=[send_nlip_message_tool],
        )
        self._agent_instances[server_session.id] = agent_instance

    async def session_ended(self, server_session: NLIPSession) -> None:
        self._agent_instances.pop(server_session.id, None)
        self.nlip_client.clear_scope(server_session.id)

    async def handle(
        self, message: NLIPMessage, server_session: NLIPSession
    ) -> NLIPMessage:
        text = message.extract_text(language=None)
        if text is None:
            return NLIPMessage.text("This example agent currently accepts NLIP text messages.")

        agent_instance = self._agent_instances.get(server_session.id)
        if agent_instance is None:
            raise RuntimeError("LangChain session has not started.")
        # Ask the framework-native agent to handle the plain text prompt.
        agent_response = await agent_instance.chat(text)
        # Wrap its plain text response for the NLIP transport pipeline.
        return NLIPMessage.text(agent_response)


# Module-level ``app`` lets both ``python -m ...`` and an ASGI server import it.
nlip_agent = LangChainNLIPAgent()
app = nlip_agent.create_nlip_app()


def main() -> None:
    import uvicorn

    settings = CONFIG["entities"][ENTITY_NAME]
    uvicorn.run(app, host=settings["host"], port=settings["port"])


if __name__ == "__main__":
    main()
