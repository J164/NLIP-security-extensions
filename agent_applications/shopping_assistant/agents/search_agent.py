"""Product search agent: finds catalog items and reads their (untrusted) product pages."""

from __future__ import annotations

from typing import Any

from langchain.tools import tool

from nlip_sdk import NLIPMessage, NLIPServer, NLIPSession
from nlip_sdk.nlip_security_extensions import SecurityExtensionManager

from .. import catalog, protocol
from .._config import CONFIG
from ..llm import AgentFactory, LLMAgent
from . import serve

ENTITY_NAME = "search_agent"

SYSTEM_PROMPT = (
    "You are the product search agent of an online shop. Use search_catalog to find "
    "products matching the request and fetch_product_page to read details when useful. "
    "Reply with a short summary of the best matches, including product ids and prices."
)


class SearchAgent(NLIPServer):
    def __init__(self, agent_factory: AgentFactory = LLMAgent) -> None:
        security = SecurityExtensionManager.from_config(CONFIG, entity=ENTITY_NAME)
        super().__init__(ENTITY_NAME, security=security)
        self.agent_factory = agent_factory

    async def handle(self, message: NLIPMessage, session: NLIPSession) -> NLIPMessage:
        query = message.extract_text(language=None) or protocol.render_for_prompt(
            message, "orchestrator"
        )
        found: dict[str, dict[str, Any]] = {}

        @tool
        async def search_catalog(query: str, max_price: float | None = None) -> list[dict]:
            """Search the shop catalog. Returns matching products with id, title and price."""
            results = catalog.search(query, max_price)
            for item in results:
                found.setdefault(item["id"], item)
            return results

        @tool
        async def fetch_product_page(product_id: str) -> str:
            """Fetch the product's web page text by product id."""
            return catalog.product_page(product_id) or f"No page for {product_id}."

        # A fresh agent per request: search has no conversational state to keep.
        agent = self.agent_factory(SYSTEM_PROMPT, [search_catalog, fetch_product_page])
        summary = await agent.chat(query)
        results = [
            {key: item[key] for key in ("id", "title", "price", "category", "url")}
            for item in found.values()
        ]
        return protocol.reply(
            summary, {protocol.PRODUCT_RESULTS: {"query": query, "results": results}}
        )


nlip_agent = SearchAgent()
app = nlip_agent.create_nlip_app()


def main() -> None:
    serve(app, ENTITY_NAME)


if __name__ == "__main__":
    main()
