"""Review agent: reads customer reviews and summarizes them."""

from __future__ import annotations

from langchain.tools import tool

from nlip_sdk import NLIPMessage, NLIPServer, NLIPSession
from nlip_sdk.nlip_security_extensions import SecurityExtensionManager

from .. import catalog, protocol
from .._config import CONFIG
from ..llm import AgentFactory, LLMAgent
from . import serve

ENTITY_NAME = "review_agent"

SYSTEM_PROMPT = (
    "You are the review agent of an online shop. Use get_reviews for each product id you "
    "are asked about, then give a short, balanced summary of what customers say, with the "
    "average rating per product."
)


class ReviewAgent(NLIPServer):
    def __init__(self, agent_factory: AgentFactory = LLMAgent) -> None:
        security = SecurityExtensionManager.from_config(CONFIG, entity=ENTITY_NAME)
        super().__init__(ENTITY_NAME, security=security)
        self.agent_factory = agent_factory

    async def handle(self, message: NLIPMessage, session: NLIPSession) -> NLIPMessage:
        prompt = message.extract_text(language=None) or protocol.render_for_prompt(
            message, "orchestrator"
        )
        ratings: dict[str, dict] = {}

        @tool
        async def get_reviews(product_id: str) -> list[dict]:
            """Get customer reviews (rating 1-5 and text) for a product id."""
            items = catalog.reviews().get(product_id, [])
            if items:
                average = sum(review["rating"] for review in items) / len(items)
                ratings[product_id] = {
                    "product_id": product_id,
                    "average_rating": round(average, 2),
                    "review_count": len(items),
                }
            return items

        agent = self.agent_factory(SYSTEM_PROMPT, [get_reviews])
        summary = await agent.chat(prompt)
        return protocol.reply(
            summary,
            {
                protocol.REVIEW_SUMMARY: {
                    "products": list(ratings.values()),
                    "summary": summary,
                }
            },
        )


nlip_agent = ReviewAgent()
app = nlip_agent.create_nlip_app()


def main() -> None:
    serve(app, ENTITY_NAME)


if __name__ == "__main__":
    main()
