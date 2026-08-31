"""A minimal audit extension that observes every event and always allows it."""

import json

from nlip_sdk.nlip_security_extensions import (
    SecurityDecision,
    SecurityEvent,
    SecurityExtension,
)


class AuditSecurityExtension(SecurityExtension):
    """Print the current SE event, without enforcing policy."""

    async def evaluate(self, event: SecurityEvent) -> SecurityDecision:
        print(
            json.dumps(
                {
                    "security_event": {
                        "point": event.point.value,
                        "component": event.component,
                        "local_entity": event.local_entity,
                        "peer": event.peer,
                        "message": event.message.to_dict(),
                        "related_message": (
                            event.related_message.to_dict()
                            if event.related_message is not None
                            else None
                        ),
                        "session": {
                            "id": event.session.id,
                            "peer": event.session.peer,
                        },
                    }
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return SecurityDecision.allow()
