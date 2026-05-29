"""Proactive greeting when conversation.agentSpeaksFirst is set in job metadata."""

from __future__ import annotations

import logging
from typing import Any

from livekit.agents import Agent, AgentSession

logger = logging.getLogger("agent")

DEFAULT_WELCOME = "Greet the user and offer your assistance."
DEFAULT_ASSISTANT = "You are a helpful voice assistant."


def greeting_from_config(config: dict[str, Any]) -> tuple[str | None, bool]:
    """Return (greeting instructions, interruptible) or (None, True) if user speaks first."""
    conv = config.get("conversation") or {}
    speaks_first = conv.get("agentSpeaksFirst") or conv.get("welcomeMessageEnabled")
    if not speaks_first:
        return None, True

    instructions = (conv.get("welcomeMessage") or "").strip() or DEFAULT_WELCOME
    interruptible = conv.get("welcomeInterruptible", True)
    return instructions, interruptible


class GreetingAgent(Agent):
    """Uses on_enter so Gemini/Grok realtime actually speaks first (post-start generate_reply is flaky)."""

    def __init__(
        self,
        *,
        instructions: str,
        greeting: str | None = None,
        greeting_interruptible: bool = True,
    ) -> None:
        super().__init__(instructions=instructions or DEFAULT_ASSISTANT)
        self._greeting = greeting
        self._greeting_interruptible = greeting_interruptible

    async def on_enter(self) -> None:
        if not self._greeting:
            return
        logger.info(
            "Agent speaks first via on_enter (interruptible=%s)",
            self._greeting_interruptible,
        )
        await self.session.generate_reply(
            instructions=self._greeting,
            allow_interruptions=self._greeting_interruptible,
        )


async def maybe_speak_first(session: AgentSession, config: dict[str, Any]) -> None:
    """Legacy fallback — prefer GreetingAgent.on_enter for realtime models."""
    greeting, interruptible = greeting_from_config(config)
    if not greeting:
        return

    logger.info(
        "Agent speaks first: greeting fallback (interruptible=%s)",
        interruptible,
    )
    await session.generate_reply(
        instructions=greeting,
        allow_interruptions=interruptible,
    )
