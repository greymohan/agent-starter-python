"""Proactive greeting when conversation.agentSpeaksFirst is set in job metadata."""

from __future__ import annotations

import logging
from typing import Any

from livekit.agents import AgentSession

logger = logging.getLogger("agent")

DEFAULT_WELCOME = "Greet the user and offer your assistance."


async def maybe_speak_first(session: AgentSession, config: dict[str, Any]) -> None:
    conv = config.get("conversation") or {}
    speaks_first = conv.get("agentSpeaksFirst") or conv.get("welcomeMessageEnabled")
    if not speaks_first:
        return

    instructions = (conv.get("welcomeMessage") or "").strip() or DEFAULT_WELCOME
    interruptible = conv.get("welcomeInterruptible", True)

    logger.info(
        "Agent speaks first: greeting (interruptible=%s)",
        interruptible,
    )
    await session.generate_reply(
        instructions=instructions,
        allow_interruptions=interruptible,
    )
