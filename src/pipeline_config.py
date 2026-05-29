"""Build STT / LLM / TTS from Studio dispatch metadata (stt-llm-tts mode)."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from livekit.plugins import assemblyai, cartesia, deepgram, openai, xai

logger = logging.getLogger("agent")

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.I,
)

# Studio short ids → Cartesia voice UUIDs (sonic-3), max 5 in UI
CARTESIA_VOICE_UUIDS: dict[str, str] = {
    "blake": "a167e0f3-df7e-4d52-a9c3-f949145efdab",
    "jacqueline": "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
    "daniela": "694f9389-aac1-45b6-b726-9d9369183238",
    "samuel": "79f8b5fb-2cc8-479a-80df-29f7a7cf1a3e",
    "evelyn": "e00d0e4c-a5c8-443f-a8a3-473eb9a62355",
}

GROK_TTS_VOICES = frozenset({"ara", "eve", "rex", "sal", "leo"})

DEEPGRAM_MODEL_IDS = {
    "deepgram-nova-3": "nova-3",
    "deepgram-nova-2": "nova-2",
    "deepgram-flux": "flux",
}

DEFAULT_INSTRUCTIONS = (
    "You are a friendly, reliable voice assistant. "
    "Respond in plain text only, briefly, for text-to-speech."
)


def _resolve_cartesia_voice(voice_id: str | None) -> str:
    if not voice_id:
        return CARTESIA_VOICE_UUIDS["blake"]
    if UUID_RE.match(voice_id):
        return voice_id
    resolved = CARTESIA_VOICE_UUIDS.get(voice_id.lower())
    if resolved:
        return resolved
    logger.warning("Unknown Cartesia voice id %r — using Blake", voice_id)
    return CARTESIA_VOICE_UUIDS["blake"]


def build_stt(stt_cfg: dict[str, Any]):
    provider = stt_cfg.get("provider", "deepgram")
    model_id = stt_cfg.get("model", "deepgram-nova-3")
    language = stt_cfg.get("language") or "en"

    if provider == "deepgram":
        model = DEEPGRAM_MODEL_IDS.get(model_id, model_id.replace("deepgram-", ""))
        lang = "multi" if language in ("multi", "multilingual") else language
        if lang == "en":
            lang = "en-US"
        logger.info("Pipeline STT: deepgram model=%s language=%s", model, lang)
        return deepgram.STT(model=model, language=lang)

    if provider == "cartesia":
        model = "ink-whisper"
        if model_id == "cartesia-ink-whisper":
            model = "ink-whisper"
        logger.info("Pipeline STT: cartesia model=%s", model)
        return cartesia.STT(model=model, language=language)

    if provider == "grok":
        logger.info("Pipeline STT: xai (grok-stt-1)")
        lang = "en" if language == "en" else language
        return xai.STT(language=lang)

    if provider == "assemblyai":
        logger.info("Pipeline STT: assemblyai universal")
        return assemblyai.STT()

    logger.warning("Unknown STT provider %r — falling back to deepgram nova-3", provider)
    return deepgram.STT(model="nova-3", language="en-US")


def _llm_reasoning_kwargs(llm_cfg: dict[str, Any]) -> dict[str, str]:
    effort = llm_cfg.get("reasoningEffort")
    if effort in ("low", "medium", "high"):
        return {"reasoning_effort": effort}
    return {}


def build_llm(llm_cfg: dict[str, Any]):
    provider = llm_cfg.get("provider", "openai")
    model_id = llm_cfg.get("model", "gpt-4o-mini")
    reasoning = _llm_reasoning_kwargs(llm_cfg)

    if provider == "grok":
        api_key = os.environ.get("XAI_API_KEY")
        grok_model = model_id if str(model_id).startswith("grok-") else "grok-3-fast"
        logger.info(
            "Pipeline LLM: xAI model=%s reasoning=%s",
            grok_model,
            reasoning.get("reasoning_effort", "default"),
        )
        return openai.LLM(
            model=grok_model,
            base_url="https://api.x.ai/v1",
            api_key=api_key,
            **reasoning,
        )

    openai_model = model_id if str(model_id).startswith("gpt-") else "gpt-4o-mini"
    logger.info(
        "Pipeline LLM: openai model=%s reasoning=%s",
        openai_model,
        reasoning.get("reasoning_effort", "default"),
    )
    return openai.LLM(model=openai_model, **reasoning)


def build_tts(tts_cfg: dict[str, Any]):
    provider = tts_cfg.get("provider", "cartesia")
    model_id = tts_cfg.get("model", "cartesia-sonic")
    voice_id = tts_cfg.get("voiceId")

    if provider == "grok":
        voice = (voice_id or "ara").lower()
        if voice not in GROK_TTS_VOICES:
            logger.warning("Unknown Grok TTS voice %r — using ara", voice)
            voice = "ara"
        logger.info("Pipeline TTS: xai voice=%s", voice)
        return xai.TTS(voice=voice)

    cartesia_model = "sonic-3"
    if model_id == "cartesia-sonic":
        cartesia_model = "sonic-3"
    voice = _resolve_cartesia_voice(voice_id)
    logger.info("Pipeline TTS: cartesia model=%s voice=%s", cartesia_model, voice)
    return cartesia.TTS(model=cartesia_model, voice=voice)


def pipeline_instructions(config: dict[str, Any]) -> str:
    llm_cfg = config.get("llm") or {}
    return (
        config.get("systemPrompt")
        or llm_cfg.get("systemPrompt")
        or DEFAULT_INSTRUCTIONS
    )
