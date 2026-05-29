import json
import logging
import os
import time

from dotenv import load_dotenv
from livekit.agents import (
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
)
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from greeting import (
    GreetingAgent,
    greeting_from_config,
    merge_realtime_instructions,
)
from pipeline_config import (
    build_llm,
    build_stt,
    build_tts,
    pipeline_instructions,
)

logger = logging.getLogger("agent")

load_dotenv(".env.local")


def _server_options() -> dict:
    """Optional overrides; production `start` already warms idle processes by default."""
    opts: dict = {}
    raw = os.environ.get("AGENT_NUM_IDLE_PROCESSES", "").strip()
    if raw:
        opts["num_idle_processes"] = max(0, int(raw))
    return opts


server = AgentServer(**_server_options())


def prewarm(proc: JobProcess):
    # VAD only — MultilingualModel() needs job context (inference_executor).
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("Prewarm complete: VAD")


server.setup_fnc = prewarm


def _load_job_config(ctx: JobContext) -> dict:
    raw = ctx.job.metadata
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Invalid job metadata, using pipeline fallback")
        return {}


def _log_dispatch(config: dict, mode: str) -> None:
    prompt = config.get("systemPrompt") or (config.get("llm") or {}).get("systemPrompt") or ""
    greeting, _ = greeting_from_config(config)
    logger.info(
        "Dispatch mode=%s speak_first=%s system_prompt_chars=%d metadata_keys=%s",
        mode,
        greeting is not None,
        len(prompt),
        list(config.keys()),
    )


@server.rtc_session(agent_name="voice-agent")
async def my_agent(ctx: JobContext):
    from livekit.plugins import google, xai

    t0 = time.monotonic()
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    config = _load_job_config(ctx)
    mode = config.get("mode", "stt-llm-tts")
    _log_dispatch(config, mode)

    greeting, greeting_interruptible = greeting_from_config(config)
    await ctx.connect()
    logger.info("Room connected in %.2fs", time.monotonic() - t0)

    if mode == "audio-to-audio":
        provider = config.get("provider", "gemini")
        model = config.get("model")
        voice = config.get("voiceId")
        prompt = config.get("systemPrompt") or ""
        realtime_instructions = merge_realtime_instructions(prompt, greeting)

        voice_name = (voice or "Puck").strip()
        if provider == "gemini":
            logger.info("Gemini realtime voice=%s", voice_name)
            llm = google.realtime.RealtimeModel(
                model=model or "gemini-3.1-flash-live-preview",
                voice=voice_name,
                instructions=realtime_instructions,
            )
        else:
            llm = xai.realtime.RealtimeModel(
                model=model or "grok-voice-think-fast-1.0",
                voice=voice or "ara",
            )

        session = AgentSession(llm=llm)
        # Gemini Live rejects generate_reply; Grok may still use on_enter when speak-first is on.
        agent = GreetingAgent(
            instructions=realtime_instructions,
            greeting=greeting,
            greeting_interruptible=greeting_interruptible,
            use_generate_reply=provider != "gemini",
        )
        await session.start(agent=agent, room=ctx.room)
        logger.info("Realtime session started in %.2fs", time.monotonic() - t0)
    else:
        stt_cfg = config.get("stt") or {}
        llm_cfg = config.get("llm") or {}
        tts_cfg = config.get("tts") or {}
        instructions = pipeline_instructions(config)

        logger.info(
            "Pipeline dispatch stt=%s/%s llm=%s/%s tts=%s/%s voice=%s",
            stt_cfg.get("provider"),
            stt_cfg.get("model"),
            llm_cfg.get("provider"),
            llm_cfg.get("model"),
            tts_cfg.get("provider"),
            tts_cfg.get("model"),
            tts_cfg.get("voiceId"),
        )

        session = AgentSession(
            stt=build_stt(stt_cfg),
            llm=build_llm(llm_cfg),
            tts=build_tts(tts_cfg),
            turn_detection=MultilingualModel(),
            vad=ctx.proc.userdata["vad"],
            preemptive_generation=True,
        )
        agent = GreetingAgent(
            instructions=instructions,
            greeting=greeting,
            greeting_interruptible=greeting_interruptible,
        )
        await session.start(agent=agent, room=ctx.room)
        logger.info("Pipeline session started in %.2fs", time.monotonic() - t0)


if __name__ == "__main__":
    cli.run_app(server)
