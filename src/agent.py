import json
import logging

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
)
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from pipeline_config import (
    build_llm,
    build_stt,
    build_tts,
    pipeline_instructions,
)

logger = logging.getLogger("agent")

load_dotenv(".env.local")


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


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


@server.rtc_session(agent_name="voice-agent")
async def my_agent(ctx: JobContext):
    from livekit.plugins import google, xai

    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    config = _load_job_config(ctx)
    mode = config.get("mode", "stt-llm-tts")
    logger.info("Job mode=%s metadata_keys=%s", mode, list(config.keys()))

    if mode == "audio-to-audio":
        provider = config.get("provider", "gemini")
        model = config.get("model")
        voice = config.get("voiceId")
        prompt = config.get("systemPrompt") or ""

        if provider == "gemini":
            kwargs = {
                "model": model or "gemini-3.1-flash-live-preview",
                "voice": voice or "Puck",
            }
            if prompt:
                kwargs["instructions"] = prompt
            llm = google.realtime.RealtimeModel(**kwargs)
        else:
            llm = xai.realtime.RealtimeModel(
                model=model or "grok-voice-think-fast-1.0",
                voice=voice or "ara",
            )

        session = AgentSession(llm=llm)
        await ctx.connect()
        await session.start(
            agent=Agent(instructions=prompt or "You are a helpful voice assistant."),
            room=ctx.room,
        )
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

        await ctx.connect()
        await session.start(
            agent=Agent(instructions=instructions),
            room=ctx.room,
        )


if __name__ == "__main__":
    cli.run_app(server)
