# Deploy on Dokploy (production)

Use this checklist so agents are **warm before the first call** and stay available.

## Container command

The Dockerfile already runs production mode:

```bash
uv run src/agent.py start
```

Do **not** use `dev` in Dokploy (`dev` sets idle processes to 0 → cold start every call).

## Dokploy app settings

| Setting | Value |
|--------|--------|
| **Replicas** | Min **1** — no scale-to-zero |
| **Restart policy** | Restart on failure |
| **Health check** | HTTP on worker metrics port if exposed, or process liveness via Dokploy |
| **Region / host** | Same VPS as `livekit.ideskai.com` (your LiveKit stack) |

## Resources (idle process pool)

Production `start` keeps several **prewarmed** worker processes (framework default scales with CPU).

Each idle process preloads **Silero VAD** only. The multilingual turn detector loads on first pipeline job (requires LiveKit job context).

**Small VPS (2 vCPU / 4 GB RAM):** set in Dokploy env:

```env
AGENT_NUM_IDLE_PROCESSES=2
```

**Larger VPS:** omit the variable (use LiveKit default, up to ~8).

## Required env (`.env` / Dokploy secrets)

- `LIVEKIT_URL` — e.g. `wss://livekit.ideskai.com`
- `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET`
- Provider keys used by your agents: `OPENAI_API_KEY`, `DEEPGRAM_API_KEY`, `CARTESIA_API_KEY`, `GOOGLE_API_KEY`, `XAI_API_KEY`, etc.

## Verify after deploy

Worker logs should include:

1. `ready to accept jobs` (or equivalent registration line)
2. `Prewarm complete: VAD` (per idle process)
3. On first call: `Job mode=...`, and if speak-first is on: `Agent speaks first: greeting`

## Build note

`Dockerfile` runs `uv run src/agent.py download-files` at build time so turn-detector models are baked into the image — required for fast prewarm.
