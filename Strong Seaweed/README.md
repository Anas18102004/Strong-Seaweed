# BlueWeave + Strong Seaweed (Full Stack + Model)

This repository contains:

- `blue-weave-aqua-main/`: frontend + Node backend + Python agent gateway
- `Strong-Seaweed-main/`: model training assets + species model API

`Strong-Seaweed-main` includes an inference runtime pack so a fresh clone can serve trained models.
See: `Strong-Seaweed-main/RUNTIME_PACK.md`.

## What You Can Run

- Web app UI (`http://127.0.0.1:8080`)
- App backend (`http://127.0.0.1:4000`)
- Agent gateway (`http://127.0.0.1:8101`)
- Species model API (`http://127.0.0.1:8000`)

## Prerequisites

- Node.js 18+ (recommended 20+)
- Python 3.10+ (3.11 recommended)
- MongoDB local (`mongodb://127.0.0.1:27017`)

## Quick Start (Windows PowerShell)

```powershell
cd "<repo-root>"
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_windows.ps1
```

Then run services in separate terminals:

```powershell
# Terminal 1 - model API
cd "<repo-root>\Strong-Seaweed-main"
.\.venv_model_api\Scripts\python serve_species_api.py

# Terminal 2 - agent gateway
cd "<repo-root>\blue-weave-aqua-main\server\agents_python"
.\.venv\Scripts\python -m uvicorn agent_gateway:app --host 127.0.0.1 --port 8101 --reload

# Terminal 3 - app backend
cd "<repo-root>\blue-weave-aqua-main\server"
npm run dev

# Terminal 4 - frontend
cd "<repo-root>\blue-weave-aqua-main"
npm run dev
```

## Manual Setup

### 1) Frontend

```powershell
cd "<repo-root>\blue-weave-aqua-main"
npm install
```

### 2) Backend

```powershell
cd "<repo-root>\blue-weave-aqua-main\server"
Copy-Item .env.example .env -Force
npm install
```

Edit `server/.env` if needed:

- `MONGODB_URI` (default local Mongo)
- LLM keys (`GROQ_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`) optional
- Voice settings:
  - default free fallback: `VOICE_TTS_PROVIDER=browser`
  - ElevenLabs optional via `ELEVENLABS_*`

### 3) Agent Gateway (Python)

```powershell
cd "<repo-root>\blue-weave-aqua-main\server\agents_python"
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

### 4) Model API (Python)

```powershell
cd "<repo-root>\Strong-Seaweed-main"
python -m venv .venv_model_api
.\.venv_model_api\Scripts\python -m pip install -r requirements-model-api.txt
```

## Environment Templates

- `blue-weave-aqua-main/server/.env.example`
- `Strong-Seaweed-main/.env.example`

Never commit real secrets to `.env`.

## Provider Keys (Optional)

The stack runs without paid providers:

- Chat/agents: uses configured gateway providers when keys exist.
- Voice: defaults to `VOICE_TTS_PROVIDER=browser` (no ElevenLabs key needed).

If you add external API keys, put them only in local `.env` files.

## Verify Running Stack

After starting all services, run:

```powershell
cd "<repo-root>"
powershell -ExecutionPolicy Bypass -File .\scripts\verify_stack_windows.ps1
```

Unix/macOS:

```bash
cd /path/to/Strong\ Seaweed
bash ./scripts/verify_stack_unix.sh
```

## Notes for Contributors

- Keep large generated files out of git (`.venv*`, caches, logs, raw outputs).
- Keep reproducible scripts in repo, store heavyweight artifacts in releases/storage.
- If you need to publish trained models, prefer:
  - GitHub Release assets, or
  - a model registry/object store and a fetch script.
