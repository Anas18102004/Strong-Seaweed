# Akuara App

This folder contains:

- `src/`: React frontend
- `server/`: Node backend (auth/chat/voice routes)
- `server/agents_python/`: Python agent gateway

For full stack + model setup, use the repository root guide:

- [`../README.md`](../README.md)

## Quick local run (app only)

### Frontend

```powershell
cd blue-weave-aqua-main
npm install
npm run dev
```

### Backend

```powershell
cd blue-weave-aqua-main/server
copy .env.example .env
npm install
npm run dev
```

### Agent Gateway

```powershell
cd blue-weave-aqua-main/server/agents_python
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m uvicorn agent_gateway:app --host 127.0.0.1 --port 8101 --reload
```

## Environment

Use `server/.env.example` as the template. Do not commit real API keys in `server/.env`.
