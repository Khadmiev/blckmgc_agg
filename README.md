# LPO — Chat API Backend

REST API backend for a mobile chat application with multi-LLM support.

## Features

- **Authentication** — JWT (email + password) and OAuth 2.0 (Google, Apple)
- **Multi-LLM chat** — OpenAI, Anthropic, Google Gemini, xAI Grok, Mistral
- **Streaming responses** — Server-Sent Events for real-time token delivery
- **Multimedia** — Image, video, and audio attachments with thumbnail generation
- **Thread management** — Create, list, update, and soft-delete chat threads
- **History context** — Full thread history sent to LLMs for contextual responses

## Tech Stack

- Python 3.12+ / FastAPI / Uvicorn
- PostgreSQL / SQLAlchemy 2.0 (async) / Alembic
- Pydantic v2

## Quick Start

```bash
# 1. Clone and enter the project
cd lpo

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your database URL, API keys, and secrets

# 5. Run database migrations
alembic upgrade head

# 6. Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs are available at `http://localhost:8000/docs` (Swagger UI).

## Docker

```bash
docker compose up -d
```

- **App**: http://localhost:8000
- Uses external PostgreSQL (e.g. Render). Set `DATABASE_URL` in `.env` — use the **External** URL for local Docker.

## Render Deployment

1. **Create a PostgreSQL database** on Render: Dashboard → New → PostgreSQL.
2. **Create a Web Service** from this repo (Docker runtime).
3. **Link the database** to the web service: Environment → Add Environment Variable → `DATABASE_URL` from your Postgres (use the **Internal** connection string).
4. Or use the **Blueprint**: New → Blueprint → connect this repo (uses `render.yaml`).

After deploy, verify the database: `https://your-app.onrender.com/health/db`. If it returns 503, `DATABASE_URL` is missing or incorrect.

## API Overview

All endpoints are prefixed with `/api/v1`.

### Auth — `/api/v1/auth`

| Method | Path             | Description                    |
|--------|------------------|--------------------------------|
| POST   | `/register`      | Register with email + password |
| POST   | `/login`         | Login, returns JWT pair        |
| POST   | `/oauth/google`  | Google ID token exchange       |
| POST   | `/oauth/apple`   | Apple auth code exchange       |
| POST   | `/refresh`       | Refresh access token           |
| GET    | `/me`            | Get current user profile       |
| PATCH  | `/me`            | Update profile                 |

### Threads — `/api/v1/threads`

| Method | Path              | Description                        |
|--------|-------------------|------------------------------------|
| GET    | `/`               | List threads (paginated)           |
| POST   | `/`               | Create new thread (title, llm)     |
| GET    | `/{thread_id}`    | Get thread with message history    |
| PATCH  | `/{thread_id}`    | Update thread (rename, change LLM) |
| DELETE | `/{thread_id}`    | Soft-delete thread                 |

### Chat — `/api/v1/chat`

| Method | Path                         | Description                              |
|--------|------------------------------|------------------------------------------|
| POST   | `/{thread_id}/send`          | Send prompt (+ files), stream response   |
| POST   | `/{thread_id}/regenerate`    | Regenerate last assistant response       |

`/send` accepts `multipart/form-data` with a `prompt` field and optional `files`. Returns an SSE stream with events: `chunk` (text delta), `done` (full response + metadata), `error`.

### Media — `/api/v1/media`

| Method | Path                    | Description          |
|--------|-------------------------|----------------------|
| GET    | `/{media_id}`           | Download media file  |
| GET    | `/{media_id}/thumbnail` | Download thumbnail   |

## Configuration

All settings are loaded from environment variables (or `.env` file). See `.env.example` for the full list.

Key variables:

| Variable            | Description                          |
|---------------------|--------------------------------------|
| `DATABASE_URL`      | PostgreSQL async connection string   |
| `SECRET_KEY`        | JWT signing secret                   |
| `OPENAI_API_KEY`    | OpenAI API key                       |
| `ANTHROPIC_API_KEY` | Anthropic API key                    |
| `GOOGLE_AI_API_KEY` | Google AI (Gemini) API key           |
| `XAI_API_KEY`       | xAI (Grok) API key                   |
| `MISTRAL_API_KEY`   | Mistral API key                      |
| `STORAGE_BACKEND`   | `local` (default) or future: `s3`    |

Only configure API keys for the LLM providers you want to use. Models from unconfigured providers are simply unavailable.

## Project Structure

```
lpo/
├── alembic/              # Database migrations
├── app/
│   ├── main.py           # FastAPI app entry point
│   ├── config.py         # Settings (env vars)
│   ├── database.py       # Async SQLAlchemy engine
│   ├── dependencies.py   # DI: get_db, get_current_user
│   ├── models/           # SQLAlchemy ORM models
│   ├── schemas/          # Pydantic request/response schemas
│   ├── routers/          # API endpoint handlers
│   ├── services/         # Business logic + LLM providers
│   └── storage/          # File storage abstraction
├── media/                # Local uploads (gitignored)
├── requirements.txt
└── .env.example
```
