# DigiAssist — Business Customer Support RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers customer
questions (shipping, returns, payments, account, orders, support,
discounts, warranty, and more) grounded in a FAQ knowledge base, with an
honest fallback when it doesn't know something.

## Why RAG instead of a plain LLM

The bot never answers from the model's general knowledge. Every answer is
generated only from FAQ entries retrieved as relevant to the question. If
nothing in the knowledge base is similar enough to the question, the bot
says so and hands off to a human agent instead of guessing — a wrong
answer about refunds or payments carries real business risk.

## Project structure

```
business_chatbot_en/
├── business_chatbot.py   # entry point — run this
├── knowledge_base.json   # 100 FAQ entries (question, answer, category)
├── requirements.txt
├── .env                  # your API key and model config (never commit this)
└── .gitignore
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Open `.env` and fill in your own key:

```
LLM_API_KEY=sk-your-key-here
LLM_MODEL=gpt-4o-mini
# LLM_BASE_URL=   # only needed for a non-OpenAI provider, see below
```

## Run

```bash
python business_chatbot.py
```

Type a question, get an answer. Type `exit` or `quit` to leave.

Example questions to try:
- "How long does shipping take?"
- "What is your return policy?"
- "Do you have a loyalty program?"
- "Do you sell on Mars?" (tests the human-handoff fallback)

## Running without an API key

If `LLM_API_KEY` is left empty, the bot still works — it returns the
best-matching FAQ answer directly instead of a natural-language reply.
Useful for testing retrieval without spending on API calls.

## Using a different provider

OpenAI is the default (no `LLM_BASE_URL` needed). For another
OpenAI-compatible provider, set both:

```
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com
```

Only use a provider's official endpoint. Never point `LLM_BASE_URL` at an
unfamiliar third-party domain — that sends your API key to someone else's
server.

## Customizing the knowledge base

Replace the contents of `knowledge_base.json` with your own company's
FAQs, keeping the same structure:

```json
{
  "id": "faq001",
  "category": "shipping",
  "question": "How long does shipping take?",
  "answer": "..."
}
```

No code changes needed — the bot re-embeds whatever is in the file each
time it starts.

## Tuning retrieval behavior

In `business_chatbot.py`:

- `TOP_K` — how many FAQ entries are retrieved per question (default 3).
- `SIMILARITY_THRESHOLD` — how close a match must be before the bot
  answers instead of handing off to a human (default 0.45). Lower it if
  the bot hands off too often; raise it if it answers questions it
  shouldn't.

## Deploying

This runs as a plain Python script (stdin/stdout), so it's easy to wrap
with:

- A web framework (FastAPI/Flask) exposing `FAQChatbot.ask()` as an
  endpoint.
- A messaging platform webhook (WhatsApp Business API, Telegram, Slack).
- A scheduled or long-running process on any VPS, Docker container, or
  platform that runs a Python process (Railway, Render, Fly.io, etc.).

Whichever route you take, keep `LLM_API_KEY` as a platform secret /
environment variable — never hardcode it or commit `.env`.

## Known limitations

- `knowledge_base.json` is a static file; updating it requires a restart
  to re-build the retrieval index.
- No persistent conversation storage — chat history resets each run.
- No authentication or rate limiting; add these before exposing it
  publicly.
