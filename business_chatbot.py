from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_PATH = BASE_DIR / "knowledge_base.json"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3
SIMILARITY_THRESHOLD = 0.45
BOT_NAME = "DigiAssist"

LLM_API_KEY = os.environ.get("LLM_API_KEY")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL")  # leave unset for the real OpenAI endpoint

HANDOFF_MESSAGE = (
    "I'm not fully sure about that based on what I have available. "
    "I don't want to give you inaccurate information, so let me connect you "
    "with a human support agent -- you can reach them at support@example.com "
    "or by typing 'talk to an agent'."
)

ERROR_MESSAGE = (
    "Sorry, I'm having trouble reaching my AI service right now. "
    "Please try again in a moment, or contact support@example.com directly."
)


def load_faqs() -> list[dict]:
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class FAQChatbot:
    def __init__(self, faqs: list[dict], index, embedder: SentenceTransformer) -> None:
        self.faqs = faqs
        self.index = index
        self.embedder = embedder
        self.client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL) if LLM_API_KEY else None
        self.history: list[dict] = []

    def retrieve(self, query: str, k: int = TOP_K) -> list[dict]:
        query_vec = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, indices = self.index.search(query_vec.astype(np.float32), k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({"score": float(score), **self.faqs[idx]})
        return results

    def generate_answer(self, user_question: str, retrieved: list[dict]) -> str:
        context_block = "\n\n".join(f"Q: {r['question']}\nA: {r['answer']}" for r in retrieved)

        if not self.client:
            return retrieved[0]["answer"]

        system_prompt = (
            "You are a friendly, concise customer support assistant. "
            "Answer the user's question using ONLY the information in the "
            "provided FAQ context. If the context doesn't fully answer the "
            "question, say so honestly instead of guessing. Keep answers short."
        )
        user_prompt = f"Customer question: {user_question}\n\nFAQ context:\n{context_block}"

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *self.history[-6:],
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            answer = response.choices[0].message.content
        except Exception as exc:
            print(f"[LLM error] {exc}")
            answer = ERROR_MESSAGE

        self.history.append({"role": "user", "content": user_question})
        self.history.append({"role": "assistant", "content": answer})
        return answer

    def ask(self, user_question: str) -> str:
        retrieved = self.retrieve(user_question)
        if not retrieved or retrieved[0]["score"] < SIMILARITY_THRESHOLD:
            self.history.append({"role": "user", "content": user_question})
            self.history.append({"role": "assistant", "content": HANDOFF_MESSAGE})
            return HANDOFF_MESSAGE
        return self.generate_answer(user_question, retrieved)


def build_chatbot() -> FAQChatbot:
    faqs = load_faqs()
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [f"{item['question']} {item['answer']}" for item in faqs]
    vectors = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype(np.float32))

    return FAQChatbot(faqs, index, embedder)


def main() -> None:
    bot = build_chatbot()

    print(f"{BOT_NAME}: Hi there! I'm your customer support assistant.")
    print(f"{BOT_NAME}: Ask me about shipping, returns, payments, your account, orders, or warranty.")
    print(f"{BOT_NAME}: Type 'exit' to end the chat.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print(f"{BOT_NAME}: Thanks for chatting, have a great day!")
            break
        if not user_input:
            continue
        print(f"{BOT_NAME}: {bot.ask(user_input)}\n")


if __name__ == "__main__":
    main()
