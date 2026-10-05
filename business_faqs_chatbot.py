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
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 3
SIMILARITY_THRESHOLD = 0.45

LLM_API_KEY = os.environ.get("LLM_API_KEY")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL")  # leave unset for the real OpenAI endpoint
USE_LLM = bool(LLM_API_KEY)

HANDOFF_MESSAGE = (
    "لست متأكداً تماماً من الإجابة على هذا السؤال بناءً على المعلومات المتوفرة لدي. "
    "لا أريد أن أعطيك معلومات غير دقيقة، لذا سأقوم بتحويلك إلى أحد ممثلي خدمة العملاء -- "
    "يمكنك التواصل معهم عبر support@example.com أو بكتابة 'أريد التحدث مع موظف'."
)

ERROR_MESSAGE = (
    "عذراً، حدثت مشكلة في الوصول إلى خدمة الذكاء الاصطناعي الآن. "
    "يرجى المحاولة مرة أخرى بعد قليل، أو التواصل مباشرة مع support@example.com."
)


def load_faqs() -> list[dict]:
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class FAQChatbot:
    def __init__(self, faqs: list[dict], index, embedder: SentenceTransformer) -> None:
        self.faqs = faqs
        self.index = index
        self.embedder = embedder
        self.client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL) if USE_LLM else None
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
            "أنت مساعد خدمة عملاء ودود ومختصر. أجب على سؤال العميل "
            "باستخدام المعلومات الموجودة في سياق الأسئلة الشائعة أدناه فقط. "
            "إذا كان السياق لا يجيب على السؤال بشكل كامل، قل ذلك بصراحة بدلاً "
            "من التخمين. أجب دائماً باللغة العربية الفصحى، وبإيجاز."
        )
        user_prompt = f"سؤال العميل: {user_question}\n\nسياق الأسئلة الشائعة:\n{context_block}"

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

    print("Mujeeb AI: مرحباً بك! أنا مساعد خدمة العملاء الخاص بك.")
    print("Mujeeb AI: يمكنك سؤالي عن الشحن، الإرجاع، الدفع، حسابك، طلباتك، أو الضمان.")
    print("Mujeeb AI: اكتب 'خروج' لإنهاء المحادثة.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "خروج":
            print("Mujeeb AI: شكراً لتواصلك معنا، نتمنى لك يوماً سعيداً")
            break
        if not user_input:
            continue
        print(f"Mujeeb AI: {bot.ask(user_input)}\n")


if __name__ == "__main__":
    main()
