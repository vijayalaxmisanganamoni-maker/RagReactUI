"""Answer generation.

Two providers:
  - "groq":  Llama 3.1 8B Instant via the Groq API (needs GROQ_API_KEY) - best quality
  - "local": google/flan-t5-base via transformers - runs on CPU, no API key
"auto" picks groq when a key is available, otherwise local.
"""
import os

from .config import Config

SYSTEM_PROMPT = (
    "You are a helpful customer-support assistant. Answer the user's question "
    "using ONLY the information in the provided context passages. Be concise and "
    "factual. If the context does not contain the answer, say you could not find "
    "the answer in the support documents - do not make anything up. When useful, "
    "reference the passage numbers you used, e.g. [1]."
)


def build_prompt(question: str, context_chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i + 1}] ({c['metadata'].get('source', '?')}) {c['text']}"
        for i, c in enumerate(context_chunks)
    )
    return (
        f"Context passages:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


class Generator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        provider = cfg.llm_provider
        if provider == "auto":
            provider = "groq" if os.getenv("GROQ_API_KEY") else "local"
        self.provider = provider
        if provider == "groq":
            from groq import Groq
            self._client = Groq()
        elif provider == "local":
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(cfg.local_model)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(cfg.local_model)
        else:
            raise ValueError(f"Unknown llm_provider: {provider}")

    def generate(self, question: str, context_chunks: list[dict]) -> str:
        prompt = build_prompt(question, context_chunks)
        if self.provider == "groq":
            resp = self._groq_call([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ], self.cfg.temperature, self.cfg.max_new_tokens)
            return resp.choices[0].message.content.strip()

        # local flan-t5: small models do better with a short, direct instruction
        # (no citation/passage-number formatting) and plain context
        context = "\n\n".join(c["text"] for c in context_chunks)
        text = (
            "Answer the question using only the context below. If the answer "
            "is not in the context, say: I could not find the answer in the "
            f"support documents.\n\nContext:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=1024
        )
        output = self._model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            num_beams=4,
            early_stopping=True,
        )
        return self._tokenizer.decode(output[0], skip_special_tokens=True).strip()

    def _groq_call(self, messages: list[dict], temperature: float, max_tokens: int):
        """Groq chat completion with backoff on free-tier rate limits."""
        import time
        from groq import RateLimitError
        for attempt in range(4):
            try:
                return self._client.chat.completions.create(
                    model=self.cfg.groq_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except RateLimitError:
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))

    def chat(self, message: str) -> str:
        """Direct answer without retrieval (used for small talk)."""
        if self.provider == "groq":
            resp = self._groq_call([
                {"role": "system",
                 "content": "You are a friendly customer-support assistant. "
                            "Reply briefly and offer to help with product questions."},
                {"role": "user", "content": message},
            ], 0.5, 128)
            return resp.choices[0].message.content.strip()
        return ("Hello! I'm your customer-support assistant. Ask me a question "
                "about the products in my knowledge base and I'll look it up for you.")
