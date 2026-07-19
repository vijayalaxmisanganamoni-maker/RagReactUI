"""LLM-judge evaluation following RAGBench's TRACe framework.

A strong LLM (Groq, ``cfg.judge_model``) is shown the question, the retrieved
context and the generated answer, both split into keyed sentences ('0a', '0b',
'1a' for documents; 'a', 'b' for the answer). It returns a structured JSON
judgment from which the four TRACe metrics are derived as in the RAGBench
paper:

  judge_relevance    = |relevant doc sentences| / |doc sentences|
  judge_utilization  = |utilized doc sentences| / |doc sentences|
  judge_completeness = |relevant AND utilized| / |relevant|
  judge_adherence    = 1.0 iff every answer sentence is fully supported
"""
import json
import os
import re
import string
import time

from .config import Config


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def _letters(i: int) -> str:
    """0 -> 'a', 25 -> 'z', 26 -> 'aa', ..."""
    out = ""
    while True:
        out = string.ascii_lowercase[i % 26] + out
        i = i // 26 - 1
        if i < 0:
            return out


class LLMJudge:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError(
                "LLM judge needs GROQ_API_KEY (set it in the environment or .env)")
        from groq import Groq
        self._client = Groq()

    # ---- prompt construction -------------------------------------------------

    def _format_documents_with_keys(self, documents: list[str]) -> tuple[str, int]:
        """Sentence-key every document ('0a', '0b', '1a', ...).

        Returns the formatted block and the total number of document
        sentences (the denominator of relevance and utilization).
        """
        lines, total = [], 0
        for d_idx, doc in enumerate(documents):
            for s_idx, sent in enumerate(_split_sentences(doc)):
                lines.append(f"{d_idx}{_letters(s_idx)}. {sent}")
                total += 1
            lines.append("")
        return "\n".join(lines).strip(), total

    def _format_answer_with_keys(self, answer: str) -> str:
        return "\n".join(
            f"{_letters(i)}. {s}" for i, s in enumerate(_split_sentences(answer)))

    def _doc_sentence_keys(self, documents: list[str]) -> set[str]:
        return {
            f"{d_idx}{_letters(s_idx)}"
            for d_idx, doc in enumerate(documents)
            for s_idx in range(len(_split_sentences(doc)))
        }

    def format_ragbench_judgment_prompt(
            self, question: str, documents: list[str], answer: str) -> tuple[str, int]:
        formatted_docs, num_doc_sentences = self._format_documents_with_keys(documents)
        formatted_answer = self._format_answer_with_keys(answer)

        prompt = f"""You are an expert judge evaluating a RAG (Retrieval-Augmented Generation) system.

You will be given a QUESTION, a set of DOCUMENTS, and an ANSWER generated from those documents.
Your task is to carefully evaluate the quality of the answer with respect to the documents.

Each document sentence is labeled with a unique key like '0a', '0b', '1a', etc. (document index + letter).
Each answer sentence is labeled with a letter key like 'a', 'b', 'c', etc.

[QUESTION]
{question}

[DOCUMENTS]
{formatted_docs}

[ANSWER]
{formatted_answer}

Carefully analyze the above and respond ONLY with a valid JSON object with exactly these six fields:

{{
  "relevance_explanation": string,
  "all_relevant_sentence_keys": [string],
  "overall_supported_explanation": string,
  "overall_supported": boolean,
  "sentence_support_information": [
    {{
      "response_sentence_key": string,
      "explanation": string,
      "supporting_sentence_keys": [string],
      "fully_supported": boolean
    }}
  ],
  "all_utilized_sentence_keys": [string]
}}

Definitions:
- "relevance_explanation": A 1-2 sentence justification for which document sentences relate to the question.
- "all_relevant_sentence_keys": Keys of ALL document sentences that contain information relevant to answering the question.
- "overall_supported_explanation": A 1-2 sentence justification for whether the answer is supported by the documents.
- "overall_supported": true if the answer is fully or mostly supported by the documents; false otherwise.
- "sentence_support_information": a list of objects, one for each sentence in the response. Each object MUST have the following fields
  - "response_sentence_key": a string identifying the sentence in the response. This key is the same as the one used in the response above
  - "explanation": a string explaining why the sentence is or is not supported by the documents.
  - "supporting_sentence_keys": Identifies the document sentence key(s) (e.g., "0a") that provide evidence for each response sentence. If a response sentence is unsupported, the list must be empty; otherwise, it must contain one or more supporting sentence keys. In cases where support exists but cannot be attributed to a specific sentence, use "supported_without_sentence" (e.g., when the response correctly states that there is insufficient information in the provided context). For general statements - such as outlining the steps to answer a question, summarizing earlier statements, or serving as a transition - use "general". For universally accepted facts (e.g., a mathematical formula), use "well_known_fact". For sentences involving numerical reasoning (e.g., addition or multiplication), use "numerical_reasoning".
  - "fully_supported": true if the sentence is fully supported by the documents; false otherwise.
    - This value should reflect the conclusion you drew at the end of your step-by-step breakdown in explanation.
    - If supporting_sentence_keys is an empty list, then fully_supported must be false.
    - Otherwise, use fully_supported to clarify whether everything in the response sentence is fully supported by the document text indicated in supporting_sentence_keys (fully_supported = true), or whether the sentence is only partially or incompletely supported by that document text (fully_supported = false).
- "all_utilized_sentence_keys": list of all sentences keys (e.g. '0a') that were used to construct the answer. Include every sentence that either directly supported the answer, or was implicitly used to construct the answer, even if it was not used in its entirety. Omit sentences that were not used, and could have been removed from the documents without affecting the answer.

Important rules:
- Return ONLY the JSON object. No markdown, no triple backticks, no preamble.
- Do not omit any field. Use empty lists [] where applicable, never null.
- Every answer sentence key (a, b, c, ...) MUST appear in sentence_support_information.
- "all_utilized_sentence_keys" must be a subset of "all_relevant_sentence_keys".

Reminder:
- your task is to review the response and assess which documents contain useful information pertaining to the question, and how each sentence in the response is supported by the text in the documents
"""
        return prompt, num_doc_sentences

    # ---- judging -------------------------------------------------------------

    def _call(self, prompt: str) -> str:
        from groq import RateLimitError
        for attempt in range(4):
            try:
                resp = self._client.chat.completions.create(
                    model=self.cfg.judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=4096,
                )
                return resp.choices[0].message.content.strip()
            except RateLimitError:
                if attempt == 3:
                    raise
                time.sleep(10 * (attempt + 1))

    @staticmethod
    def _parse_json(raw: str) -> dict:
        text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE)
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"No JSON object in judge output: {raw[:200]!r}")
        return json.loads(text[start:end + 1])

    def judge(self, question: str, documents: list[str], answer: str) -> dict:
        """Judge one example; returns {"metrics": {...}, "judgment": raw dict}."""
        prompt, num_doc_sentences = self.format_ragbench_judgment_prompt(
            question, documents, answer)
        raw = self._call(prompt)
        try:
            judgment = self._parse_json(raw)
        except (ValueError, json.JSONDecodeError):
            judgment = self._parse_json(self._call(prompt))  # one retry

        valid = self._doc_sentence_keys(documents)
        relevant = {k for k in judgment.get("all_relevant_sentence_keys", []) if k in valid}
        utilized = {k for k in judgment.get("all_utilized_sentence_keys", []) if k in valid}
        support = judgment.get("sentence_support_information", [])

        metrics: dict = {}
        if num_doc_sentences:
            metrics["judge_relevance"] = len(relevant) / num_doc_sentences
            metrics["judge_utilization"] = len(utilized) / num_doc_sentences
        metrics["judge_completeness"] = (
            len(relevant & utilized) / len(relevant) if relevant else None)
        metrics["judge_adherence"] = (
            1.0 if support and all(s.get("fully_supported") for s in support) else 0.0)
        return {"metrics": metrics, "judgment": judgment}
