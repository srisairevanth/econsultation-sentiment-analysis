"""
Part 6.2 / 6.4 / 6.5 / 6.6: U.S. Financial Regulatory AI Assistant backend.

Architecture (deliberately simple - no RAG/vector DB):
    User Question -> selected regulatory context + system instructions -> Gemini API -> response

Uses Google's Gemini API (free-tier eligible for the default model), via the
official `google-genai` SDK. The API key is read ONLY from the environment
(GEMINI_API_KEY via .env); it is never hardcoded and this module is
backend-only (never import it from client-side/browser code).
"""
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from context_manager import get_context, suggest_regulation_for_question
from prompts import DISCLAIMER, SYSTEM_INSTRUCTIONS, build_user_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ConversationManager:
    """Keeps a bounded in-session chat history (not persisted to disk)."""

    def __init__(self, max_messages: int = config.MAX_CHAT_HISTORY_MESSAGES):
        self.max_messages = max_messages
        self.history: List[Tuple[str, str]] = []  # (role, content) with role in {"user", "assistant"}

    def add(self, role: str, content: str) -> None:
        self.history.append((role, content))
        # keep only the most recent N messages so the context window stays bounded
        if len(self.history) > self.max_messages:
            self.history = self.history[-self.max_messages:]

    def clear(self) -> None:
        self.history = []

    def as_gemini_contents(self) -> List[dict]:
        """Gemini uses role='model' (not 'assistant') and a parts=[{'text': ...}] shape."""
        role_map = {"user": "user", "assistant": "model"}
        return [{"role": role_map.get(role, role), "parts": [{"text": content}]}
                for role, content in self.history]


def _get_client():
    if not config.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and set your free key from "
            "https://aistudio.google.com/apikey (never hardcode it in source)."
        )
    from google import genai
    return genai.Client(api_key=config.GEMINI_API_KEY)


def ask_regulatory_assistant(question: str, regulation_id: Optional[str] = None,
                              conversation: Optional[ConversationManager] = None) -> dict:
    """
    Returns:
        {"answer": str|None, "regulation_used": str|None, "disclaimer": str, "error": str|None}
    """
    if not question or not question.strip():
        return {"answer": None, "regulation_used": None, "disclaimer": DISCLAIMER,
                "error": "Please enter a question."}

    if regulation_id is None:
        regulation_id = suggest_regulation_for_question(question)

    context_text = get_context(regulation_id) if regulation_id else None
    if not context_text:
        return {"answer": None, "regulation_used": regulation_id, "disclaimer": DISCLAIMER,
                "error": ("No relevant regulatory context is available for this question. "
                          "Add a regulation file under data/regulatory_context/ or select a topic.")}

    try:
        client = _get_client()
    except RuntimeError as exc:
        return {"answer": None, "regulation_used": regulation_id, "disclaimer": DISCLAIMER, "error": str(exc)}

    from google.genai import types

    contents = []
    if conversation is not None:
        contents.extend(conversation.as_gemini_contents())
    user_message = build_user_prompt(context_text, question)
    contents.append({"role": "user", "parts": [{"text": user_message}]})

    try:
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTIONS,
                temperature=0.2,
                max_output_tokens=600,
            ),
        )
        answer = response.text
        if not answer:
            raise ValueError("Empty response from the Gemini API.")
    except Exception as exc:  # covers rate limits, auth errors, malformed responses, network errors
        logger.error("Gemini API call failed: %s", exc)
        return {"answer": None, "regulation_used": regulation_id, "disclaimer": DISCLAIMER,
                "error": f"The AI assistant is temporarily unavailable ({type(exc).__name__}). "
                         "Please try again shortly."}

    if conversation is not None:
        conversation.add("user", question)
        conversation.add("assistant", answer)

    return {"answer": answer, "regulation_used": regulation_id, "disclaimer": DISCLAIMER, "error": None}
