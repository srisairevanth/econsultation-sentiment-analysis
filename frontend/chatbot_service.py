"""
Part 6.2 / 6.4 / 6.5 / 6.6: U.S. Financial Regulatory AI Assistant backend.

Architecture (deliberately simple - no RAG/vector DB):
    User Question -> selected regulatory context + system instructions -> Gemini API -> response

Uses Google's Gemini API (free-tier eligible for the default model), via the
official `google-genai` SDK. The API key is read ONLY from the environment
(GEMINI_API_KEY via .env); it is never hardcoded and this module is
backend-only (never import it from client-side/browser code).

Reliability: the free Gemini tier regularly answers 503 "high demand" for a given
model. Each question therefore walks a chain of models (GEMINI_MODEL first, then
GEMINI_FALLBACK_MODELS), retrying server-side/timeout failures once per model, under
an overall time budget so the UI never hangs.
"""
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from context_manager import get_context, suggest_regulation_for_question
from prompts import DISCLAIMER, SYSTEM_INSTRUCTIONS, build_user_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# HTTP statuses worth retrying on the SAME model (transient, server-side).
_RETRY_SAME_MODEL = {500, 502, 503, 504}
# Statuses that mean "this key is bad" - no other model will help, stop immediately.
_FATAL_AUTH = {401, 403}


class AssistantUnavailable(RuntimeError):
    """Raised when no model in the chain produced an answer. `kind` drives the user message."""

    def __init__(self, kind: str, detail: str = ""):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind  # "auth" | "overloaded" | "failed"
        self.detail = detail


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


def _status_code(exc: BaseException) -> Optional[int]:
    """HTTP status carried by a google-genai APIError (None for non-HTTP failures)."""
    code = getattr(exc, "code", None)
    return code if isinstance(code, int) else None


def _call_with_timeout(fn: Callable[[], str], timeout_s: float) -> str:
    """Run fn() but stop waiting after timeout_s (the SDK version pinned here has no timeout option)."""
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(fn).result(timeout=timeout_s)
    finally:
        pool.shutdown(wait=False)


# model name -> monotonic time until which it is tried last. Shared by every chat session in this
# process, so one overloaded question doesn't make every later question wait through it again.
_MODEL_COOLDOWN_UNTIL: Dict[str, float] = {}
_COOLDOWN_S = 60.0


def generate_with_fallback(
    call: Callable[[str], str],
    models: List[str],
    attempt_timeout_s: float = config.GEMINI_ATTEMPT_TIMEOUT_S,
    total_budget_s: float = config.GEMINI_TOTAL_BUDGET_S,
    retries_per_model: int = config.GEMINI_RETRIES_PER_MODEL,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    cooldowns: Optional[Dict[str, float]] = None,
) -> Tuple[str, str]:
    """
    Try each model in order until one returns text. Returns (answer, model_used).

    - 500/502/503/504 or a per-attempt timeout: retry the same model (short backoff), then move on.
    - 429 (quota), 404 (model retired) and other client errors: move straight to the next model.
    - 401/403: the key itself is rejected - raise immediately, no other model can help.
    - A model that exhausted its retries on overload is tried LAST for the next _COOLDOWN_S seconds
      (never dropped, so if everything is down it is still attempted).
    Raises AssistantUnavailable if nothing answers within the overall budget.
    """
    started = clock()
    overloaded = False
    last_detail = ""
    if cooldowns is None:
        cooldowns = {}
    models = sorted(models, key=lambda m: cooldowns.get(m, 0.0) > started)  # stable: healthy first

    for model in models:
        model_overloaded = False
        for attempt in range(retries_per_model + 1):
            remaining = total_budget_s - (clock() - started)
            if remaining <= 0:
                raise AssistantUnavailable("overloaded" if overloaded else "failed", "time budget exhausted")
            try:
                answer = _call_with_timeout(lambda m=model: call(m), min(attempt_timeout_s, remaining))
                if not answer or not answer.strip():
                    raise ValueError("empty response")
                if model != models[0] or attempt:
                    logger.info("Answered by %s (attempt %d)", model, attempt + 1)
                return answer, model
            except FutureTimeout:
                overloaded = model_overloaded = True
                last_detail = f"{model}: timed out"
                logger.warning("Gemini %s timed out (attempt %d)", model, attempt + 1)
            except Exception as exc:  # SDK/network/empty-response errors; classified below
                code = _status_code(exc)
                last_detail = f"{model}: {type(exc).__name__} {code or ''}".strip()
                logger.warning("Gemini %s failed (attempt %d): %s", model, attempt + 1, str(exc)[:200])
                if code in _FATAL_AUTH:
                    raise AssistantUnavailable("auth", last_detail) from exc
                if code in _RETRY_SAME_MODEL:
                    overloaded = model_overloaded = True
                else:
                    break  # 429 / 404 / other 4xx / empty answer: no point retrying this model
            if attempt < retries_per_model:
                sleep(min(2.0 * (attempt + 1), max(0.0, total_budget_s - (clock() - started))))
        if model_overloaded:
            cooldowns[model] = clock() + _COOLDOWN_S

    raise AssistantUnavailable("overloaded" if overloaded else "failed", last_detail)


_UNAVAILABLE_MESSAGES = {
    "auth": "The Gemini API key was rejected. Check GEMINI_API_KEY in this app's secrets.",
    "overloaded": ("Google's Gemini service is overloaded right now (all backup models were busy "
                   "too). This is temporary - please ask again in a minute."),
    "failed": "The AI assistant could not produce an answer. Please try again shortly.",
}


def ask_regulatory_assistant(question: str, regulation_id: Optional[str] = None,
                              conversation: Optional[ConversationManager] = None) -> dict:
    """
    Returns:
        {"answer": str|None, "regulation_used": str|None, "model_used": str|None,
         "disclaimer": str, "error": str|None}
    """
    def _result(answer=None, regulation=None, model=None, error=None):
        return {"answer": answer, "regulation_used": regulation, "model_used": model,
                "disclaimer": DISCLAIMER, "error": error}

    if not question or not question.strip():
        return _result(error="Please enter a question.")

    if regulation_id is None:
        regulation_id = suggest_regulation_for_question(question)

    context_text = get_context(regulation_id) if regulation_id else None
    if not context_text:
        return _result(regulation=regulation_id,
                       error=("No relevant regulatory context is available for this question. "
                              "Add a regulation file under data/regulatory_context/ or select a topic."))

    try:
        client = _get_client()
    except RuntimeError as exc:
        return _result(regulation=regulation_id, error=str(exc))

    from google.genai import types

    contents = []
    if conversation is not None:
        contents.extend(conversation.as_gemini_contents())
    contents.append({"role": "user", "parts": [{"text": build_user_prompt(context_text, question)}]})

    def _call(model_name: str) -> str:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTIONS,
                temperature=0.2,
                max_output_tokens=600,
            ),
        )
        return response.text

    models = [config.GEMINI_MODEL] + [m for m in config.GEMINI_FALLBACK_MODELS if m != config.GEMINI_MODEL]
    try:
        answer, model_used = generate_with_fallback(_call, models, cooldowns=_MODEL_COOLDOWN_UNTIL)
    except AssistantUnavailable as exc:
        logger.error("Gemini unavailable (%s): %s", exc.kind, exc.detail)
        return _result(regulation=regulation_id, error=_UNAVAILABLE_MESSAGES[exc.kind])

    if conversation is not None:
        conversation.add("user", question)
        conversation.add("assistant", answer)

    return _result(answer=answer, regulation=regulation_id, model=model_used)
