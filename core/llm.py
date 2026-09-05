"""Shared Gemini client and rate-limit-aware invocation helpers."""

from __future__ import annotations

import os
import random
import time
from email.utils import parsedate_to_datetime
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI


DEFAULT_MAX_ATTEMPTS = 6


def get_llm() -> ChatGoogleGenerativeAI:
    """Create the Gemini chat model.

    Calls are retried by ``invoke_with_retry`` so a 429 produces useful output
    and observes a retry hint when the API supplies one.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add your Gemini API key to .env.")

    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
        temperature=0,
        google_api_key=api_key,
    )


def _retry_after_seconds(error: Exception) -> float | None:
    """Read Retry-After from an httpx error, if the API sent one."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {})
    retry_after = headers.get("retry-after") if headers else None
    if not retry_after:
        return None
    try:
        return max(0.0, float(retry_after))
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(retry_after).timestamp() - time.time())
        except (TypeError, ValueError):
            return None


def _is_rate_limit(error: Exception) -> bool:
    response = getattr(error, "response", None)
    message = str(error).lower()
    return (
        getattr(response, "status_code", None) == 429
        or "rate limit" in message
        or "resource_exhausted" in message
        or "429" in message
    )


def invoke_with_retry(chain: Any, payload: Any, *, operation: str) -> Any:
    """Invoke a LangChain runnable with bounded exponential backoff for 429s."""
    try:
        max_attempts = int(os.getenv("GEMINI_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS))
    except ValueError:
        max_attempts = DEFAULT_MAX_ATTEMPTS
    max_attempts = max(1, max_attempts)

    for attempt in range(1, max_attempts + 1):
        try:
            return chain.invoke(payload)
        except Exception as error:
            if not _is_rate_limit(error) or attempt == max_attempts:
                if _is_rate_limit(error):
                    raise RuntimeError(
                        f"Gemini is still rate-limiting {operation} after {max_attempts} attempts. "
                        "Wait for the quota window to reset or check your Gemini API quota."
                    ) from error
                raise

            # Exponential backoff with jitter prevents immediate retry bursts.
            delay = _retry_after_seconds(error)
            if delay is None:
                # Start at two seconds; the six default attempts cover a little
                # over a minute, which accommodates common per-minute limits.
                delay = min(60.0, 2 ** attempt) + random.uniform(0, 1)
            print(
                f"Gemini rate limit reached while {operation}. "
                f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_attempts})..."
            )
            time.sleep(delay)

    raise AssertionError("unreachable")
