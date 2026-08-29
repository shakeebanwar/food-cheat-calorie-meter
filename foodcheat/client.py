"""OpenRouter chat client with retries and usage capture."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from foodcheat.config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    MAX_RETRIES,
    OPENROUTER_API_KEY,
    OPENROUTER_CHAT_URL,
    REQUEST_TIMEOUT_S,
    RETRY_BACKOFF_S,
    estimated_cost,
)
from foodcheat.schema import UsageStats, food_analysis_json_schema


class OpenRouterError(Exception):
    def __init__(self, message: str, status: int | None = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


class OpenRouterClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = OPENROUTER_CHAT_URL,
        timeout: float = REQUEST_TIMEOUT_S,
    ):
        self.api_key = api_key or OPENROUTER_API_KEY
        if not self.api_key:
            raise OpenRouterError("OPENROUTER_API_KEY missing in environment / .env")
        self.base_url = base_url
        self.timeout = timeout
        self._schema = food_analysis_json_schema()

    def chat(
        self,
        model_id: str,
        messages: list[dict],
        *,
        model_key: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        use_structured: bool = True,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call OpenRouter; returns {content, usage, latency_ms, attempts, provider, raw}."""
        body: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if use_structured:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "FoodAnalysis",
                    "strict": True,
                    "schema": self._schema,
                },
            }
        if extra:
            body.update(extra)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://foodcheat.local",
            "X-Title": "FoodCheat AI Vision R&D",
        }

        last_err: Exception | None = None
        attempts = 0
        t0 = time.perf_counter()

        with httpx.Client(timeout=self.timeout) as client:
            for attempt in range(1, MAX_RETRIES + 1):
                attempts = attempt
                try:
                    resp = client.post(self.base_url, headers=headers, json=body)
                    if resp.status_code in (429, 500, 502, 503, 504):
                        raise OpenRouterError(
                            f"Transient HTTP {resp.status_code}",
                            status=resp.status_code,
                            body=resp.text[:800],
                        )
                    if resp.status_code >= 400:
                        # Structured-output may be unsupported — caller can retry plain
                        raise OpenRouterError(
                            f"HTTP {resp.status_code}: {resp.text[:500]}",
                            status=resp.status_code,
                            body=resp.text[:2000],
                        )
                    data = resp.json()
                    if "error" in data:
                        raise OpenRouterError(
                            str(data["error"]),
                            status=resp.status_code,
                            body=json.dumps(data)[:800],
                        )
                    choice = (data.get("choices") or [{}])[0]
                    message = choice.get("message") or {}
                    content = message.get("content") or ""
                    # Some models put JSON in refusal / reasoning fields — keep content
                    usage_raw = data.get("usage") or {}
                    usage = self._parse_usage(usage_raw, model_key=model_key)
                    latency_ms = (time.perf_counter() - t0) * 1000
                    return {
                        "content": content,
                        "usage": usage,
                        "latency_ms": latency_ms,
                        "attempts": attempts,
                        "provider": data.get("provider"),
                        "raw": data,
                        "finish_reason": choice.get("finish_reason"),
                    }
                except (httpx.TimeoutException, httpx.TransportError, OpenRouterError) as e:
                    last_err = e
                    # Don't retry non-transient 4xx except 429
                    if isinstance(e, OpenRouterError) and e.status and e.status < 500 and e.status != 429:
                        if e.status != 400:  # 400 might be schema — let caller handle
                            break
                        # For 400 (often schema), break so caller can fall back
                        break
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_BACKOFF_S * attempt)

        latency_ms = (time.perf_counter() - t0) * 1000
        raise OpenRouterError(
            f"Failed after {attempts} attempts: {last_err}",
            status=getattr(last_err, "status", None),
            body=getattr(last_err, "body", ""),
        )

    @staticmethod
    def _parse_usage(usage_raw: dict, model_key: str | None = None) -> UsageStats:
        prompt = int(usage_raw.get("prompt_tokens") or 0)
        completion = int(usage_raw.get("completion_tokens") or 0)
        total = int(usage_raw.get("total_tokens") or (prompt + completion))
        details = usage_raw.get("completion_tokens_details") or {}
        reasoning = int(details.get("reasoning_tokens") or 0)

        cost = usage_raw.get("cost")
        cost_estimated = False
        if cost is None:
            cost_details = usage_raw.get("cost_details") or {}
            cost = cost_details.get("upstream_inference_cost")
        if cost is None and model_key:
            cost = estimated_cost(prompt, completion, model_key)
            cost_estimated = True
        return UsageStats(
            prompt_tokens=prompt,
            completion_tokens=completion,
            reasoning_tokens=reasoning,
            total_tokens=total,
            cost_usd=float(cost or 0.0),
            cost_estimated=cost_estimated,
        )
