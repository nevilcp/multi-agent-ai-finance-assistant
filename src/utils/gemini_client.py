"""Google AI Studio client with rate limiting and retry logic."""
import asyncio
import json
import random
from typing import Type

from pydantic import BaseModel
from google import genai
from google.genai import types
from google.api_core.exceptions import ServiceUnavailable, ResourceExhausted

from src.config import config
from src.utils.rate_limiter import rate_limiter, ModelType
from src.observability.logger import get_logger

logger = get_logger(__name__)

_MAX_ATTEMPTS = 3

# Model name map keyed by ModelType enum
_MODEL_NAMES = {
    ModelType.FLASH: config.google.flash_model,
    ModelType.FLASH_LITE: config.google.flash_lite_model,
}


def _clean_schema_for_gemini(schema: dict) -> dict:
    """
    Recursively strip keys that Gemini's schema validator rejects.
    Gemini does not support: additionalProperties, $defs, $schema, title (on nested objects).
    Inline any $ref references by resolving against $defs.
    """
    import copy
    schema = copy.deepcopy(schema)

    defs = schema.pop("$defs", {})

    def resolve(node: dict) -> dict:
        if "$ref" in node:
            ref_name = node["$ref"].split("/")[-1]
            node = copy.deepcopy(defs.get(ref_name, node))
        node.pop("additionalProperties", None)
        node.pop("$schema", None)
        node.pop("title", None)
        for key, val in node.items():
            if isinstance(val, dict):
                node[key] = resolve(val)
            elif isinstance(val, list):
                node[key] = [resolve(i) if isinstance(i, dict) else i for i in val]
        return node

    return resolve(schema)


class GeminiClient:
    """
    Thin wrapper over google.genai with built-in rate limiting.

    Uses the new centralized Client object (google-genai >= 1.0.0).
    Every call acquires a rate-limiter slot before hitting the API.
    Retries on transient 503 / 429-RPM errors with exponential backoff + jitter.
    """

    def __init__(self) -> None:
        self._client = genai.Client(api_key=config.google.api_key)

    async def _generate_with_retry(
        self,
        model_name: str,
        contents: str,
        config_obj: types.GenerateContentConfig,
    ):
        """
        Call generate_content with retry on transient errors.

        Retries up to _MAX_ATTEMPTS total (1 initial + 2 retries).
        Retries only on ServiceUnavailable (503) and ResourceExhausted (429)
        when the 429 is an RPM limit. Daily-quota (RPD) 429s raise immediately.
        """
        last_exc: Exception | None = None

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                return await self._client.aio.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config_obj,
                )
            except (ServiceUnavailable, ResourceExhausted) as exc:
                last_exc = exc

                # For 429s, distinguish RPM (retryable) from RPD (fatal).
                if isinstance(exc, ResourceExhausted):
                    msg = str(exc).lower()
                    if "per day" in msg or "per_day" in msg or "daily" in msg:
                        raise RuntimeError("Daily quota exhausted") from exc

                # If this was the last attempt, don't sleep - fall through to raise.
                if attempt == _MAX_ATTEMPTS:
                    break

                wait_seconds = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    "gemini_retry",
                    attempt=attempt,
                    max_attempts=_MAX_ATTEMPTS,
                    exception_type=type(exc).__name__,
                    wait_seconds=round(wait_seconds, 2),
                )
                await asyncio.sleep(wait_seconds)

        # All attempts exhausted — propagate the original exception.
        raise last_exc  # type: ignore[misc]

    async def generate(
        self,
        prompt: str,
        model: ModelType = ModelType.FLASH_LITE,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a text response."""
        await rate_limiter.acquire(model)

        model_name = _MODEL_NAMES[model]
        logger.info("gemini_request", model=model_name, prompt_len=len(prompt))

        response = await self._generate_with_retry(
            model_name=model_name,
            contents=prompt,
            config_obj=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )

        text = response.text
        logger.info("gemini_response", model=model_name, response_len=len(text))
        return text

    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        model: ModelType = ModelType.FLASH_LITE,
        temperature: float = 0.3,
        max_tokens: int = 200000,
    ) -> BaseModel:
        """Generate a response and parse it into a Pydantic model.

        Uses Gemini's native JSON mode (response_mime_type + response_schema)
        to guarantee schema-conformant output without prompt-level injection.
        """
        # Extract JSON schema upfront; fail fast on malformed models.
        try:
            schema = _clean_schema_for_gemini(response_model.model_json_schema())
        except Exception as exc:
            raise ValueError(
                f"Cannot extract JSON schema from {response_model.__name__}: {exc}"
            ) from exc

        await rate_limiter.acquire(model)

        model_name = _MODEL_NAMES[model]
        logger.info(
            "gemini_structured_request",
            model=model_name,
            response_model=response_model.__name__,
            prompt_len=len(prompt),
        )

        try:
            response = await self._generate_with_retry(
                model_name=model_name,
                contents=prompt,
                config_obj=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
        except (TypeError, ValueError) as exc:
            # SDK rejected the schema or config — surface clearly.
            raise ValueError(
                f"Gemini SDK rejected schema for {response_model.__name__}: {exc}"
            ) from exc

        data = json.loads(response.text)
        return response_model.model_validate(data)


# Global instance
gemini_client = GeminiClient()
