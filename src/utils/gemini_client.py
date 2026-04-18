"""Google AI Studio client with rate limiting."""
import json
from typing import Type
from pydantic import BaseModel
from google import genai
from google.genai import types

from src.config import config
from src.utils.rate_limiter import rate_limiter, ModelType
from src.observability.logger import get_logger

logger = get_logger(__name__)

# Model name map keyed by ModelType enum
_MODEL_NAMES = {
    ModelType.FLASH: config.google.flash_model,
    ModelType.FLASH_LITE: config.google.flash_lite_model,
}


class GeminiClient:
    """
    Thin wrapper over google.genai with built-in rate limiting.

    Uses the new centralized Client object (google-genai >= 1.0.0).
    Every call acquires a rate-limiter slot before hitting the API.
    """

    def __init__(self) -> None:
        self._client = genai.Client(api_key=config.google.api_key)

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

        response = await self._client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
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
    ) -> BaseModel:
        """Generate a response and parse it into a Pydantic model."""
        schema_prompt = (
            f"{prompt}\n\nRespond with valid JSON matching this schema:\n"
            f"{json.dumps(response_model.model_json_schema(), indent=2)}"
        )

        raw = await self.generate(
            prompt=schema_prompt,
            model=model,
            temperature=temperature,
        )

        # Strip markdown code fences if the model wraps output in ```json ... ```
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            cleaned = cleaned.rsplit("```", 1)[0]

        data = json.loads(cleaned)
        return response_model.model_validate(data)


# Global instance
gemini_client = GeminiClient()
