import json
import re
from typing import override

from openai import OpenAI

from llm.interface import LLMConfig, LLMProvider


class GroqProvider(LLMProvider):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._client = OpenAI(
            api_key=config.api_key,
            base_url="https://api.groq.com/openai/v1",
        )

    @override
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        model = self.config.model or "llama-3.3-70b-versatile"

        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens or 1024,
        }

        response = self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""

        return self._clean_response(content)

    @staticmethod
    def _clean_response(content: str) -> str:
        stripped = content.strip()

        if match := re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped):
            return match.group(1).strip()

        return stripped
