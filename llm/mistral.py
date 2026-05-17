from typing import override

from mistralai.client import Mistral

from llm.interface import LLMConfig, LLMProvider


class MistralProvider(LLMProvider):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._client = Mistral(api_key=config.api_key)

    @override
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        model = self.config.model or "mistral-large-latest"
        response = self._client.chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens or 1024,
        )
        return response.choices[0].message.content or ""
