from typing import override

from anthropic import Anthropic

from llm.interface import LLMConfig, LLMProvider


class ClaudeProvider(LLMProvider):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._client = Anthropic(api_key=config.api_key)

    @override
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        model = self.config.model or "claude-sonnet-4-20250514"
        response = self._client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens or 1024,
        )
        return response.content[0].text
