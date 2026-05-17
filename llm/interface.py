from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import override


@dataclass
class LLMConfig:
    api_key: str
    model: str = ""
    temperature: float = 0.7
    max_tokens: int | None = None


class LLMProvider(ABC):
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str: ...
