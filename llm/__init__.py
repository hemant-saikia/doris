from llm.interface import LLMProvider, LLMConfig
from llm.claude import ClaudeProvider
from llm.groq import GroqProvider
from llm.openrouter import OpenRouterProvider
from llm.mistral import MistralProvider


def create_llm(provider: str, config: LLMConfig | None = None) -> LLMProvider:
    registry = {
        "claude": ClaudeProvider,
        "groq": GroqProvider,
        "openrouter": OpenRouterProvider,
        "mistral": MistralProvider,
    }

    if provider not in registry:
        raise ValueError(
            f"Unknown provider '{provider}'. Choose from: {list(registry.keys())}"
        )

    if config is None:
        config = LLMConfig(api_key="", model="")

    return registry[provider](config)


__all__ = ["LLMProvider", "LLMConfig", "create_llm"]
