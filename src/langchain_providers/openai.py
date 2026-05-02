"""OpenAI LLM provider implementation."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from langchain_providers.base import BaseLLMProvider
from log import get_logger
from models.config import LangChainProviderConfig

logger = get_logger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider using LangChain's ChatOpenAI."""

    def __init__(self, config: LangChainProviderConfig):
        """Initialize OpenAI provider.

        Parameters:
            config: OpenAI provider configuration.
        """
        super().__init__("openai", config)

    async def get_chat_model(self, model_id: str, **kwargs: Any) -> BaseChatModel:
        """Get an OpenAI chat model instance.

        Parameters:
            model_id: OpenAI model ID (e.g., "gpt-4", "gpt-3.5-turbo").
            **kwargs: Additional ChatOpenAI parameters (temperature, max_tokens, etc.).

        Returns:
            BaseChatModel: Initialized ChatOpenAI model.

        Raises:
            ValueError: If model_id is not available from this provider.
        """
        if not self.is_model_available(model_id):
            available = ", ".join(self.config.models) if self.config.models else "all"
            raise ValueError(
                f"Model '{model_id}' not available from OpenAI provider. "
                f"Available models: {available}"
            )

        logger.debug("Creating OpenAI chat model: %s", model_id)

        # Build ChatOpenAI parameters
        model_params = {
            "model": model_id,
            "timeout": self.config.timeout,
            "max_retries": self.config.max_retries,
        }

        # Add API key if configured (otherwise uses OPENAI_API_KEY env var)
        if self.config.api_key:
            model_params["api_key"] = self.config.api_key.get_secret_value()

        # Add extra params from config
        model_params.update(self.config.extra_params)

        # Override with any runtime kwargs
        model_params.update(kwargs)

        return ChatOpenAI(**model_params)
