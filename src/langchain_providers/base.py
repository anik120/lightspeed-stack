"""Base provider interface for LangChain LLM providers."""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from models.config import LangChainProviderConfig


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    All provider implementations (OpenAI, Azure, Watsonx) must extend this
    class and implement the get_chat_model() method.
    """

    def __init__(self, provider_name: str, config: LangChainProviderConfig):
        """Initialize the provider.

        Parameters:
            provider_name: Name of the provider (e.g., "openai", "azure").
            config: Provider-specific configuration.
        """
        self.provider_name = provider_name
        self.config = config

    @abstractmethod
    async def get_chat_model(self, model_id: str, **kwargs: Any) -> BaseChatModel:
        """Get a chat model instance for the specified model ID.

        Parameters:
            model_id: Model identifier (e.g., "gpt-4", "gpt-3.5-turbo").
            **kwargs: Additional model-specific parameters.

        Returns:
            BaseChatModel: Initialized LangChain chat model.

        Raises:
            ValueError: If model_id is not available from this provider.
            Exception: If model initialization fails.
        """

    def is_model_available(self, model_id: str) -> bool:
        """Check if a model is available from this provider.

        Parameters:
            model_id: Model identifier to check.

        Returns:
            bool: True if model is available, False otherwise.
        """
        if not self.config.models:
            return True  # If no models list, assume all models are available
        return model_id in self.config.models

    def get_available_models(self) -> list[str]:
        """Get list of available models from this provider.

        Returns:
            list[str]: List of model identifiers.
        """
        return self.config.models.copy()
