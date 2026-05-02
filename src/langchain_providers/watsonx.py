"""IBM Watsonx LLM provider implementation."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ibm import ChatWatsonx

from langchain_providers.base import BaseLLMProvider
from log import get_logger
from models.config import LangChainProviderConfig

logger = get_logger(__name__)


class WatsonxProvider(BaseLLMProvider):
    """IBM Watsonx LLM provider using LangChain's ChatWatsonx."""

    def __init__(self, config: LangChainProviderConfig):
        """Initialize Watsonx provider.

        Parameters:
            config: Watsonx provider configuration.
        """
        super().__init__("watsonx", config)

        # Validate required Watsonx-specific config
        if not config.api_base:
            raise ValueError("Watsonx provider requires api_base (Watsonx URL)")

    async def get_chat_model(self, model_id: str, **kwargs: Any) -> BaseChatModel:
        """Get a Watsonx chat model instance.

        Parameters:
            model_id: Watsonx model ID (e.g., "ibm/granite-13b-chat-v2").
            **kwargs: Additional ChatWatsonx parameters.

        Returns:
            BaseChatModel: Initialized ChatWatsonx model.

        Raises:
            ValueError: If model_id is not available from this provider.
        """
        if not self.is_model_available(model_id):
            available = ", ".join(self.config.models) if self.config.models else "all"
            raise ValueError(
                f"Model '{model_id}' not available from Watsonx provider. "
                f"Available models: {available}"
            )

        logger.debug("Creating Watsonx chat model: %s", model_id)

        # Build ChatWatsonx parameters
        model_params = {
            "model_id": model_id,
            "url": self.config.api_base,
            "timeout": self.config.timeout,
            "max_retries": self.config.max_retries,
        }

        # Add API key if configured (otherwise uses WATSONX_APIKEY env var)
        if self.config.api_key:
            model_params["apikey"] = self.config.api_key.get_secret_value()

        # Add extra params from config
        model_params.update(self.config.extra_params)

        # Override with any runtime kwargs
        model_params.update(kwargs)

        return ChatWatsonx(**model_params)
