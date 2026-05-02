"""Azure OpenAI LLM provider implementation."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI

from langchain_providers.base import BaseLLMProvider
from log import get_logger
from models.config import LangChainProviderConfig

logger = get_logger(__name__)


class AzureProvider(BaseLLMProvider):
    """Azure OpenAI LLM provider using LangChain's AzureChatOpenAI."""

    def __init__(self, config: LangChainProviderConfig):
        """Initialize Azure provider.

        Parameters:
            config: Azure provider configuration.
        """
        super().__init__("azure", config)

        # Validate required Azure-specific config
        if not config.api_base:
            raise ValueError("Azure provider requires api_base (Azure endpoint URL)")

    async def get_chat_model(self, model_id: str, **kwargs: Any) -> BaseChatModel:
        """Get an Azure OpenAI chat model instance.

        For Azure, model_id can be either:
        - The deployment name (if models config maps deployment names)
        - The model name (gpt-4, gpt-35-turbo, etc.)

        Parameters:
            model_id: Azure deployment name or model ID.
            **kwargs: Additional AzureChatOpenAI parameters.

        Returns:
            BaseChatModel: Initialized AzureChatOpenAI model.

        Raises:
            ValueError: If model_id is not available from this provider.
        """
        if not self.is_model_available(model_id):
            available = ", ".join(self.config.models) if self.config.models else "all"
            raise ValueError(
                f"Model '{model_id}' not available from Azure provider. "
                f"Available models: {available}"
            )

        logger.debug("Creating Azure chat model: %s", model_id)

        # Build AzureChatOpenAI parameters
        model_params = {
            "deployment_name": model_id,  # In Azure, this is the deployment name
            "azure_endpoint": self.config.api_base,
            "timeout": self.config.timeout,
            "max_retries": self.config.max_retries,
        }

        # Add API key if configured (otherwise uses AZURE_OPENAI_API_KEY env var)
        if self.config.api_key:
            model_params["api_key"] = self.config.api_key.get_secret_value()

        # Add API version if configured (otherwise uses default)
        if self.config.api_version:
            model_params["api_version"] = self.config.api_version

        # Add extra params from config
        model_params.update(self.config.extra_params)

        # Override with any runtime kwargs
        model_params.update(kwargs)

        return AzureChatOpenAI(**model_params)
