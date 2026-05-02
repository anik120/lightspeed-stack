"""LangChain LLM provider registry.

Manages multiple LLM providers (OpenAI, Azure, Watsonx) and provides
a unified interface for getting chat models.
"""

from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from langchain_providers.azure import AzureProvider
from langchain_providers.base import BaseLLMProvider
from langchain_providers.openai import OpenAIProvider
from langchain_providers.watsonx import WatsonxProvider
from log import get_logger
from models.config import LangChainConfiguration

logger = get_logger(__name__)


class LLMProviderRegistry:
    """Registry for managing LangChain LLM providers.

    Provides:
    - Provider initialization from configuration
    - Unified interface for getting chat models
    - Provider discovery and validation
    """

    def __init__(self):
        """Initialize empty registry."""
        self._providers: dict[str, BaseLLMProvider] = {}
        self._config: Optional[LangChainConfiguration] = None
        self._initialized = False

    async def initialize(self, config: LangChainConfiguration) -> None:
        """Initialize registry with providers from configuration.

        Parameters:
            config: LangChain configuration with provider definitions.

        Raises:
            ValueError: If configuration is invalid or provider initialization fails.
        """
        if self._initialized:
            logger.warning("Registry already initialized, reinitializing...")
            self._providers.clear()

        self._config = config
        logger.info("Initializing LLM provider registry with %d providers", len(config.providers))

        # Initialize each configured provider
        for provider_name, provider_config in config.providers.items():
            try:
                provider = self._create_provider(provider_name, provider_config)
                self._providers[provider_name] = provider
                logger.info(
                    "Initialized provider '%s' with models: %s",
                    provider_name,
                    provider_config.models or "all",
                )
            except Exception as e:
                logger.error("Failed to initialize provider '%s': %s", provider_name, e)
                raise ValueError(f"Failed to initialize provider '{provider_name}': {e}") from e

        # Validate default provider exists
        if config.default_provider not in self._providers:
            available = ", ".join(self._providers.keys())
            raise ValueError(
                f"Default provider '{config.default_provider}' not found. "
                f"Available providers: {available}"
            )

        self._initialized = True
        logger.info("Provider registry initialized successfully")

    def _create_provider(self, name: str, config: Any) -> BaseLLMProvider:
        """Create a provider instance based on provider name.

        Parameters:
            name: Provider name (openai, azure, watsonx).
            config: Provider configuration.

        Returns:
            BaseLLMProvider: Initialized provider instance.

        Raises:
            ValueError: If provider name is not recognized.
        """
        provider_classes = {
            "openai": OpenAIProvider,
            "azure": AzureProvider,
            "watsonx": WatsonxProvider,
        }

        provider_class = provider_classes.get(name.lower())
        if not provider_class:
            available = ", ".join(provider_classes.keys())
            raise ValueError(
                f"Unknown provider '{name}'. Supported providers: {available}"
            )

        return provider_class(config)

    async def get_chat_model(
        self,
        model_id: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Get a chat model instance.

        Parameters:
            model_id: Model identifier. If None, uses default from config.
            provider: Provider name. If None, uses default from config.
            **kwargs: Additional model-specific parameters.

        Returns:
            BaseChatModel: Initialized chat model.

        Raises:
            RuntimeError: If registry not initialized.
            ValueError: If provider or model not found.
        """
        if not self._initialized or not self._config:
            raise RuntimeError("Registry not initialized. Call initialize() first.")

        # Use defaults if not specified
        provider_name = provider or self._config.default_provider
        model_identifier = model_id or self._config.default_model

        logger.debug(
            "Getting chat model: provider=%s, model=%s", provider_name, model_identifier
        )

        # Get provider instance
        provider_instance = self._providers.get(provider_name)
        if not provider_instance:
            available = ", ".join(self._providers.keys())
            raise ValueError(
                f"Provider '{provider_name}' not found. Available: {available}"
            )

        # Get chat model from provider
        return await provider_instance.get_chat_model(model_identifier, **kwargs)

    def get_provider(self, provider_name: str) -> BaseLLMProvider:
        """Get a provider instance by name.

        Parameters:
            provider_name: Name of the provider.

        Returns:
            BaseLLMProvider: Provider instance.

        Raises:
            RuntimeError: If registry not initialized.
            ValueError: If provider not found.
        """
        if not self._initialized:
            raise RuntimeError("Registry not initialized. Call initialize() first.")

        provider = self._providers.get(provider_name)
        if not provider:
            available = ", ".join(self._providers.keys())
            raise ValueError(
                f"Provider '{provider_name}' not found. Available: {available}"
            )

        return provider

    def list_providers(self) -> list[str]:
        """Get list of available provider names.

        Returns:
            list[str]: List of provider names.
        """
        return list(self._providers.keys())

    def list_models(self, provider: Optional[str] = None) -> list[str]:
        """Get list of available models.

        Parameters:
            provider: Provider name. If None, returns models from all providers.

        Returns:
            list[str]: List of model identifiers.

        Raises:
            RuntimeError: If registry not initialized.
            ValueError: If provider not found.
        """
        if not self._initialized:
            raise RuntimeError("Registry not initialized. Call initialize() first.")

        if provider:
            provider_instance = self.get_provider(provider)
            return provider_instance.get_available_models()

        # Return all models from all providers
        all_models = []
        for provider_instance in self._providers.values():
            all_models.extend(provider_instance.get_available_models())
        return all_models

    @property
    def is_initialized(self) -> bool:
        """Check if registry is initialized.

        Returns:
            bool: True if initialized, False otherwise.
        """
        return self._initialized
