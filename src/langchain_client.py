"""LangChain LLM provider registry and client management.

Manages initialization and retrieval of LangChain LLM providers, supporting
multiple provider types (OpenAI, Azure, Watsonx) with lazy initialization.
"""

from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from log import get_logger
from models.config import LangChainConfiguration, LangChainProviderConfig
from utils.types import Singleton

logger = get_logger(__name__)


class LLMProviderRegistry(metaclass=Singleton):
    """Registry for managing LangChain LLM providers.

    Provides centralized management of LLM provider instances with lazy
    initialization and caching. Supports multiple provider types and
    configurations.

    Usage:
        registry = LLMProviderRegistry()
        await registry.initialize(config)
        provider = await registry.get_provider("azure/gpt-4")
    """

    _config: Optional[LangChainConfiguration] = None
    _providers: dict[str, BaseChatModel] = {}
    _initialized: bool = False

    async def initialize(self, config: LangChainConfiguration) -> None:
        """Initialize the provider registry with configuration.

        Parameters:
        ----------
            config: LangChain configuration with provider settings.

        Raises:
        ------
            ValueError: If configuration is invalid.
        """
        if self._initialized:
            logger.info("LLM provider registry already initialized")
            return

        self._config = config
        self._providers = {}
        self._initialized = True

        logger.info(
            "Initialized LLM provider registry with providers: %s",
            list(config.providers.keys()),
        )

    async def get_provider(
        self,
        model_id: Optional[str] = None,
        provider_name: Optional[str] = None,
    ) -> BaseChatModel:
        """Get or create an LLM provider instance.

        Supports two formats:
        - model_id="azure/gpt-4" (provider/model format)
        - provider_name="azure", model_id="gpt-4" (separate args)

        Parameters:
        ----------
            model_id: Model identifier, optionally with provider prefix (provider/model).
            provider_name: Explicit provider name (overrides model_id prefix).

        Returns:
        -------
            BaseChatModel: Initialized LangChain chat model instance.

        Raises:
        ------
            RuntimeError: If registry not initialized.
            ValueError: If provider not found or configuration invalid.
        """
        if not self._initialized or self._config is None:
            raise RuntimeError(
                "LLM provider registry not initialized. Call initialize() first."
            )

        # Parse provider and model from model_id
        if model_id and "/" in model_id and not provider_name:
            provider_name, model_id = model_id.split("/", 1)

        # Use defaults if not specified
        if not provider_name:
            provider_name = self._config.default_provider
        if not model_id:
            model_id = self._config.default_model

        # Check cache
        cache_key = f"{provider_name}/{model_id}"
        if cache_key in self._providers:
            logger.debug("Returning cached provider: %s", cache_key)
            return self._providers[cache_key]

        # Get provider config
        if provider_name not in self._config.providers:
            raise ValueError(
                f"Provider '{provider_name}' not found in configuration. "
                f"Available providers: {list(self._config.providers.keys())}"
            )

        provider_config = self._config.providers[provider_name]

        # Create and cache provider
        provider = self._create_provider(provider_name, model_id, provider_config)
        self._providers[cache_key] = provider

        logger.info("Created LLM provider: %s", cache_key)
        return provider

    def _create_provider(
        self,
        provider_name: str,
        model_id: str,
        config: LangChainProviderConfig,
    ) -> BaseChatModel:
        """Create a new LLM provider instance.

        Parameters:
        ----------
            provider_name: Name of the provider (openai, azure, watsonx).
            model_id: Model identifier.
            config: Provider-specific configuration.

        Returns:
        -------
            BaseChatModel: Initialized provider instance.

        Raises:
        ------
            ValueError: If provider type is not supported.
        """
        api_key = config.api_key.get_secret_value() if config.api_key else None

        if provider_name == "openai":
            return ChatOpenAI(
                model=model_id,
                api_key=api_key,
                timeout=config.timeout,
                max_retries=config.max_retries,
                **config.extra_params,
            )

        elif provider_name == "azure":
            if not config.api_base:
                raise ValueError("Azure provider requires api_base configuration")
            if not config.api_version:
                raise ValueError("Azure provider requires api_version configuration")

            return AzureChatOpenAI(
                model=model_id,
                api_key=api_key,
                azure_endpoint=config.api_base,
                api_version=config.api_version,
                timeout=config.timeout,
                max_retries=config.max_retries,
                **config.extra_params,
            )

        elif provider_name == "watsonx":
            # Import here to avoid dependency issues if watsonx not installed
            try:
                from langchain_community.chat_models import ChatWatsonx

                return ChatWatsonx(
                    model=model_id,
                    api_key=api_key,
                    url=config.api_base,
                    timeout=config.timeout,
                    **config.extra_params,
                )
            except ImportError as e:
                raise ValueError(
                    "Watsonx provider requires langchain-community with watsonx support"
                ) from e

        else:
            raise ValueError(
                f"Unsupported provider: {provider_name}. "
                f"Supported providers: openai, azure, watsonx"
            )

    def supports_streaming(self, model_id: str) -> bool:
        """Check if a model supports streaming responses.

        Parameters:
        ----------
            model_id: Model identifier to check.

        Returns:
        -------
            bool: True if model supports streaming, False otherwise.
        """
        # Most modern LLM providers support streaming
        # This can be overridden with provider-specific logic if needed
        if self._config:
            return self._config.enable_streaming
        return True

    def list_providers(self) -> list[str]:
        """List all configured provider names.

        Returns:
        -------
            list[str]: Names of all configured providers.

        Raises:
        ------
            RuntimeError: If registry not initialized.
        """
        if not self._initialized or self._config is None:
            raise RuntimeError(
                "LLM provider registry not initialized. Call initialize() first."
            )

        return list(self._config.providers.keys())

    def list_models(self, provider_name: Optional[str] = None) -> list[str]:
        """List all models for a provider or all providers.

        Parameters:
        ----------
            provider_name: Optional provider name to filter models.

        Returns:
        -------
            list[str]: List of model identifiers in "provider/model" format.

        Raises:
        ------
            RuntimeError: If registry not initialized.
            ValueError: If specified provider not found.
        """
        if not self._initialized or self._config is None:
            raise RuntimeError(
                "LLM provider registry not initialized. Call initialize() first."
            )

        models = []

        if provider_name:
            if provider_name not in self._config.providers:
                raise ValueError(
                    f"Provider '{provider_name}' not found in configuration"
                )
            provider_config = self._config.providers[provider_name]
            models = [f"{provider_name}/{model}" for model in provider_config.models]
        else:
            for prov_name, prov_config in self._config.providers.items():
                models.extend([f"{prov_name}/{model}" for model in prov_config.models])

        return models

    # Helper method for /models endpoint to get models grouped by provider
    # The /models endpoint needs to iterate over providers and their models separately
    # to build the response format, so this returns dict[provider -> list[models]]
    # instead of the flat list that list_models() returns
    async def list_models_by_provider(self) -> dict[str, list[str]]:
        """List all models grouped by provider.

        Returns:
        -------
            dict[str, list[str]]: Mapping from provider name to list of model IDs.

        Raises:
        ------
            RuntimeError: If registry not initialized.
        """
        if not self._initialized or self._config is None:
            raise RuntimeError(
                "LLM provider registry not initialized. Call initialize() first."
            )

        models_by_provider: dict[str, list[str]] = {}
        for prov_name, prov_config in self._config.providers.items():
            models_by_provider[prov_name] = prov_config.models

        return models_by_provider

    def clear_cache(self) -> None:
        """Clear the provider cache, forcing re-initialization on next access."""
        self._providers = {}
        logger.info("Cleared LLM provider cache")
