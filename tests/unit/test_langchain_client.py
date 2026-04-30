"""Unit tests for LangChain client and configuration."""

import pytest
from pydantic import SecretStr, ValidationError

from langchain_client import LLMProviderRegistry
from models.config import LangChainConfiguration, LangChainProviderConfig


class TestLangChainProviderConfig:
    """Test LangChain provider configuration."""

    def test_provider_config_defaults(self) -> None:
        """Test provider config with default values."""
        config = LangChainProviderConfig()

        assert config.api_key is None
        assert config.api_base is None
        assert config.api_version is None
        assert config.timeout == 60
        assert config.max_retries == 3
        assert config.models == []
        assert config.extra_params == {}

    def test_provider_config_with_values(self) -> None:
        """Test provider config with explicit values."""
        config = LangChainProviderConfig(
            api_key=SecretStr("test-key"),
            api_base="https://api.example.com",
            api_version="2024-01-01",
            timeout=120,
            max_retries=5,
            models=["gpt-4", "gpt-3.5-turbo"],
            extra_params={"temperature": 0.7},
        )

        assert config.api_key.get_secret_value() == "test-key"
        assert config.api_base == "https://api.example.com"
        assert config.api_version == "2024-01-01"
        assert config.timeout == 120
        assert config.max_retries == 5
        assert config.models == ["gpt-4", "gpt-3.5-turbo"]
        assert config.extra_params == {"temperature": 0.7}

    def test_provider_config_invalid_timeout(self) -> None:
        """Test provider config with invalid timeout."""
        with pytest.raises(ValidationError):
            LangChainProviderConfig(timeout=0)

        with pytest.raises(ValidationError):
            LangChainProviderConfig(timeout=601)

    def test_provider_config_invalid_retries(self) -> None:
        """Test provider config with invalid max_retries."""
        with pytest.raises(ValidationError):
            LangChainProviderConfig(max_retries=-1)

        with pytest.raises(ValidationError):
            LangChainProviderConfig(max_retries=11)


class TestLangChainConfiguration:
    """Test LangChain configuration."""

    def test_configuration_defaults(self) -> None:
        """Test configuration with default values."""
        config = LangChainConfiguration()

        assert config.providers == {}
        assert config.default_provider == "openai"
        assert config.default_model == "gpt-4"
        assert config.enable_streaming is True
        assert config.enable_tracing is False

    def test_configuration_with_providers(self) -> None:
        """Test configuration with multiple providers."""
        config = LangChainConfiguration(
            providers={
                "openai": LangChainProviderConfig(models=["gpt-4", "gpt-3.5-turbo"]),
                "azure": LangChainProviderConfig(
                    api_base="https://example.openai.azure.com",
                    api_version="2024-01-01",
                    models=["gpt-4"],
                ),
            },
            default_provider="azure",
            default_model="gpt-4",
        )

        assert len(config.providers) == 2
        assert "openai" in config.providers
        assert "azure" in config.providers
        assert config.default_provider == "azure"
        assert config.default_model == "gpt-4"


@pytest.mark.asyncio
class TestLLMProviderRegistry:
    """Test LLM provider registry."""

    async def test_registry_singleton(self) -> None:
        """Test that registry is a singleton."""
        registry1 = LLMProviderRegistry()
        registry2 = LLMProviderRegistry()

        assert registry1 is registry2

    async def test_registry_not_initialized(self) -> None:
        """Test accessing registry before initialization."""
        # Clear singleton state
        LLMProviderRegistry._initialized = False
        registry = LLMProviderRegistry()

        with pytest.raises(RuntimeError, match="not initialized"):
            await registry.get_provider()

    async def test_registry_initialize(self) -> None:
        """Test registry initialization."""
        config = LangChainConfiguration(
            providers={
                "openai": LangChainProviderConfig(models=["gpt-4"]),
            }
        )

        registry = LLMProviderRegistry()
        await registry.initialize(config)

        assert registry._initialized is True
        assert registry._config is config

    async def test_get_provider_openai(self) -> None:
        """Test getting OpenAI provider."""
        config = LangChainConfiguration(
            providers={
                "openai": LangChainProviderConfig(
                    api_key=SecretStr("test-key"),
                    models=["gpt-4"],
                ),
            }
        )

        registry = LLMProviderRegistry()
        # Clear previous state
        registry._initialized = False
        registry._providers = {}

        await registry.initialize(config)
        provider = await registry.get_provider("openai/gpt-4")

        assert provider is not None
        assert provider.model_name == "gpt-4"

    async def test_get_provider_azure(self) -> None:
        """Test getting Azure provider."""
        config = LangChainConfiguration(
            providers={
                "azure": LangChainProviderConfig(
                    api_key=SecretStr("test-key"),
                    api_base="https://example.openai.azure.com",
                    api_version="2024-01-01",
                    models=["gpt-4"],
                ),
            }
        )

        registry = LLMProviderRegistry()
        # Clear previous state
        registry._initialized = False
        registry._providers = {}

        await registry.initialize(config)
        provider = await registry.get_provider("azure/gpt-4")

        assert provider is not None

    async def test_get_provider_with_defaults(self) -> None:
        """Test getting provider using default values."""
        config = LangChainConfiguration(
            providers={
                "openai": LangChainProviderConfig(
                    api_key=SecretStr("test-key"),
                    models=["gpt-4"],
                ),
            },
            default_provider="openai",
            default_model="gpt-4",
        )

        registry = LLMProviderRegistry()
        # Clear previous state
        registry._initialized = False
        registry._providers = {}

        await registry.initialize(config)
        provider = await registry.get_provider()

        assert provider is not None
        assert provider.model_name == "gpt-4"

    async def test_get_provider_cached(self) -> None:
        """Test that providers are cached."""
        config = LangChainConfiguration(
            providers={
                "openai": LangChainProviderConfig(
                    api_key=SecretStr("test-key"),
                    models=["gpt-4"],
                ),
            }
        )

        registry = LLMProviderRegistry()
        # Clear previous state
        registry._initialized = False
        registry._providers = {}

        await registry.initialize(config)

        provider1 = await registry.get_provider("openai/gpt-4")
        provider2 = await registry.get_provider("openai/gpt-4")

        assert provider1 is provider2

    async def test_get_provider_not_found(self) -> None:
        """Test getting non-existent provider."""
        config = LangChainConfiguration(
            providers={
                "openai": LangChainProviderConfig(models=["gpt-4"]),
            }
        )

        registry = LLMProviderRegistry()
        # Clear previous state
        registry._initialized = False
        registry._providers = {}

        await registry.initialize(config)

        with pytest.raises(ValueError, match="not found in configuration"):
            await registry.get_provider("nonexistent/model")

    async def test_list_providers(self) -> None:
        """Test listing providers."""
        config = LangChainConfiguration(
            providers={
                "openai": LangChainProviderConfig(models=["gpt-4"]),
                "azure": LangChainProviderConfig(
                    api_base="https://example.com",
                    api_version="2024-01-01",
                    models=["gpt-4"],
                ),
            }
        )

        registry = LLMProviderRegistry()
        # Clear previous state
        registry._initialized = False
        registry._providers = {}

        await registry.initialize(config)

        providers = registry.list_providers()
        assert set(providers) == {"openai", "azure"}

    async def test_list_models(self) -> None:
        """Test listing models."""
        config = LangChainConfiguration(
            providers={
                "openai": LangChainProviderConfig(models=["gpt-4", "gpt-3.5-turbo"]),
                "azure": LangChainProviderConfig(
                    api_base="https://example.com",
                    api_version="2024-01-01",
                    models=["gpt-4"],
                ),
            }
        )

        registry = LLMProviderRegistry()
        # Clear previous state
        registry._initialized = False
        registry._providers = {}

        await registry.initialize(config)

        all_models = registry.list_models()
        assert set(all_models) == {
            "openai/gpt-4",
            "openai/gpt-3.5-turbo",
            "azure/gpt-4",
        }

        openai_models = registry.list_models("openai")
        assert set(openai_models) == {"openai/gpt-4", "openai/gpt-3.5-turbo"}

    async def test_clear_cache(self) -> None:
        """Test clearing provider cache."""
        config = LangChainConfiguration(
            providers={
                "openai": LangChainProviderConfig(
                    api_key=SecretStr("test-key"),
                    models=["gpt-4"],
                ),
            }
        )

        registry = LLMProviderRegistry()
        # Clear previous state
        registry._initialized = False
        registry._providers = {}

        await registry.initialize(config)

        await registry.get_provider("openai/gpt-4")
        assert len(registry._providers) == 1

        registry.clear_cache()
        assert len(registry._providers) == 0

    async def test_supports_streaming(self) -> None:
        """Test streaming support check."""
        config = LangChainConfiguration(
            providers={
                "openai": LangChainProviderConfig(models=["gpt-4"]),
            },
            enable_streaming=True,
        )

        registry = LLMProviderRegistry()
        # Clear previous state
        registry._initialized = False
        registry._providers = {}

        await registry.initialize(config)

        assert registry.supports_streaming("gpt-4") is True
