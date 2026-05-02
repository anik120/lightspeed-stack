"""Unit tests for LangChain provider registry."""

import pytest
from pydantic import SecretStr

from langchain_providers.azure import AzureProvider
from langchain_providers.openai import OpenAIProvider
from langchain_providers.registry import LLMProviderRegistry
from langchain_providers.watsonx import WatsonxProvider
from models.config import LangChainConfiguration, LangChainProviderConfig


@pytest.fixture
def openai_config():
    """OpenAI provider configuration."""
    return LangChainProviderConfig(
        api_key=SecretStr("test-key"),
        models=["gpt-4", "gpt-3.5-turbo"],
        timeout=30,
        max_retries=3,
    )


@pytest.fixture
def azure_config():
    """Azure provider configuration."""
    return LangChainProviderConfig(
        api_key=SecretStr("test-key"),
        api_base="https://test.openai.azure.com",
        api_version="2023-05-15",
        models=["gpt-4", "gpt-35-turbo"],
        timeout=30,
    )


@pytest.fixture
def watsonx_config():
    """Watsonx provider configuration."""
    return LangChainProviderConfig(
        api_key=SecretStr("test-key"),
        api_base="https://us-south.ml.cloud.ibm.com",
        models=["ibm/granite-13b-chat-v2"],
        timeout=60,
    )


@pytest.fixture
def langchain_config(openai_config, azure_config):
    """LangChain configuration with multiple providers."""
    return LangChainConfiguration(
        providers={
            "openai": openai_config,
            "azure": azure_config,
        },
        default_provider="openai",
        default_model="gpt-4",
    )


@pytest.mark.asyncio
async def test_registry_initialization(langchain_config):
    """Test successful registry initialization."""
    registry = LLMProviderRegistry()
    assert not registry.is_initialized

    await registry.initialize(langchain_config)

    assert registry.is_initialized
    assert len(registry.list_providers()) == 2
    assert "openai" in registry.list_providers()
    assert "azure" in registry.list_providers()


@pytest.mark.asyncio
async def test_registry_initialization_single_provider(openai_config):
    """Test registry with single provider."""
    config = LangChainConfiguration(
        providers={"openai": openai_config},
        default_provider="openai",
        default_model="gpt-4",
    )

    registry = LLMProviderRegistry()
    await registry.initialize(config)

    assert registry.is_initialized
    assert registry.list_providers() == ["openai"]


@pytest.mark.asyncio
async def test_registry_initialization_all_providers(
    openai_config, azure_config, watsonx_config
):
    """Test registry with all supported providers."""
    config = LangChainConfiguration(
        providers={
            "openai": openai_config,
            "azure": azure_config,
            "watsonx": watsonx_config,
        },
        default_provider="openai",
        default_model="gpt-4",
    )

    registry = LLMProviderRegistry()
    await registry.initialize(config)

    assert len(registry.list_providers()) == 3
    assert set(registry.list_providers()) == {"openai", "azure", "watsonx"}


@pytest.mark.asyncio
async def test_registry_invalid_default_provider(openai_config):
    """Test registry initialization with invalid default provider."""
    config = LangChainConfiguration(
        providers={"openai": openai_config},
        default_provider="nonexistent",
        default_model="gpt-4",
    )

    registry = LLMProviderRegistry()
    with pytest.raises(ValueError, match="Default provider .* not found"):
        await registry.initialize(config)


@pytest.mark.asyncio
async def test_get_provider(langchain_config):
    """Test getting provider instances."""
    registry = LLMProviderRegistry()
    await registry.initialize(langchain_config)

    openai_provider = registry.get_provider("openai")
    assert isinstance(openai_provider, OpenAIProvider)
    assert openai_provider.provider_name == "openai"

    azure_provider = registry.get_provider("azure")
    assert isinstance(azure_provider, AzureProvider)
    assert azure_provider.provider_name == "azure"


@pytest.mark.asyncio
async def test_get_provider_not_found(langchain_config):
    """Test getting non-existent provider."""
    registry = LLMProviderRegistry()
    await registry.initialize(langchain_config)

    with pytest.raises(ValueError, match="Provider .* not found"):
        registry.get_provider("nonexistent")


@pytest.mark.asyncio
async def test_get_provider_before_initialization():
    """Test getting provider before initialization."""
    registry = LLMProviderRegistry()

    with pytest.raises(RuntimeError, match="not initialized"):
        registry.get_provider("openai")


@pytest.mark.asyncio
async def test_get_chat_model_with_defaults(langchain_config):
    """Test getting chat model with default provider and model."""
    registry = LLMProviderRegistry()
    await registry.initialize(langchain_config)

    model = await registry.get_chat_model()

    assert model is not None
    assert model.model_name == "gpt-4"


@pytest.mark.asyncio
async def test_get_chat_model_specific_provider(langchain_config):
    """Test getting chat model from specific provider."""
    registry = LLMProviderRegistry()
    await registry.initialize(langchain_config)

    # Get from OpenAI
    openai_model = await registry.get_chat_model(model_id="gpt-3.5-turbo", provider="openai")
    assert openai_model.model_name == "gpt-3.5-turbo"

    # Get from Azure
    azure_model = await registry.get_chat_model(model_id="gpt-35-turbo", provider="azure")
    assert azure_model.deployment_name == "gpt-35-turbo"


@pytest.mark.asyncio
async def test_get_chat_model_invalid_provider(langchain_config):
    """Test getting chat model from invalid provider."""
    registry = LLMProviderRegistry()
    await registry.initialize(langchain_config)

    with pytest.raises(ValueError, match="Provider .* not found"):
        await registry.get_chat_model(provider="nonexistent")


@pytest.mark.asyncio
async def test_get_chat_model_unavailable_model(langchain_config):
    """Test getting unavailable model from provider."""
    registry = LLMProviderRegistry()
    await registry.initialize(langchain_config)

    with pytest.raises(ValueError, match="not available"):
        await registry.get_chat_model(model_id="gpt-5", provider="openai")


@pytest.mark.asyncio
async def test_get_chat_model_before_initialization():
    """Test getting chat model before initialization."""
    registry = LLMProviderRegistry()

    with pytest.raises(RuntimeError, match="not initialized"):
        await registry.get_chat_model()


@pytest.mark.asyncio
async def test_list_models_all_providers(langchain_config):
    """Test listing models from all providers."""
    registry = LLMProviderRegistry()
    await registry.initialize(langchain_config)

    models = registry.list_models()

    # Should include models from both OpenAI and Azure
    assert "gpt-4" in models
    assert "gpt-3.5-turbo" in models
    assert "gpt-35-turbo" in models


@pytest.mark.asyncio
async def test_list_models_specific_provider(langchain_config):
    """Test listing models from specific provider."""
    registry = LLMProviderRegistry()
    await registry.initialize(langchain_config)

    openai_models = registry.list_models(provider="openai")
    assert set(openai_models) == {"gpt-4", "gpt-3.5-turbo"}

    azure_models = registry.list_models(provider="azure")
    assert set(azure_models) == {"gpt-4", "gpt-35-turbo"}


@pytest.mark.asyncio
async def test_reinitialize_registry(langchain_config, openai_config):
    """Test reinitializing registry with different configuration."""
    registry = LLMProviderRegistry()

    # Initial initialization
    await registry.initialize(langchain_config)
    assert len(registry.list_providers()) == 2

    # Reinitialize with different config
    new_config = LangChainConfiguration(
        providers={"openai": openai_config},
        default_provider="openai",
        default_model="gpt-4",
    )
    await registry.initialize(new_config)
    assert len(registry.list_providers()) == 1


@pytest.mark.asyncio
async def test_azure_provider_missing_api_base():
    """Test Azure provider requires api_base."""
    config = LangChainProviderConfig(
        api_key=SecretStr("test-key"),
        models=["gpt-4"],
    )

    with pytest.raises(ValueError, match="api_base"):
        AzureProvider(config)


@pytest.mark.asyncio
async def test_watsonx_provider_missing_api_base():
    """Test Watsonx provider requires api_base."""
    config = LangChainProviderConfig(
        api_key=SecretStr("test-key"),
        models=["ibm/granite-13b-chat-v2"],
    )

    with pytest.raises(ValueError, match="api_base"):
        WatsonxProvider(config)
