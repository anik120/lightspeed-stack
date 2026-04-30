"""Unit tests for the LangChain implementation of /models endpoint."""

from typing import Any

import pytest
from fastapi import HTTPException, status
from pytest_mock import MockerFixture

from app.endpoints.models_langchain import models_langchain
from configuration import AppConfig
from models.requests import ModelFilter


@pytest.mark.asyncio
async def test_models_langchain_configuration_not_loaded(
    mocker: MockerFixture,
) -> None:
    """Test models_langchain when configuration is not loaded."""
    # simulate state when no configuration is loaded
    mock_config = AppConfig()
    mocker.patch("app.endpoints.models_langchain.configuration", mock_config)

    with pytest.raises(HTTPException) as e:
        await models_langchain(model_type=ModelFilter(model_type=None))
        assert e.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert e.value.detail["response"] == "Configuration is not loaded"  # type: ignore


@pytest.mark.asyncio
async def test_models_langchain_registry_error(
    mocker: MockerFixture,
) -> None:
    """Test models_langchain when registry initialization fails."""
    # Minimal configuration for LangChain tests
    config_dict: dict[str, Any] = {
        "name": "test",
        "service": {"host": "localhost", "port": 8080},
        "user_data_collection": {},
        "authentication": {"module": "noop"},
        "authorization": {"access_rules": []},
        "langchain": {
            "providers": {
                "openai": {
                    "api_key": "test-key",
                    "models": ["gpt-4"],
                }
            },
            "default_provider": "openai",
            "default_model": "gpt-4",
        },
    }
    cfg = AppConfig()
    cfg.init_from_dict(config_dict)

    mocker.patch("app.endpoints.models_langchain.configuration", cfg)

    # Mock LLMProviderRegistry to raise an error
    mock_registry_class = mocker.patch(
        "app.endpoints.models_langchain.LLMProviderRegistry"
    )
    mock_registry = mocker.AsyncMock()
    mock_registry.initialize.side_effect = Exception("Registry initialization failed")
    mock_registry_class.return_value = mock_registry

    with pytest.raises(HTTPException) as e:
        await models_langchain(model_type=ModelFilter(model_type=None))
    assert e.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "LangChain" in e.value.detail["response"]  # type: ignore


@pytest.mark.asyncio
async def test_models_langchain_models_retrieved(
    mocker: MockerFixture,
) -> None:
    """Test models_langchain when models are successfully retrieved."""
    # Minimal configuration for LangChain tests
    config_dict: dict[str, Any] = {
        "name": "test",
        "service": {"host": "localhost", "port": 8080},
        "user_data_collection": {},
        "authentication": {"module": "noop"},
        "authorization": {"access_rules": []},
        "langchain": {
            "providers": {
                "openai": {
                    "api_key": "test-key",
                    "models": ["gpt-4", "gpt-3.5-turbo"],
                },
                "azure": {
                    "api_key": "test-key",
                    "models": ["gpt-4-azure"],
                },
            },
            "default_provider": "openai",
            "default_model": "gpt-4",
        },
    }
    cfg = AppConfig()
    cfg.init_from_dict(config_dict)

    mocker.patch("app.endpoints.models_langchain.configuration", cfg)

    # Mock LLMProviderRegistry
    mock_registry_class = mocker.patch(
        "app.endpoints.models_langchain.LLMProviderRegistry"
    )
    mock_registry = mocker.AsyncMock()
    mock_registry.list_models_by_provider.return_value = {
        "openai": ["gpt-4", "gpt-3.5-turbo"],
        "azure": ["gpt-4-azure"],
    }
    mock_registry_class.return_value = mock_registry

    response = await models_langchain(model_type=ModelFilter(model_type=None))

    assert response is not None
    assert len(response.models) == 3
    assert response.models[0]["identifier"] == "openai/gpt-4"
    assert response.models[0]["provider_id"] == "openai"
    assert response.models[0]["model_type"] == "llm"
    assert response.models[1]["identifier"] == "openai/gpt-3.5-turbo"
    assert response.models[2]["identifier"] == "azure/gpt-4-azure"


@pytest.mark.asyncio
async def test_models_langchain_filter_by_type(
    mocker: MockerFixture,
) -> None:
    """Test models_langchain with model_type filter."""
    # Minimal configuration for LangChain tests
    config_dict: dict[str, Any] = {
        "name": "test",
        "service": {"host": "localhost", "port": 8080},
        "user_data_collection": {},
        "authentication": {"module": "noop"},
        "authorization": {"access_rules": []},
        "langchain": {
            "providers": {
                "openai": {
                    "api_key": "test-key",
                    "models": ["gpt-4"],
                }
            },
            "default_provider": "openai",
            "default_model": "gpt-4",
        },
    }
    cfg = AppConfig()
    cfg.init_from_dict(config_dict)

    mocker.patch("app.endpoints.models_langchain.configuration", cfg)

    # Mock LLMProviderRegistry
    mock_registry_class = mocker.patch(
        "app.endpoints.models_langchain.LLMProviderRegistry"
    )
    mock_registry = mocker.AsyncMock()
    mock_registry.list_models_by_provider.return_value = {
        "openai": ["gpt-4"],
    }
    mock_registry_class.return_value = mock_registry

    # Test with llm filter (should return results)
    response = await models_langchain(model_type=ModelFilter(model_type="llm"))
    assert response is not None
    assert len(response.models) == 1

    # Test with embedding filter (should return empty since all are llm)
    response = await models_langchain(model_type=ModelFilter(model_type="embedding"))
    assert response is not None
    assert len(response.models) == 0


@pytest.mark.asyncio
async def test_models_langchain_empty_models(
    mocker: MockerFixture,
) -> None:
    """Test models_langchain when no models are available."""
    # Minimal configuration for LangChain tests
    config_dict: dict[str, Any] = {
        "name": "test",
        "service": {"host": "localhost", "port": 8080},
        "user_data_collection": {},
        "authentication": {"module": "noop"},
        "authorization": {"access_rules": []},
        "langchain": {
            "providers": {},
            "default_provider": "openai",
            "default_model": "gpt-4",
        },
    }
    cfg = AppConfig()
    cfg.init_from_dict(config_dict)

    mocker.patch("app.endpoints.models_langchain.configuration", cfg)

    # Mock LLMProviderRegistry
    mock_registry_class = mocker.patch(
        "app.endpoints.models_langchain.LLMProviderRegistry"
    )
    mock_registry = mocker.AsyncMock()
    mock_registry.list_models_by_provider.return_value = {}
    mock_registry_class.return_value = mock_registry

    response = await models_langchain(model_type=ModelFilter(model_type=None))
    assert response is not None
    assert len(response.models) == 0
