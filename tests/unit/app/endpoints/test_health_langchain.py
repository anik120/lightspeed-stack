"""Unit tests for the LangChain implementation of /health endpoint."""

import pytest
from pytest_mock import MockerFixture

from app.endpoints.health_langchain import readiness_langchain
from configuration import AppConfig


@pytest.mark.asyncio
async def test_readiness_langchain_registry_init_fails(mocker: MockerFixture) -> None:
    """Test readiness_langchain when registry initialization fails."""
    # Mock configuration with LangChain
    cfg = AppConfig()
    cfg.init_from_dict(
        {
            "name": "test",
            "service": {"host": "localhost", "port": 8080},
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
            "user_data_collection": {},
            "authentication": {"module": "noop"},
            "authorization": {"access_rules": []},
        }
    )
    mocker.patch("app.endpoints.health_langchain.configuration", cfg)

    # Mock registry to fail initialization
    mock_registry_class = mocker.patch("app.endpoints.health_langchain.LLMProviderRegistry")
    mock_registry = mocker.AsyncMock()
    mock_registry.initialize.side_effect = Exception("Init failed")
    mock_registry_class.return_value = mock_registry

    mock_response = mocker.Mock()
    response = await readiness_langchain(response=mock_response)

    assert response.ready is False
    assert "Failed to initialize LangChain registry" in response.reason
    assert mock_response.status_code == 503
    assert len(response.providers) == 1
    assert response.providers[0].provider_id == "langchain"
    assert response.providers[0].status == "error"


@pytest.mark.asyncio
async def test_readiness_langchain_provider_unhealthy(mocker: MockerFixture) -> None:
    """Test readiness_langchain when a provider fails to initialize."""
    # Mock configuration with LangChain
    cfg = AppConfig()
    cfg.init_from_dict(
        {
            "name": "test",
            "service": {"host": "localhost", "port": 8080},
            "langchain": {
                "providers": {
                    "openai": {
                        "api_key": "test-key",
                        "models": ["gpt-4"],
                    },
                    "azure": {
                        "api_key": "test-key",
                        "api_base": "https://test.openai.azure.com",
                        "models": ["gpt-4"],
                    },
                },
                "default_provider": "openai",
                "default_model": "gpt-4",
            },
            "user_data_collection": {},
            "authentication": {"module": "noop"},
            "authorization": {"access_rules": []},
        }
    )
    mocker.patch("app.endpoints.health_langchain.configuration", cfg)

    # Mock registry - openai succeeds, azure fails
    mock_registry_class = mocker.patch("app.endpoints.health_langchain.LLMProviderRegistry")
    mock_registry = mocker.AsyncMock()
    mock_registry.initialize = mocker.AsyncMock()

    # Side effect: first call (openai) succeeds, second call (azure) fails
    call_count = 0

    async def mock_get_provider(model_str: str):
        nonlocal call_count
        call_count += 1
        if "azure" in model_str:
            raise Exception("Azure provider failed")
        return mocker.Mock()

    mock_registry.get_provider = mock_get_provider
    mock_registry_class.return_value = mock_registry

    mock_response = mocker.Mock()
    response = await readiness_langchain(response=mock_response)

    assert response.ready is False
    assert "azure" in response.reason
    assert "Providers not healthy" in response.reason
    assert mock_response.status_code == 503
    assert len(response.providers) == 1
    assert response.providers[0].provider_id == "azure"
    assert response.providers[0].status == "error"


@pytest.mark.asyncio
async def test_readiness_langchain_all_providers_healthy(mocker: MockerFixture) -> None:
    """Test readiness_langchain when all providers are healthy."""
    # Mock configuration with LangChain
    cfg = AppConfig()
    cfg.init_from_dict(
        {
            "name": "test",
            "service": {"host": "localhost", "port": 8080},
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
            "user_data_collection": {},
            "authentication": {"module": "noop"},
            "authorization": {"access_rules": []},
        }
    )
    mocker.patch("app.endpoints.health_langchain.configuration", cfg)

    # Mock registry - all providers succeed
    mock_registry_class = mocker.patch("app.endpoints.health_langchain.LLMProviderRegistry")
    mock_registry = mocker.AsyncMock()
    mock_registry.initialize = mocker.AsyncMock()
    mock_registry.get_provider = mocker.AsyncMock(return_value=mocker.Mock())
    mock_registry_class.return_value = mock_registry

    mock_response = mocker.Mock()
    response = await readiness_langchain(response=mock_response)

    assert response.ready is True
    assert response.reason == "All providers are healthy"
    assert len(response.providers) == 0  # No unhealthy providers
