"""Unit tests for the LangChain implementation of /providers endpoint."""

from typing import Any

import pytest
from fastapi import HTTPException, status
from pytest_mock import MockerFixture

from app.endpoints.providers_langchain import (
    get_provider_langchain,
    providers_langchain,
)
from configuration import AppConfig


@pytest.mark.asyncio
async def test_providers_langchain_configuration_not_loaded(
    mocker: MockerFixture,
) -> None:
    """Test providers_langchain when configuration is not loaded."""
    mock_config = AppConfig()
    mocker.patch("app.endpoints.providers_langchain.configuration", mock_config)

    with pytest.raises(HTTPException) as e:
        await providers_langchain()
    assert e.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert e.value.detail["response"] == "Configuration is not loaded"  # type: ignore


@pytest.mark.asyncio
async def test_providers_langchain_not_configured(
    mocker: MockerFixture,
) -> None:
    """Test providers_langchain when LangChain is not configured."""
    config_dict: dict[str, Any] = {
        "name": "test",
        "service": {"host": "localhost", "port": 8080},
        "user_data_collection": {},
        "authentication": {"module": "noop"},
        "authorization": {"access_rules": []},
        "llama_stack": {
            "api_key": "test-key",
            "url": "http://test.com:1234",
            "use_as_library_client": False,
        },
    }
    cfg = AppConfig()
    cfg.init_from_dict(config_dict)

    mocker.patch("app.endpoints.providers_langchain.configuration", cfg)

    with pytest.raises(HTTPException) as e:
        await providers_langchain()
    assert e.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "LangChain" in e.value.detail["response"]  # type: ignore


@pytest.mark.asyncio
async def test_providers_langchain_success(
    mocker: MockerFixture,
) -> None:
    """Test providers_langchain returns list of configured providers."""
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
                    "api_base": "https://example.openai.azure.com",
                    "api_version": "2024-01-01",
                    "models": ["gpt-4"],
                },
            },
            "default_provider": "openai",
            "default_model": "gpt-4",
        },
    }
    cfg = AppConfig()
    cfg.init_from_dict(config_dict)

    mocker.patch("app.endpoints.providers_langchain.configuration", cfg)

    response = await providers_langchain()
    assert "inference" in response.providers
    assert len(response.providers["inference"]) == 2
    provider_ids = [p["provider_id"] for p in response.providers["inference"]]
    assert "openai" in provider_ids
    assert "azure" in provider_ids


@pytest.mark.asyncio
async def test_providers_langchain_empty_providers(
    mocker: MockerFixture,
) -> None:
    """Test providers_langchain when no providers are configured."""
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

    mocker.patch("app.endpoints.providers_langchain.configuration", cfg)

    response = await providers_langchain()
    assert response.providers == {}


@pytest.mark.asyncio
async def test_get_provider_langchain_not_configured(
    mocker: MockerFixture,
) -> None:
    """Test get_provider_langchain when LangChain is not configured."""
    config_dict: dict[str, Any] = {
        "name": "test",
        "service": {"host": "localhost", "port": 8080},
        "user_data_collection": {},
        "authentication": {"module": "noop"},
        "authorization": {"access_rules": []},
        "llama_stack": {
            "api_key": "test-key",
            "url": "http://test.com:1234",
            "use_as_library_client": False,
        },
    }
    cfg = AppConfig()
    cfg.init_from_dict(config_dict)

    mocker.patch("app.endpoints.providers_langchain.configuration", cfg)

    with pytest.raises(HTTPException) as e:
        await get_provider_langchain(provider_id="openai")
    assert e.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "LangChain" in e.value.detail["response"]  # type: ignore


@pytest.mark.asyncio
async def test_get_provider_langchain_not_found(
    mocker: MockerFixture,
) -> None:
    """Test get_provider_langchain when provider is not found."""
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

    mocker.patch("app.endpoints.providers_langchain.configuration", cfg)

    with pytest.raises(HTTPException) as e:
        await get_provider_langchain(provider_id="azure")
    assert e.value.status_code == status.HTTP_404_NOT_FOUND
    detail = e.value.detail
    assert isinstance(detail, dict)
    assert "not found" in detail["response"]  # type: ignore


@pytest.mark.asyncio
async def test_get_provider_langchain_success(
    mocker: MockerFixture,
) -> None:
    """Test get_provider_langchain returns provider details."""
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
                    "timeout": 60,
                    "max_retries": 3,
                },
                "azure": {
                    "api_key": "test-key",
                    "api_base": "https://example.openai.azure.com",
                    "api_version": "2024-01-01",
                    "models": ["gpt-4"],
                    "timeout": 120,
                    "max_retries": 5,
                },
            },
            "default_provider": "openai",
            "default_model": "gpt-4",
        },
    }
    cfg = AppConfig()
    cfg.init_from_dict(config_dict)

    mocker.patch("app.endpoints.providers_langchain.configuration", cfg)

    response = await get_provider_langchain(provider_id="azure")
    assert response.provider_id == "azure"
    assert response.api == "inference"
    assert response.provider_type == "remote"
    assert response.health == {}
    assert response.config["models"] == ["gpt-4"]
    assert response.config["timeout"] == 120
    assert response.config["max_retries"] == 5
    assert response.config["api_base"] == "https://example.openai.azure.com"
    assert response.config["api_version"] == "2024-01-01"
