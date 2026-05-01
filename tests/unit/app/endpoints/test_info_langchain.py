"""Unit tests for the LangChain implementation of /info endpoint."""

from typing import Any

import pytest
from fastapi import HTTPException, status
from pytest_mock import MockerFixture

from app.endpoints.info_langchain import info_langchain
from configuration import AppConfig


@pytest.mark.asyncio
async def test_info_langchain_configuration_not_loaded(
    mocker: MockerFixture,
) -> None:
    """Test info_langchain when configuration is not loaded."""
    mock_config = AppConfig()
    mocker.patch("app.endpoints.info_langchain.configuration", mock_config)

    with pytest.raises(HTTPException) as e:
        await info_langchain()
    assert e.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert e.value.detail["response"] == "Configuration is not loaded"  # type: ignore


@pytest.mark.asyncio
async def test_info_langchain_not_configured(
    mocker: MockerFixture,
) -> None:
    """Test info_langchain when LangChain is not configured."""
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

    mocker.patch("app.endpoints.info_langchain.configuration", cfg)

    with pytest.raises(HTTPException) as e:
        await info_langchain()
    assert e.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "LangChain" in e.value.detail["response"]  # type: ignore


@pytest.mark.asyncio
async def test_info_langchain_success(
    mocker: MockerFixture,
) -> None:
    """Test info_langchain returns service info with LangChain version."""
    config_dict: dict[str, Any] = {
        "name": "Test Service",
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

    mocker.patch("app.endpoints.info_langchain.configuration", cfg)

    response = await info_langchain()
    assert response is not None
    assert response.name == "Test Service"
    assert response.service_version is not None
    assert response.llama_stack_version is not None  # Contains LangChain version
    assert len(response.llama_stack_version) > 0
