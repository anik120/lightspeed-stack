"""Unit tests for the Llama Stack implementation of /shields endpoint."""

from typing import Any

import pytest
from fastapi import HTTPException, status
from llama_stack_client import APIConnectionError
from pytest_mock import MockerFixture

from app.endpoints.shields_llama_stack import shields_llama_stack
from configuration import AppConfig
from models.responses import ShieldsResponse


@pytest.mark.asyncio
async def test_shields_llama_stack_configuration_not_loaded(
    mocker: MockerFixture,
) -> None:
    """Test shields_llama_stack when configuration is not loaded."""
    # simulate state when no configuration is loaded
    mock_config = AppConfig()
    mocker.patch("app.endpoints.shields_llama_stack.configuration", mock_config)

    with pytest.raises(HTTPException) as e:
        await shields_llama_stack()
        assert e.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert e.value.detail["response"] == "Configuration is not loaded"  # type: ignore


@pytest.mark.asyncio
async def test_shields_llama_stack_empty_shields_list(
    mocker: MockerFixture,
) -> None:
    """Test shields_llama_stack when shields list is empty."""
    config_dict: dict[str, Any] = {
        "name": "test",
        "service": {
            "host": "localhost",
            "port": 8080,
        },
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

    mocker.patch("app.endpoints.shields_llama_stack.configuration", cfg)

    mock_client_holder = mocker.patch(
        "app.endpoints.shields_llama_stack.AsyncLlamaStackClientHolder"
    )
    mock_client = mocker.AsyncMock()
    mock_client.shields.list.return_value = []
    mock_client_holder.return_value.get_client.return_value = mock_client

    response = await shields_llama_stack()
    assert isinstance(response, ShieldsResponse)
    assert response.shields == []


@pytest.mark.asyncio
async def test_shields_llama_stack_connection_error(
    mocker: MockerFixture,
) -> None:
    """Test shields_llama_stack when connection to Llama Stack fails."""
    config_dict: dict[str, Any] = {
        "name": "test",
        "service": {
            "host": "localhost",
            "port": 8080,
        },
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

    mocker.patch("app.endpoints.shields_llama_stack.configuration", cfg)

    mock_client_holder = mocker.patch(
        "app.endpoints.shields_llama_stack.AsyncLlamaStackClientHolder"
    )
    mock_client = mocker.AsyncMock()
    mock_client.shields.list.side_effect = APIConnectionError(request=None)  # type: ignore
    mock_client_holder.return_value.get_client.return_value = mock_client

    with pytest.raises(HTTPException) as e:
        await shields_llama_stack()
    assert e.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert e.value.detail["response"] == "Unable to connect to Llama Stack"  # type: ignore


@pytest.mark.asyncio
async def test_shields_llama_stack_retrieves_shields(
    mocker: MockerFixture,
) -> None:
    """Test shields_llama_stack successfully retrieves shields."""
    config_dict: dict[str, Any] = {
        "name": "test",
        "service": {
            "host": "localhost",
            "port": 8080,
        },
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

    mocker.patch("app.endpoints.shields_llama_stack.configuration", cfg)

    mock_client = mocker.AsyncMock()
    mock_client.shields.list.return_value = []
    mock_lsc = mocker.patch("client.AsyncLlamaStackClientHolder.get_client")
    mock_lsc.return_value = mock_client

    response = await shields_llama_stack()
    assert response is not None
    assert isinstance(response, ShieldsResponse)


@pytest.mark.asyncio
async def test_shields_llama_stack_success_with_shields_data(
    mocker: MockerFixture,
) -> None:
    """Test shields_llama_stack with successful response and shields data."""
    config_dict: dict[str, Any] = {
        "name": "test",
        "service": {
            "host": "localhost",
            "port": 8080,
        },
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

    mocker.patch("app.endpoints.shields_llama_stack.configuration", cfg)

    mock_shields_data = [
        {
            "identifier": "lightspeed_question_validity-shield",
            "provider_resource_id": "lightspeed_question_validity-shield",
            "provider_id": "lightspeed_question_validity",
            "type": "shield",
            "params": {},
        },
        {
            "identifier": "content_filter-shield",
            "provider_resource_id": "content_filter-shield",
            "provider_id": "content_filter",
            "type": "shield",
            "params": {"threshold": 0.8},
        },
    ]

    mock_client = mocker.AsyncMock()
    mock_client.shields.list.return_value = mock_shields_data
    mock_lsc = mocker.patch("client.AsyncLlamaStackClientHolder.get_client")
    mock_lsc.return_value = mock_client

    response = await shields_llama_stack()

    assert response is not None
    assert hasattr(response, "shields")
    assert len(response.shields) == 2
    assert response.shields[0]["identifier"] == "lightspeed_question_validity-shield"
    assert response.shields[1]["identifier"] == "content_filter-shield"
