"""Unit tests for the Llama Stack implementation of /info endpoint."""

from typing import Any

import pytest
from fastapi import HTTPException, status
from llama_stack_client import APIConnectionError
from llama_stack_client.types import VersionInfo
from pytest_mock import MockerFixture

from app.endpoints.info_llama_stack import info_llama_stack
from configuration import AppConfig


@pytest.mark.asyncio
async def test_info_llama_stack(mocker: MockerFixture) -> None:
    """Test info_llama_stack returns service info."""
    config_dict: dict[Any, Any] = {
        "name": "foo",
        "service": {
            "host": "localhost",
            "port": 8080,
        },
        "user_data_collection": {},
        "authentication": {"module": "noop"},
        "authorization": {"access_rules": []},
        "llama_stack": {
            "api_key": "xyzzy",
            "url": "http://x.y.com:1234",
            "use_as_library_client": False,
        },
    }
    cfg = AppConfig()
    cfg.init_from_dict(config_dict)

    mocker.patch("app.endpoints.info_llama_stack.configuration", cfg)

    # Mock the LlamaStack client
    mock_client = mocker.AsyncMock()
    mock_client.inspect.version.return_value = VersionInfo(version="0.1.2")
    mock_lsc = mocker.patch("client.AsyncLlamaStackClientHolder.get_client")
    mock_lsc.return_value = mock_client

    response = await info_llama_stack()
    assert response is not None
    assert response.name == "foo"
    assert response.service_version is not None
    assert response.llama_stack_version == "0.1.2"


@pytest.mark.asyncio
async def test_info_llama_stack_connection_error(mocker: MockerFixture) -> None:
    """Test info_llama_stack when Llama Stack connection fails."""
    config_dict: dict[Any, Any] = {
        "name": "foo",
        "service": {
            "host": "localhost",
            "port": 8080,
        },
        "user_data_collection": {},
        "authentication": {"module": "noop"},
        "authorization": {"access_rules": []},
        "llama_stack": {
            "api_key": "xyzzy",
            "url": "http://x.y.com:1234",
            "use_as_library_client": False,
        },
    }
    cfg = AppConfig()
    cfg.init_from_dict(config_dict)

    mocker.patch("app.endpoints.info_llama_stack.configuration", cfg)

    # Mock the LlamaStack client
    mock_client = mocker.AsyncMock()
    mock_client.inspect.version.side_effect = APIConnectionError(request=None)  # type: ignore
    mock_lsc = mocker.patch("client.AsyncLlamaStackClientHolder.get_client")
    mock_lsc.return_value = mock_client

    with pytest.raises(HTTPException) as e:
        await info_llama_stack()
    assert e.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "Unable to connect to Llama Stack" in e.value.detail["response"]  # type: ignore
