"""Unit tests for the Llama Stack implementation of /providers endpoint."""

import pytest
from fastapi import HTTPException, status
from llama_stack_client import APIConnectionError, BadRequestError
from llama_stack_client.types import ProviderInfo
from pytest_mock import MockerFixture

from app.endpoints.providers_llama_stack import (
    get_provider_llama_stack,
    providers_llama_stack,
)
from configuration import AppConfig


@pytest.mark.asyncio
async def test_providers_llama_stack_configuration_not_loaded(
    mocker: MockerFixture,
) -> None:
    """Test providers_llama_stack when configuration is not loaded."""
    mock_config = AppConfig()
    mocker.patch("app.endpoints.providers_llama_stack.configuration", mock_config)

    with pytest.raises(HTTPException) as e:
        await providers_llama_stack()
    assert e.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert e.value.detail["response"] == "Configuration is not loaded"  # type: ignore


@pytest.mark.asyncio
async def test_providers_llama_stack_connection_error(
    mocker: MockerFixture, minimal_config: AppConfig
) -> None:
    """Test providers_llama_stack when Llama Stack connection fails."""
    mocker.patch("app.endpoints.providers_llama_stack.configuration", minimal_config)

    mocker.patch(
        "app.endpoints.providers_llama_stack.AsyncLlamaStackClientHolder"
    ).return_value.get_client.side_effect = APIConnectionError(request=mocker.Mock())

    with pytest.raises(HTTPException) as e:
        await providers_llama_stack()
    assert e.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    detail = e.value.detail
    assert isinstance(detail, dict)
    assert detail["response"] == "Unable to connect to Llama Stack"  # type: ignore


@pytest.mark.asyncio
async def test_providers_llama_stack_success(
    mocker: MockerFixture, minimal_config: AppConfig
) -> None:
    """Test providers_llama_stack returns grouped list of providers."""
    mocker.patch("app.endpoints.providers_llama_stack.configuration", minimal_config)

    provider_list = [
        ProviderInfo(
            api="inference",
            provider_id="openai",
            provider_type="remote::openai",
            config={},
            health={},
        ),
        ProviderInfo(
            api="inference",
            provider_id="st",
            provider_type="inline::sentence-transformers",
            config={},
            health={},
        ),
        ProviderInfo(
            api="datasetio",
            provider_id="huggingface",
            provider_type="remote::huggingface",
            config={},
            health={},
        ),
    ]
    mock_client = mocker.AsyncMock()
    mock_client.providers.list.return_value = provider_list
    mocker.patch(
        "app.endpoints.providers_llama_stack.AsyncLlamaStackClientHolder"
    ).return_value.get_client.return_value = mock_client

    response = await providers_llama_stack()
    assert "inference" in response.providers
    assert len(response.providers["inference"]) == 2
    assert "datasetio" in response.providers


@pytest.mark.asyncio
async def test_get_provider_llama_stack_not_found(
    mocker: MockerFixture, minimal_config: AppConfig
) -> None:
    """Test get_provider_llama_stack when provider is not found."""
    mocker.patch("app.endpoints.providers_llama_stack.configuration", minimal_config)

    mock_client_holder = mocker.patch(
        "app.endpoints.providers_llama_stack.AsyncLlamaStackClientHolder"
    )
    mock_client = mocker.AsyncMock()
    mock_client.providers.retrieve = mocker.AsyncMock(
        side_effect=BadRequestError(
            message="Provider not found",
            response=mocker.Mock(request=None),
            body=None,
        )
    )  # type: ignore
    mock_client_holder.return_value.get_client.return_value = mock_client

    with pytest.raises(HTTPException) as e:
        await get_provider_llama_stack(provider_id="openai")
    assert e.value.status_code == status.HTTP_404_NOT_FOUND
    detail = e.value.detail
    assert isinstance(detail, dict)
    assert "not found" in detail["response"]  # type: ignore
    assert "Provider with ID openai does not exist" in detail["cause"]  # type: ignore


@pytest.mark.asyncio
async def test_get_provider_llama_stack_success(
    mocker: MockerFixture, minimal_config: AppConfig
) -> None:
    """Test get_provider_llama_stack returns provider details."""
    mocker.patch("app.endpoints.providers_llama_stack.configuration", minimal_config)

    provider = ProviderInfo(
        api="inference",
        provider_id="openai",
        provider_type="remote::openai",
        config={"api_key": "*****"},
        health={"status": "OK", "message": "Healthy"},
    )
    mock_client = mocker.AsyncMock()
    mock_client.providers.retrieve = mocker.AsyncMock(return_value=provider)
    mocker.patch(
        "app.endpoints.providers_llama_stack.AsyncLlamaStackClientHolder"
    ).return_value.get_client.return_value = mock_client

    response = await get_provider_llama_stack(provider_id="openai")
    assert response.provider_id == "openai"
    assert response.api == "inference"


@pytest.mark.asyncio
async def test_get_provider_llama_stack_connection_error(
    mocker: MockerFixture, minimal_config: AppConfig
) -> None:
    """Test get_provider_llama_stack when Llama Stack connection fails."""
    mocker.patch("app.endpoints.providers_llama_stack.configuration", minimal_config)

    mocker.patch(
        "app.endpoints.providers_llama_stack.AsyncLlamaStackClientHolder"
    ).return_value.get_client.side_effect = APIConnectionError(request=mocker.Mock())

    with pytest.raises(HTTPException) as e:
        await get_provider_llama_stack(provider_id="openai")
    assert e.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    detail = e.value.detail
    assert isinstance(detail, dict)
    assert detail["response"] == "Unable to connect to Llama Stack"  # type: ignore
