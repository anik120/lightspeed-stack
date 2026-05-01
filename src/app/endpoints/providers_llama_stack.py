"""Llama Stack implementation for /providers endpoint."""

from typing import Any

from fastapi import HTTPException
from llama_stack_client import APIConnectionError, BadRequestError
from llama_stack_client.types import ProviderListResponse

from client import AsyncLlamaStackClientHolder
from configuration import configuration
from log import get_logger
from models.responses import (
    NotFoundResponse,
    ProviderResponse,
    ProvidersListResponse,
    ServiceUnavailableResponse,
)
from utils.endpoints import check_configuration_loaded

logger = get_logger(__name__)


async def providers_llama_stack() -> ProvidersListResponse:
    """
    Retrieve available providers from Llama Stack.

    Returns:
        ProvidersListResponse: Mapping from API type to list of providers.

    Raises:
        HTTPException: If unable to connect to Llama Stack or retrieval fails.
    """
    check_configuration_loaded(configuration)

    llama_stack_configuration = configuration.llama_stack_configuration
    logger.info("Llama stack config: %s", llama_stack_configuration)

    try:
        client = AsyncLlamaStackClientHolder().get_client()
        providers: ProviderListResponse = await client.providers.list()
        return ProvidersListResponse(providers=group_providers(providers))

    except APIConnectionError as e:
        logger.error("Unable to connect to Llama Stack: %s", e)
        response = ServiceUnavailableResponse(backend_name="Llama Stack", cause=str(e))
        raise HTTPException(**response.model_dump()) from e


def group_providers(providers: ProviderListResponse) -> dict[str, list[dict[str, Any]]]:
    """Group a list of ProviderInfo objects by their API type.

    Args:
        providers: List of ProviderInfo objects.

    Returns:
        Mapping from API type to list of providers containing
        only 'provider_id' and 'provider_type'.
    """
    result: dict[str, list[dict[str, Any]]] = {}
    for provider in providers:
        result.setdefault(provider.api, []).append(
            {
                "provider_id": provider.provider_id,
                "provider_type": provider.provider_type,
            }
        )
    return result


async def get_provider_llama_stack(provider_id: str) -> ProviderResponse:
    """
    Retrieve a single provider from Llama Stack.

    Parameters:
        provider_id: Provider identification string.

    Returns:
        ProviderResponse: Provider details.

    Raises:
        HTTPException: If provider not found or unable to connect to Llama Stack.
    """
    check_configuration_loaded(configuration)

    llama_stack_configuration = configuration.llama_stack_configuration
    logger.info("Llama stack config: %s", llama_stack_configuration)

    try:
        client = AsyncLlamaStackClientHolder().get_client()
        provider = await client.providers.retrieve(provider_id)
        return ProviderResponse(**provider.model_dump())

    except APIConnectionError as e:
        logger.error("Unable to connect to Llama Stack: %s", e)
        response = ServiceUnavailableResponse(backend_name="Llama Stack", cause=str(e))
        raise HTTPException(**response.model_dump()) from e

    except BadRequestError as e:
        response = NotFoundResponse(resource="provider", resource_id=provider_id)
        raise HTTPException(**response.model_dump()) from e
