"""LangChain implementation for /providers endpoint."""

from typing import Any

from fastapi import HTTPException

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


async def providers_langchain() -> ProvidersListResponse:
    """
    Retrieve available providers from LangChain configuration.

    Returns:
        ProvidersListResponse: Mapping from API type to list of providers.

    Raises:
        HTTPException: If unable to retrieve providers from LangChain.
    """
    check_configuration_loaded(configuration)

    langchain_config = configuration.langchain_configuration
    if langchain_config is None:
        raise HTTPException(
            status_code=503,
            detail={
                "backend_name": "LangChain",
                "response": "LangChain not configured",
            },
        )

    logger.info("Retrieving providers from LangChain configuration")

    try:
        # Group LangChain providers by API type
        # LangChain providers are primarily for LLM inference
        providers_by_api: dict[str, list[dict[str, Any]]] = {}

        for provider_name in langchain_config.providers.keys():
            providers_by_api.setdefault("inference", []).append(
                {
                    "provider_id": provider_name,
                    "provider_type": "remote",
                }
            )

        return ProvidersListResponse(providers=providers_by_api)

    except Exception as e:
        logger.error("Unable to retrieve providers from LangChain: %s", e)
        response = ServiceUnavailableResponse(backend_name="LangChain", cause=str(e))
        raise HTTPException(**response.model_dump()) from e


async def get_provider_langchain(provider_id: str) -> ProviderResponse:
    """
    Retrieve a single provider from LangChain configuration.

    Parameters:
        provider_id: Provider identification string.

    Returns:
        ProviderResponse: Provider details.

    Raises:
        HTTPException: If provider not found or unable to retrieve from LangChain.
    """
    check_configuration_loaded(configuration)

    langchain_config = configuration.langchain_configuration
    if langchain_config is None:
        raise HTTPException(
            status_code=503,
            detail={
                "backend_name": "LangChain",
                "response": "LangChain not configured",
            },
        )

    logger.info("Retrieving provider %s from LangChain configuration", provider_id)

    try:
        if provider_id not in langchain_config.providers:
            response = NotFoundResponse(resource="provider", resource_id=provider_id)
            raise HTTPException(**response.model_dump())

        provider_config = langchain_config.providers[provider_id]

        # Build provider response with available metadata
        provider_data = {
            "provider_id": provider_id,
            "provider_type": "remote",
            "api": "inference",
            "config": {
                "models": provider_config.models,
                "timeout": provider_config.timeout,
                "max_retries": provider_config.max_retries,
            },
            "health": {},
        }

        if provider_config.api_base:
            provider_data["config"]["api_base"] = provider_config.api_base
        if provider_config.api_version:
            provider_data["config"]["api_version"] = provider_config.api_version

        return ProviderResponse(**provider_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unable to retrieve provider from LangChain: %s", e)
        response = ServiceUnavailableResponse(backend_name="LangChain", cause=str(e))
        raise HTTPException(**response.model_dump()) from e
