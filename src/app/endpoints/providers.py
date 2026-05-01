"""Providers endpoint for listing and retrieving available providers."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from authentication import get_auth_dependency
from authentication.interface import AuthTuple
from authorization.middleware import authorize
from configuration import configuration
from log import get_logger
from models.config import Action
from models.responses import (
    UNAUTHORIZED_OPENAPI_EXAMPLES,
    ForbiddenResponse,
    InternalServerErrorResponse,
    NotFoundResponse,
    ProviderResponse,
    ProvidersListResponse,
    ServiceUnavailableResponse,
    UnauthorizedResponse,
)
from utils.endpoints import check_configuration_loaded

logger = get_logger(__name__)
router = APIRouter(tags=["providers"])


providers_list_responses: dict[int | str, dict[str, Any]] = {
    200: ProvidersListResponse.openapi_response(),
    401: UnauthorizedResponse.openapi_response(examples=UNAUTHORIZED_OPENAPI_EXAMPLES),
    403: ForbiddenResponse.openapi_response(examples=["endpoint"]),
    500: InternalServerErrorResponse.openapi_response(examples=["configuration"]),
    503: ServiceUnavailableResponse.openapi_response(
        examples=["backend service", "kubernetes api"]
    ),
}

provider_get_responses: dict[int | str, dict[str, Any]] = {
    200: ProviderResponse.openapi_response(),
    401: UnauthorizedResponse.openapi_response(examples=UNAUTHORIZED_OPENAPI_EXAMPLES),
    403: ForbiddenResponse.openapi_response(examples=["endpoint"]),
    404: NotFoundResponse.openapi_response(examples=["provider"]),
    500: InternalServerErrorResponse.openapi_response(examples=["configuration"]),
    503: ServiceUnavailableResponse.openapi_response(
        examples=["backend service", "kubernetes api"]
    ),
}


@router.get("/providers", responses=providers_list_responses)
@authorize(Action.LIST_PROVIDERS)
async def providers_endpoint_handler(
    request: Request,
    auth: Annotated[AuthTuple, Depends(get_auth_dependency())],
) -> ProvidersListResponse:
    """
    List all available providers grouped by API type.

    Routes to either Llama Stack or LangChain implementation based on feature flags.

    Parameters:
        request: The incoming HTTP request.
        auth: Authentication tuple from the auth dependency.

    Returns:
        ProvidersListResponse: Mapping from API type to list of providers.
    """
    check_configuration_loaded(configuration)

    feature_flags = configuration.configuration.feature_flags
    use_langchain = (
        feature_flags.use_langchain or "providers" in feature_flags.langchain_endpoints
    )

    if use_langchain:
        from app.endpoints.providers_langchain import providers_langchain

        logger.info("Routing /providers to LangChain implementation")
        return await providers_langchain()

    from app.endpoints.providers_llama_stack import providers_llama_stack

    logger.info("Routing /providers to Llama Stack implementation")
    return await providers_llama_stack()


@router.get("/providers/{provider_id}", responses=provider_get_responses)
@authorize(Action.GET_PROVIDER)
async def get_provider_endpoint_handler(
    request: Request,
    provider_id: str,
    auth: Annotated[AuthTuple, Depends(get_auth_dependency())],
) -> ProviderResponse:
    """
    Retrieve a single provider identified by its unique ID.

    Routes to either Llama Stack or LangChain implementation based on feature flags.

    Parameters:
        request: The incoming HTTP request.
        provider_id: Provider identification string.
        auth: Authentication tuple from the auth dependency.

    Returns:
        ProviderResponse: Provider details.
    """
    check_configuration_loaded(configuration)

    feature_flags = configuration.configuration.feature_flags
    use_langchain = (
        feature_flags.use_langchain or "providers" in feature_flags.langchain_endpoints
    )

    if use_langchain:
        from app.endpoints.providers_langchain import get_provider_langchain

        logger.info("Routing /providers/{provider_id} to LangChain implementation")
        return await get_provider_langchain(provider_id)

    from app.endpoints.providers_llama_stack import get_provider_llama_stack

    logger.info("Routing /providers/{provider_id} to Llama Stack implementation")
    return await get_provider_llama_stack(provider_id)
