"""Handler for REST API calls to list and retrieve available RAGs."""

from typing import Annotated, Any

from fastapi import APIRouter, Request
from fastapi.params import Depends

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
    RAGInfoResponse,
    RAGListResponse,
    ServiceUnavailableResponse,
    UnauthorizedResponse,
)
from utils.endpoints import check_configuration_loaded

logger = get_logger(__name__)
router = APIRouter(tags=["rags"])


rags_responses: dict[int | str, dict[str, Any]] = {
    200: RAGListResponse.openapi_response(),
    401: UnauthorizedResponse.openapi_response(examples=UNAUTHORIZED_OPENAPI_EXAMPLES),
    403: ForbiddenResponse.openapi_response(examples=["endpoint"]),
    500: InternalServerErrorResponse.openapi_response(examples=["configuration"]),
    503: ServiceUnavailableResponse.openapi_response(
        examples=["backend service", "kubernetes api"]
    ),
}

rag_responses: dict[int | str, dict[str, Any]] = {
    200: RAGInfoResponse.openapi_response(),
    401: UnauthorizedResponse.openapi_response(examples=UNAUTHORIZED_OPENAPI_EXAMPLES),
    403: ForbiddenResponse.openapi_response(examples=["endpoint"]),
    404: NotFoundResponse.openapi_response(examples=["rag"]),
    500: InternalServerErrorResponse.openapi_response(examples=["configuration"]),
    503: ServiceUnavailableResponse.openapi_response(
        examples=["backend service", "kubernetes api"]
    ),
}


@router.get("/rags", responses=rags_responses)
@authorize(Action.LIST_RAGS)
async def rags_endpoint_handler(
    request: Request,
    auth: Annotated[AuthTuple, Depends(get_auth_dependency())],
) -> RAGListResponse:
    """
    List all available RAGs.

    Routes to either Llama Stack or LangChain implementation based on feature flags.

    Parameters:
        request: The incoming HTTP request.
        auth: Authentication tuple from the auth dependency.

    Returns:
        RAGListResponse: List of RAG identifiers.
    """
    # Used only by the middleware
    _ = auth
    _ = request

    check_configuration_loaded(configuration)

    feature_flags = configuration.configuration.feature_flags
    use_langchain = (
        feature_flags.use_langchain or "rags" in feature_flags.langchain_endpoints
    )

    if use_langchain:
        from app.endpoints.rags_langchain import list_rags_langchain

        logger.info("Routing /rags to LangChain implementation")
        return await list_rags_langchain()

    from app.endpoints.rags_llama_stack import list_rags_llama_stack

    logger.info("Routing /rags to Llama Stack implementation")
    return await list_rags_llama_stack()


@router.get("/rags/{rag_id}", responses=rag_responses)
@authorize(Action.GET_RAG)
async def get_rag_endpoint_handler(
    request: Request,
    rag_id: str,
    auth: Annotated[AuthTuple, Depends(get_auth_dependency())],
) -> RAGInfoResponse:
    """Retrieve a single RAG identified by its unique ID.

    Routes to either Llama Stack or LangChain implementation based on feature flags.

    Parameters:
        request: The incoming HTTP request.
        rag_id: The RAG identifier.
        auth: Authentication tuple from the auth dependency.

    Returns:
        RAGInfoResponse: A single RAG's details.
    """
    # Used only by the middleware
    _ = auth
    _ = request

    check_configuration_loaded(configuration)

    feature_flags = configuration.configuration.feature_flags
    use_langchain = (
        feature_flags.use_langchain or "rags" in feature_flags.langchain_endpoints
    )

    if use_langchain:
        from app.endpoints.rags_langchain import get_rag_langchain

        logger.info("Routing /rags/{rag_id} to LangChain implementation")
        return await get_rag_langchain(rag_id)

    from app.endpoints.rags_llama_stack import get_rag_llama_stack

    logger.info("Routing /rags/{rag_id} to Llama Stack implementation")
    return await get_rag_llama_stack(rag_id)
