"""Handler for REST API call to provide answer to query using Response API."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request

from authentication import get_auth_dependency
from authentication.interface import AuthTuple
from authorization.middleware import authorize
from configuration import configuration
from log import get_logger
from models.config import Action
from models.requests import QueryRequest
from models.responses import (
    UNAUTHORIZED_OPENAPI_EXAMPLES_WITH_MCP_OAUTH,
    ForbiddenResponse,
    InternalServerErrorResponse,
    NotFoundResponse,
    PromptTooLongResponse,
    QueryResponse,
    QuotaExceededResponse,
    ServiceUnavailableResponse,
    UnauthorizedResponse,
    UnprocessableEntityResponse,
)
from utils.endpoints import check_configuration_loaded
from utils.mcp_headers import McpHeaders, mcp_headers_dependency

logger = get_logger(__name__)
router = APIRouter(tags=["query"])

query_response: dict[int | str, dict[str, Any]] = {
    200: QueryResponse.openapi_response(),
    401: UnauthorizedResponse.openapi_response(
        examples=UNAUTHORIZED_OPENAPI_EXAMPLES_WITH_MCP_OAUTH
    ),
    403: ForbiddenResponse.openapi_response(
        examples=["endpoint", "conversation read", "model override"]
    ),
    404: NotFoundResponse.openapi_response(
        examples=["conversation", "model", "provider"]
    ),
    413: PromptTooLongResponse.openapi_response(examples=["context window exceeded"]),
    422: UnprocessableEntityResponse.openapi_response(),
    429: QuotaExceededResponse.openapi_response(),
    500: InternalServerErrorResponse.openapi_response(examples=["configuration"]),
    503: ServiceUnavailableResponse.openapi_response(
        examples=["backend service", "kubernetes api"]
    ),
}


@router.post("/query", responses=query_response, summary="Query Endpoint Handler")
@authorize(Action.QUERY)
async def query_endpoint_handler(
    request: Request,
    query_request: QueryRequest,
    auth: Annotated[AuthTuple, Depends(get_auth_dependency())],
    mcp_headers: McpHeaders = Depends(mcp_headers_dependency),
) -> QueryResponse:
    """Handle request to the /query endpoint.

    Routes to either Llama Stack or LangChain implementation based on feature flags.

    Parameters:
        request: The incoming HTTP request (used by middleware).
        query_request: Request to the LLM.
        auth: Auth context tuple resolved from the authentication dependency.
        mcp_headers: Headers that should be passed to MCP servers.

    Returns:
        QueryResponse: Contains the conversation ID and the LLM-generated response.

    Raises:
        HTTPException: Various status codes for different error conditions.
    """
    check_configuration_loaded(configuration)

    feature_flags = configuration.configuration.feature_flags
    use_langchain = (
        feature_flags.use_langchain or "query" in feature_flags.langchain_endpoints
    )

    if use_langchain:
        from app.endpoints.query_langchain import query_endpoint_handler_langchain

        logger.info("Routing POST /query to LangChain implementation")
        return await query_endpoint_handler_langchain(
            request, query_request, auth, mcp_headers
        )

    from app.endpoints.query_llama_stack import query_endpoint_handler_llama_stack

    logger.info("Routing POST /query to Llama Stack implementation")
    return await query_endpoint_handler_llama_stack(
        request, query_request, auth, mcp_headers
    )
