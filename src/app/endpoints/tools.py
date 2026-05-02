"""Handler for REST API call to list available tools from MCP servers."""

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
    ServiceUnavailableResponse,
    ToolsResponse,
    UnauthorizedResponse,
)
from utils.endpoints import check_configuration_loaded
from utils.mcp_headers import (
    McpHeaders,
    build_mcp_headers,
    mcp_headers_dependency,
)
from utils.mcp_oauth_probe import check_mcp_auth

logger = get_logger(__name__)
router = APIRouter(tags=["tools"])


tools_responses: dict[int | str, dict[str, Any]] = {
    200: ToolsResponse.openapi_response(),
    401: UnauthorizedResponse.openapi_response(examples=UNAUTHORIZED_OPENAPI_EXAMPLES),
    403: ForbiddenResponse.openapi_response(examples=["endpoint"]),
    500: InternalServerErrorResponse.openapi_response(examples=["configuration"]),
    503: ServiceUnavailableResponse.openapi_response(
        examples=["backend service", "kubernetes api"]
    ),
}


@router.get("/tools", responses=tools_responses)
@authorize(Action.GET_TOOLS)
async def tools_endpoint_handler(
    request: Request,
    auth: Annotated[AuthTuple, Depends(get_auth_dependency())],
    mcp_headers: McpHeaders = Depends(mcp_headers_dependency),
) -> ToolsResponse:
    """List all available tools from MCP servers and built-in toolgroups.

    Routes to either Llama Stack or LangChain implementation based on feature flags.

    Parameters:
        request: The incoming HTTP request (used by middleware).
        auth: Authentication tuple from the auth dependency (used by middleware).
        mcp_headers: Headers that should be passed to MCP servers.

    Returns:
        ToolsResponse: Consolidated list of tools with metadata.
    """
    _, _, _, token = auth

    check_configuration_loaded(configuration)

    complete_mcp_headers = build_mcp_headers(
        configuration, mcp_headers, request.headers, token
    )

    # Check MCP Auth
    await check_mcp_auth(configuration, mcp_headers, token, request.headers)

    feature_flags = configuration.configuration.feature_flags
    use_langchain = (
        feature_flags.use_langchain or "tools" in feature_flags.langchain_endpoints
    )

    if use_langchain:
        from app.endpoints.tools_langchain import tools_langchain

        logger.info("Routing GET /tools to LangChain implementation")
        return await tools_langchain(complete_mcp_headers)

    from app.endpoints.tools_llama_stack import tools_llama_stack

    logger.info("Routing GET /tools to Llama Stack implementation")
    return await tools_llama_stack(complete_mcp_headers)
