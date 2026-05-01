"""Handler for REST API calls to dynamically manage MCP servers."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request, status

from authentication import get_auth_dependency
from authentication.interface import AuthTuple
from authorization.middleware import authorize
from configuration import configuration
from log import get_logger
from models.config import Action
from models.requests import MCPServerRegistrationRequest
from models.responses import (
    UNAUTHORIZED_OPENAPI_EXAMPLES,
    ConflictResponse,
    ForbiddenResponse,
    InternalServerErrorResponse,
    MCPServerDeleteResponse,
    MCPServerListResponse,
    MCPServerRegistrationResponse,
    NotFoundResponse,
    ServiceUnavailableResponse,
    UnauthorizedResponse,
)
from utils.endpoints import check_configuration_loaded

logger = get_logger(__name__)
router = APIRouter(tags=["mcp-servers"])


register_responses: dict[int | str, dict[str, Any]] = {
    201: MCPServerRegistrationResponse.openapi_response(),
    401: UnauthorizedResponse.openapi_response(examples=UNAUTHORIZED_OPENAPI_EXAMPLES),
    403: ForbiddenResponse.openapi_response(examples=["endpoint"]),
    409: ConflictResponse.openapi_response(examples=["mcp server"]),
    500: InternalServerErrorResponse.openapi_response(examples=["configuration"]),
    503: ServiceUnavailableResponse.openapi_response(
        examples=["backend service", "kubernetes api"]
    ),
}


@router.post(
    "/mcp-servers",
    responses=register_responses,
    status_code=status.HTTP_201_CREATED,
)
@authorize(Action.REGISTER_MCP_SERVER)
async def register_mcp_server_handler(
    request: Request,
    body: MCPServerRegistrationRequest,
    auth: Annotated[AuthTuple, Depends(get_auth_dependency())],
) -> MCPServerRegistrationResponse:
    """Register an MCP server dynamically at runtime.

    Routes to either Llama Stack or LangChain implementation based on feature flags.

    Parameters:
        request: The incoming HTTP request (used by middleware).
        body: MCP server registration parameters.
        auth: Authentication tuple from the auth dependency (used by middleware).

    Returns:
        MCPServerRegistrationResponse: Details of the newly registered server.
    """
    _ = auth
    _ = request

    check_configuration_loaded(configuration)

    feature_flags = configuration.configuration.feature_flags
    use_langchain = (
        feature_flags.use_langchain
        or "mcp_servers" in feature_flags.langchain_endpoints
    )

    if use_langchain:
        from app.endpoints.mcp_servers_langchain import register_mcp_server_langchain

        logger.info("Routing POST /mcp-servers to LangChain implementation")
        return await register_mcp_server_langchain(body)

    from app.endpoints.mcp_servers_llama_stack import register_mcp_server_llama_stack

    logger.info("Routing POST /mcp-servers to Llama Stack implementation")
    return await register_mcp_server_llama_stack(body)


list_responses: dict[int | str, dict[str, Any]] = {
    200: MCPServerListResponse.openapi_response(),
    401: UnauthorizedResponse.openapi_response(examples=UNAUTHORIZED_OPENAPI_EXAMPLES),
    403: ForbiddenResponse.openapi_response(examples=["endpoint"]),
    500: InternalServerErrorResponse.openapi_response(examples=["configuration"]),
    503: ServiceUnavailableResponse.openapi_response(examples=["kubernetes api"]),
}


@router.get("/mcp-servers", responses=list_responses)
@authorize(Action.LIST_MCP_SERVERS)
async def list_mcp_servers_handler(
    request: Request,
    auth: Annotated[AuthTuple, Depends(get_auth_dependency())],
) -> MCPServerListResponse:
    """List all registered MCP servers.

    Returns both statically configured (from YAML) and dynamically
    registered (via API) MCP servers. This operation is backend-agnostic.

    Parameters:
        request: The incoming HTTP request (used by middleware).
        auth: Authentication tuple from the auth dependency (used by middleware).

    Returns:
        MCPServerListResponse: List of all registered MCP servers with source info.
    """
    _ = auth
    _ = request

    check_configuration_loaded(configuration)

    # List operation is backend-agnostic - both implementations use the same code
    from app.endpoints.mcp_servers_llama_stack import list_mcp_servers

    return await list_mcp_servers()


delete_responses: dict[int | str, dict[str, Any]] = {
    200: MCPServerDeleteResponse.openapi_response(),
    401: UnauthorizedResponse.openapi_response(examples=UNAUTHORIZED_OPENAPI_EXAMPLES),
    403: ForbiddenResponse.openapi_response(examples=["endpoint"]),
    404: NotFoundResponse.openapi_response(examples=["mcp server"]),
    500: InternalServerErrorResponse.openapi_response(examples=["configuration"]),
    503: ServiceUnavailableResponse.openapi_response(
        examples=["backend service", "kubernetes api"]
    ),
}


@router.delete("/mcp-servers/{name}", responses=delete_responses)
@authorize(Action.DELETE_MCP_SERVER)
async def delete_mcp_server_handler(
    request: Request,
    name: str,
    auth: Annotated[AuthTuple, Depends(get_auth_dependency())],
) -> MCPServerDeleteResponse:
    """Unregister a dynamically registered MCP server.

    Routes to either Llama Stack or LangChain implementation based on feature flags.

    Parameters:
        request: The incoming HTTP request (used by middleware).
        name: MCP server name.
        auth: Authentication tuple from the auth dependency (used by middleware).

    Returns:
        MCPServerDeleteResponse: Confirmation of the deletion.
    """
    _ = auth
    _ = request

    check_configuration_loaded(configuration)

    feature_flags = configuration.configuration.feature_flags
    use_langchain = (
        feature_flags.use_langchain
        or "mcp_servers" in feature_flags.langchain_endpoints
    )

    if use_langchain:
        from app.endpoints.mcp_servers_langchain import delete_mcp_server_langchain

        logger.info("Routing DELETE /mcp-servers/{name} to LangChain implementation")
        return await delete_mcp_server_langchain(name)

    from app.endpoints.mcp_servers_llama_stack import delete_mcp_server_llama_stack

    logger.info("Routing DELETE /mcp-servers/{name} to Llama Stack implementation")
    return await delete_mcp_server_llama_stack(name)
