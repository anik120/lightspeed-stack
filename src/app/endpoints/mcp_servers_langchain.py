"""LangChain implementation for MCP servers endpoints."""

from fastapi import HTTPException

from configuration import configuration
from log import get_logger
from models.config import ModelContextProtocolServer
from models.requests import MCPServerRegistrationRequest
from models.responses import (
    ConflictResponse,
    ForbiddenResponse,
    MCPServerDeleteResponse,
    MCPServerRegistrationResponse,
    NotFoundResponse,
)

logger = get_logger(__name__)


async def register_mcp_server_langchain(
    body: MCPServerRegistrationRequest,
) -> MCPServerRegistrationResponse:
    """Register an MCP server dynamically at runtime using LangChain.

    Adds the MCP server to the runtime configuration. Unlike Llama Stack,
    LangChain doesn't have a separate toolgroup registration API - MCP tools
    are discovered and used directly when needed.

    Parameters:
        body: MCP server registration parameters.

    Returns:
        MCPServerRegistrationResponse: Details of the newly registered server.

    Raises:
        HTTPException: On duplicate name.
    """
    mcp_server = ModelContextProtocolServer(
        name=body.name,
        url=body.url,
        provider_id=body.provider_id,
        authorization_headers=body.authorization_headers or {},
        headers=body.headers or [],
        timeout=body.timeout,
    )

    try:
        configuration.add_mcp_server(mcp_server)
    except ValueError as e:
        response = ConflictResponse(resource="MCP server", resource_id=body.name)
        raise HTTPException(**response.model_dump()) from e

    logger.info(
        "Dynamically registered MCP server (LangChain): %s at %s", body.name, body.url
    )

    return MCPServerRegistrationResponse(
        name=mcp_server.name,
        url=mcp_server.url,
        provider_id=mcp_server.provider_id,
        message=f"MCP server '{mcp_server.name}' registered successfully",
    )


async def delete_mcp_server_langchain(name: str) -> MCPServerDeleteResponse:
    """Unregister a dynamically registered MCP server using LangChain.

    Removes the MCP server from the runtime configuration. Unlike Llama Stack,
    LangChain doesn't have a separate toolgroup unregistration API - MCP tools
    are simply no longer available once the configuration is removed.

    Only servers registered via the API can be deleted; statically configured
    servers cannot be removed.

    Parameters:
        name: MCP server name.

    Returns:
        MCPServerDeleteResponse: Confirmation of the deletion.

    Raises:
        HTTPException: If the server is not found or is statically configured.
    """
    if not configuration.is_dynamic_mcp_server(name):
        found = any(s.name == name for s in configuration.mcp_servers)
        if found:
            response = ForbiddenResponse(
                response="Cannot delete statically configured MCP server",
                cause=f"MCP server '{name}' was configured in lightspeed-stack.yaml "
                "and cannot be removed via the API.",
            )
        else:
            response = NotFoundResponse(resource="MCP server", resource_id=name)
        raise HTTPException(**response.model_dump())

    try:
        configuration.remove_mcp_server(name)
    except ValueError as e:
        logger.error("Failed to remove MCP server from configuration: %s", e)
        response = NotFoundResponse(resource="MCP server", resource_id=name)
        raise HTTPException(**response.model_dump()) from e

    logger.info("Dynamically unregistered MCP server (LangChain): %s", name)

    return MCPServerDeleteResponse(
        name=name,
        message=f"MCP server '{name}' unregistered successfully",
    )
