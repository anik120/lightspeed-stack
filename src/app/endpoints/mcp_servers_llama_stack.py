"""Llama Stack implementation for MCP servers endpoints."""

from fastapi import HTTPException
from llama_stack_client import APIConnectionError

from client import AsyncLlamaStackClientHolder
from configuration import configuration
from log import get_logger
from models.config import ModelContextProtocolServer
from models.requests import MCPServerRegistrationRequest
from models.responses import (
    ConflictResponse,
    ForbiddenResponse,
    InternalServerErrorResponse,
    MCPServerDeleteResponse,
    MCPServerInfo,
    MCPServerListResponse,
    MCPServerRegistrationResponse,
    NotFoundResponse,
    ServiceUnavailableResponse,
)

logger = get_logger(__name__)


async def register_mcp_server_llama_stack(
    body: MCPServerRegistrationRequest,
) -> MCPServerRegistrationResponse:
    """Register an MCP server dynamically at runtime using Llama Stack.

    Adds the MCP server to the runtime configuration and registers it
    as a toolgroup with Llama Stack so it becomes available for queries.

    Parameters:
        body: MCP server registration parameters.

    Returns:
        MCPServerRegistrationResponse: Details of the newly registered server.

    Raises:
        HTTPException: On duplicate name, Llama Stack connection error, or
        registration failure.
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

    try:
        client = AsyncLlamaStackClientHolder().get_client()
        await client.toolgroups.register(  # pyright: ignore[reportDeprecated]
            toolgroup_id=mcp_server.name,
            provider_id=mcp_server.provider_id,
            mcp_endpoint={"uri": mcp_server.url},
        )
    except APIConnectionError as e:
        configuration.remove_mcp_server(body.name)
        logger.error("Failed to register MCP server with Llama Stack: %s", e)
        response = ServiceUnavailableResponse(backend_name="Llama Stack", cause=str(e))
        raise HTTPException(**response.model_dump()) from e
    except Exception as e:  # pylint: disable=broad-exception-caught
        configuration.remove_mcp_server(body.name)
        logger.error("Failed to register MCP toolgroup: %s", e)
        error_response = InternalServerErrorResponse(
            response="Failed to register MCP server",
            cause=str(e),
        )
        raise HTTPException(**error_response.model_dump()) from e

    logger.info("Dynamically registered MCP server: %s at %s", body.name, body.url)

    return MCPServerRegistrationResponse(
        name=mcp_server.name,
        url=mcp_server.url,
        provider_id=mcp_server.provider_id,
        message=f"MCP server '{mcp_server.name}' registered successfully",
    )


async def list_mcp_servers() -> MCPServerListResponse:
    """List all registered MCP servers.

    Returns both statically configured (from YAML) and dynamically
    registered (via API) MCP servers. This operation is backend-agnostic
    and just reads from configuration.

    Returns:
        MCPServerListResponse: List of all registered MCP servers with source info.
    """
    servers = []
    for mcp in configuration.mcp_servers:
        source = "api" if configuration.is_dynamic_mcp_server(mcp.name) else "config"
        servers.append(
            MCPServerInfo(
                name=mcp.name,
                url=mcp.url,
                provider_id=mcp.provider_id,
                source=source,
            )
        )

    return MCPServerListResponse(servers=servers)


async def delete_mcp_server_llama_stack(name: str) -> MCPServerDeleteResponse:
    """Unregister a dynamically registered MCP server using Llama Stack.

    Removes the MCP server from the runtime configuration and unregisters
    its toolgroup from Llama Stack. Only servers registered via the API
    can be deleted; statically configured servers cannot be removed.

    Parameters:
        name: MCP server name.

    Returns:
        MCPServerDeleteResponse: Confirmation of the deletion.

    Raises:
        HTTPException: If the server is not found, is statically configured, or
        Llama Stack unregistration fails.
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
        client = AsyncLlamaStackClientHolder().get_client()
        await client.toolgroups.unregister(  # pyright: ignore[reportDeprecated]
            toolgroup_id=name
        )
    except APIConnectionError as e:
        logger.error("Failed to unregister MCP toolgroup from Llama Stack: %s", e)
        svc_response = ServiceUnavailableResponse(
            backend_name="Llama Stack", cause=str(e)
        )
        raise HTTPException(**svc_response.model_dump()) from e
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            "Llama Stack toolgroup unregister failed for '%s', "
            "proceeding with local removal: %s",
            name,
            e,
        )

    try:
        configuration.remove_mcp_server(name)
    except ValueError as e:
        logger.error("Failed to remove MCP server from configuration: %s", e)
        response = NotFoundResponse(resource="MCP server", resource_id=name)
        raise HTTPException(**response.model_dump()) from e

    logger.info("Dynamically unregistered MCP server: %s", name)

    return MCPServerDeleteResponse(
        name=name,
        message=f"MCP server '{name}' unregistered successfully",
    )
