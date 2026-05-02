"""LangChain implementation for tools endpoint."""

from typing import Any

from fastapi import HTTPException

from configuration import configuration
from log import get_logger
from models.responses import (
    ServiceUnavailableResponse,
    ToolsResponse,
)
from utils.mcp_headers import find_unresolved_auth_headers
from utils.tool_formatter import format_tools_list

logger = get_logger(__name__)


async def tools_langchain(
    complete_mcp_headers: dict[str, dict[str, str]],
) -> ToolsResponse:
    """List all available tools from MCP servers using LangChain.

    Discovers tools directly from configured MCP servers (both statically configured
    and dynamically registered via API). LangChain connects to each MCP server individually 
    to discover available tools.

    Parameters:
        complete_mcp_headers: Headers to pass to MCP servers, keyed by server name.

    Returns:
        ToolsResponse: Consolidated list of tools with metadata.

    Raises:
        HTTPException: If unable to connect to MCP servers or tool discovery fails.
    """
    consolidated_tools = []

    # Iterate over ALL configured MCP servers (static + dynamic)
    # configuration.mcp_servers includes both:
    # - Statically configured servers from lightspeed-stack.yaml
    # - Dynamically registered servers via POST /mcp-servers
    if not configuration.mcp_servers:
        logger.info("No MCP servers configured, returning empty tools list")
        return ToolsResponse(tools=[])

    for mcp_server in configuration.mcp_servers:
        headers = complete_mcp_headers.get(mcp_server.name, {})

        # Check if all required auth headers are resolved
        if mcp_server.authorization_headers:
            unresolved = find_unresolved_auth_headers(
                mcp_server.authorization_headers, headers
            )
            if unresolved:
                logger.warning(
                    "Skipping MCP server %s: required %d auth headers "
                    "but only resolved %d",
                    mcp_server.name,
                    len(mcp_server.authorization_headers),
                    len(mcp_server.authorization_headers) - len(unresolved),
                )
                continue

        try:
            # Discover tools from this MCP server
            tools_from_server = await _discover_tools_from_mcp_server(
                mcp_server.name, mcp_server.url, headers
            )

            # Add server metadata to each tool
            for tool_dict in tools_from_server:
                tool_dict["server_source"] = mcp_server.url
                tool_dict["provider_id"] = mcp_server.provider_id
                tool_dict["type"] = "tool"
                consolidated_tools.append(tool_dict)

            logger.debug(
                "Retrieved %d tools from MCP server %s (source: %s)",
                len(tools_from_server),
                mcp_server.name,
                mcp_server.url,
            )

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to discover tools from MCP server %s: %s", mcp_server.name, e)
            response = ServiceUnavailableResponse(
                backend_name=f"MCP server {mcp_server.name}", cause=str(e)
            )
            raise HTTPException(**response.model_dump()) from e

    logger.info(
        "Retrieved total of %d tools from %d MCP servers (static + dynamic)",
        len(consolidated_tools),
        len(configuration.mcp_servers),
    )

    # Format tools with structured description parsing
    formatted_tools = format_tools_list(consolidated_tools)

    return ToolsResponse(tools=formatted_tools)


async def _discover_tools_from_mcp_server(
    server_name: str, server_url: str, headers: dict[str, str]
) -> list[dict[str, Any]]:
    """Discover tools from a single MCP server using LangChain.

    Parameters:
        server_name: Name of the MCP server.
        server_url: URL of the MCP server.
        headers: HTTP headers to include in the request.

    Returns:
        List of tool dictionaries in our endpoint's format.
    """
    # TODO: Implement actual MCP tool discovery using LangChain
    # For now, return empty list as placeholder to keep commit scoped
    # to infrastrcture (routing) migration only.
    # [anik120] will follow up with "LangChain MCP integration" commit.
    logger.warning(
        "MCP tool discovery not yet implemented for LangChain backend. "
        "Server: %s, URL: %s",
        server_name,
        server_url,
    )
    _ = headers  # Unused for now
    return []
