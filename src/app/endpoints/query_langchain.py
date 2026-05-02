"""LangChain implementation for query endpoint."""

from fastapi import HTTPException, Request

from authentication.interface import AuthTuple
from log import get_logger
from models.requests import QueryRequest
from models.responses import (
    QueryResponse,
    ServiceUnavailableResponse,
)
from utils.mcp_headers import McpHeaders

logger = get_logger(__name__)


async def query_endpoint_handler_langchain(
    request: Request,
    query_request: QueryRequest,
    auth: AuthTuple,
    mcp_headers: McpHeaders,
) -> QueryResponse:
    """Handle query request using LangChain.

    Processes the query through LangChain's chat models instead of Llama Stack.

    Parameters:
        request: The incoming HTTP request.
        query_request: Request to the LLM.
        auth: Auth context tuple (user_id, username, skip_check, token).
        mcp_headers: Headers that should be passed to MCP servers.

    Returns:
        QueryResponse: Contains the conversation ID and the LLM-generated response.

    Raises:
        HTTPException: Various status codes for different error conditions.
    """
    # TODO: Implement full LangChain query pipeline
    # This is a placeholder to maintain dual-mode infrastructure.
    # Full implementation will include:
    # - Shield moderation via LangChain callbacks
    # - RAG context building via LangChain retrievers
    # - Chat model invocation (OpenAI, Azure, Watsonx via LangChain)
    # - Token tracking and quota management
    # - Conversation persistence
    # - Response formatting
    logger.error("LangChain query endpoint not yet implemented")
    error_response = ServiceUnavailableResponse(
        backend_name="LangChain",
        cause="LangChain query implementation is not yet complete. "
        "This is infrastructure-only migration commit. "
        "Full implementation will follow in another commit.",
    )
    raise HTTPException(**error_response.model_dump())
