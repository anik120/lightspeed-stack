"""LangChain implementation for query endpoint."""

import datetime

from fastapi import HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from authentication.interface import AuthTuple
from configuration import configuration
from langchain_providers import LLMProviderRegistry
from log import get_logger
from models.config import Action
from models.requests import QueryRequest
from models.responses import (
    QueryResponse,
    ServiceUnavailableResponse,
)
from utils.mcp_headers import McpHeaders
from utils.mcp_oauth_probe import check_mcp_auth
from utils.query import (
    consume_query_tokens,
    store_query_results,
    validate_attachments_metadata,
    validate_model_provider_override,
)
from utils.quota import check_tokens_available, get_available_quotas
from utils.shields import validate_shield_ids_override
from utils.suid import get_suid, normalize_conversation_id
from utils.token_counter import TokenCounter
from utils.types import TurnSummary

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
    started_at = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    user_id, _, _skip_userid_check, token = auth

    # Check MCP Auth
    await check_mcp_auth(configuration, mcp_headers, token, request.headers)

    # Check token availability
    check_tokens_available(configuration.quota_limiters, user_id)

    # Enforce RBAC: optionally disallow overriding model/provider in requests
    validate_model_provider_override(
        query_request.model, query_request.provider, request.state.authorized_actions
    )

    # Validate shield_ids override if provided
    validate_shield_ids_override(query_request, configuration)

    # Validate attachments if provided
    if query_request.attachments:
        validate_attachments_metadata(query_request.attachments)

    # Get LangChain configuration
    langchain_config = configuration.configuration.langchain
    if not langchain_config:
        error_response = ServiceUnavailableResponse(
            backend_name="LangChain",
            cause="LangChain not configured. Add langchain section to lightspeed-stack.yaml",
        )
        raise HTTPException(**error_response.model_dump())

    # Initialize provider registry
    try:
        registry = LLMProviderRegistry()
        await registry.initialize(langchain_config)
    except Exception as e:
        logger.error("Failed to initialize LangChain provider registry: %s", e)
        error_response = ServiceUnavailableResponse(
            backend_name="LangChain", cause=f"Provider initialization failed: {e}"
        )
        raise HTTPException(**error_response.model_dump()) from e

    # Determine which model and provider to use
    model_id = query_request.model or langchain_config.default_model
    provider_name = query_request.provider or langchain_config.default_provider

    # Get chat model from registry
    try:
        chat_model = await registry.get_chat_model(
            model_id=model_id, provider=provider_name
        )
    except Exception as e:
        logger.error("Failed to get chat model: %s", e)
        error_response = ServiceUnavailableResponse(
            backend_name="LangChain", cause=f"Failed to get model '{model_id}': {e}"
        )
        raise HTTPException(**error_response.model_dump()) from e

    # Build messages for chat model
    # TODO: Add conversation history, RAG context, system prompts
    messages = [
        HumanMessage(content=query_request.query),
    ]

    # Invoke chat model
    try:
        logger.info(
            "Invoking LangChain chat model: provider=%s, model=%s",
            provider_name,
            model_id,
        )
        response = await chat_model.ainvoke(messages)
    except Exception as e:
        logger.error("Chat model invocation failed: %s", e)
        error_response = ServiceUnavailableResponse(
            backend_name="LangChain", cause=f"Model invocation failed: {e}"
        )
        raise HTTPException(**error_response.model_dump()) from e

    # Extract response text
    if isinstance(response, AIMessage):
        # LangChain AIMessage.content can be str or list of content blocks
        if isinstance(response.content, str):
            response_text = response.content
        elif isinstance(response.content, list):
            # Join multiple content blocks
            response_text = " ".join(
                str(item) if isinstance(item, str) else str(item)
                for item in response.content
            )
        else:
            response_text = str(response.content)
    else:
        response_text = str(response)

    logger.info(
        "Received response from LangChain model, length: %d", len(response_text)
    )

    # Count tokens (approximate - word count * 1.3)
    # TODO: Use proper tokenization (e.g., tiktoken) for accurate counts
    input_text = query_request.query
    input_tokens = int(len(input_text.split()) * 1.3)
    output_tokens = (
        int(len(response_text.split()) * 1.3) if isinstance(response_text, str) else 0
    )

    token_usage = TokenCounter(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    # Build turn summary
    turn_summary = TurnSummary(
        id="langchain-turn",  # TODO: Generate proper turn ID
        llm_response=response_text,
        token_usage=token_usage,
        rag_chunks=[],  # TODO: Add RAG support
        referenced_documents=[],  # TODO: Add RAG support
        tool_calls=[],  # TODO: Add tool calling support
        tool_results=[],  # TODO: Add tool calling support
    )

    # Consume tokens
    logger.info("Consuming tokens")
    consume_query_tokens(
        user_id=user_id,
        model_id=model_id,
        token_usage=turn_summary.token_usage,
    )

    # Get available quotas
    logger.info("Getting available quotas")
    available_quotas = get_available_quotas(
        quota_limiters=configuration.quota_limiters, user_id=user_id
    )

    # Generate conversation ID (or use existing)
    if query_request.conversation_id:
        conversation_id = normalize_conversation_id(query_request.conversation_id)
    else:
        # Create new conversation with proper UUID
        conversation_id = get_suid()

    completed_at = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Store query results
    logger.info("Storing query results")
    store_query_results(
        user_id=user_id,
        conversation_id=conversation_id,
        model=model_id,
        started_at=started_at,
        completed_at=completed_at,
        summary=turn_summary,
        query=query_request.query,
        attachments=query_request.attachments,
        skip_userid_check=_skip_userid_check,
        topic_summary=None,  # TODO: Generate topic summaries
    )

    logger.info("Building final response")
    return QueryResponse(
        conversation_id=conversation_id,
        response=turn_summary.llm_response,
        tool_calls=turn_summary.tool_calls,
        tool_results=turn_summary.tool_results,
        rag_chunks=turn_summary.rag_chunks,
        referenced_documents=turn_summary.referenced_documents,
        truncated=False,
        input_tokens=turn_summary.token_usage.input_tokens,
        output_tokens=turn_summary.token_usage.output_tokens,
        available_quotas=available_quotas,
    )
