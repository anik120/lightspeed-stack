# pylint: disable=redefined-outer-name

"""Unit tests for LangChain query endpoint."""

import pytest
from fastapi import HTTPException, Request
from langchain_core.messages import AIMessage
from pytest_mock import MockerFixture

from app.endpoints.query_langchain import query_endpoint_handler_langchain
from models.config import Action, LangChainConfiguration, LangChainProviderConfig
from models.requests import Attachment, QueryRequest
from models.responses import QueryResponse

MOCK_AUTH = (
    "00000001-0001-0001-0001-000000000001",
    "mock_username",
    False,
    "mock_token",
)


@pytest.fixture(name="dummy_request")
def create_dummy_request() -> Request:
    """Create dummy request fixture for testing."""
    request = Request(scope={"type": "http", "headers": []})
    request.state.authorized_actions = {Action.QUERY}
    return request


@pytest.fixture(name="langchain_config")
def create_langchain_config() -> LangChainConfiguration:
    """Create LangChain configuration fixture."""
    from pydantic import SecretStr

    return LangChainConfiguration(
        providers={
            "openai": LangChainProviderConfig(
                api_key=SecretStr("test-key"),
                api_base=None,
                api_version=None,
                timeout=60,
                max_retries=3,
                models=["gpt-4", "gpt-3.5-turbo"],
            )
        },
        default_provider="openai",
        default_model="gpt-4",
        enable_streaming=True,
        enable_tracing=False,
    )


@pytest.mark.asyncio
async def test_successful_query_no_conversation(
    dummy_request: Request,
    langchain_config: LangChainConfiguration,
    mocker: MockerFixture,
) -> None:
    """Test successful query without existing conversation."""
    query_request = QueryRequest(
        query="What is Kubernetes?"
    )  # pyright: ignore[reportCallIssue]

    # Mock configuration
    mock_config = mocker.Mock()
    mock_config.configuration.langchain = langchain_config
    mock_config.quota_limiters = []
    mocker.patch("app.endpoints.query_langchain.configuration", mock_config)

    # Mock auth and validation
    mocker.patch("app.endpoints.query_langchain.check_mcp_auth", new=mocker.AsyncMock())
    mocker.patch("app.endpoints.query_langchain.check_tokens_available")
    mocker.patch("app.endpoints.query_langchain.validate_model_provider_override")
    mocker.patch("app.endpoints.query_langchain.validate_shield_ids_override")

    # Mock provider registry and chat model
    mock_chat_model = mocker.AsyncMock()
    mock_ai_message = AIMessage(
        content="Kubernetes is a container orchestration platform"
    )
    mock_chat_model.ainvoke = mocker.AsyncMock(return_value=mock_ai_message)

    mock_registry = mocker.AsyncMock()
    mock_registry.initialize = mocker.AsyncMock()
    mock_registry.get_chat_model = mocker.AsyncMock(return_value=mock_chat_model)

    mocker.patch(
        "app.endpoints.query_langchain.LLMProviderRegistry",
        return_value=mock_registry,
    )

    # Mock token counting and storage
    mocker.patch("app.endpoints.query_langchain.consume_query_tokens")
    mocker.patch("app.endpoints.query_langchain.get_available_quotas", return_value={})
    mocker.patch("app.endpoints.query_langchain.store_query_results")

    response = await query_endpoint_handler_langchain(
        request=dummy_request,
        query_request=query_request,
        auth=MOCK_AUTH,
        mcp_headers={},
    )

    assert isinstance(response, QueryResponse)
    assert response.response == "Kubernetes is a container orchestration platform"
    assert response.input_tokens > 0
    assert response.output_tokens > 0


@pytest.mark.asyncio
async def test_langchain_not_configured(
    dummy_request: Request,
    mocker: MockerFixture,
) -> None:
    """Test query when LangChain is not configured."""
    query_request = QueryRequest(
        query="What is Kubernetes?"
    )  # pyright: ignore[reportCallIssue]

    # Mock configuration without langchain
    mock_config = mocker.Mock()
    mock_config.configuration.langchain = None
    mock_config.quota_limiters = []
    mocker.patch("app.endpoints.query_langchain.configuration", mock_config)

    mocker.patch("app.endpoints.query_langchain.check_mcp_auth", new=mocker.AsyncMock())
    mocker.patch("app.endpoints.query_langchain.check_tokens_available")
    mocker.patch("app.endpoints.query_langchain.validate_model_provider_override")
    mocker.patch("app.endpoints.query_langchain.validate_shield_ids_override")

    with pytest.raises(HTTPException) as exc_info:
        await query_endpoint_handler_langchain(
            request=dummy_request,
            query_request=query_request,
            auth=MOCK_AUTH,
            mcp_headers={},
        )

    assert exc_info.value.status_code == 503
    assert "LangChain not configured" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_provider_initialization_failure(
    dummy_request: Request,
    langchain_config: LangChainConfiguration,
    mocker: MockerFixture,
) -> None:
    """Test query when provider registry initialization fails."""
    query_request = QueryRequest(
        query="What is Kubernetes?"
    )  # pyright: ignore[reportCallIssue]

    mock_config = mocker.Mock()
    mock_config.configuration.langchain = langchain_config
    mock_config.quota_limiters = []
    mocker.patch("app.endpoints.query_langchain.configuration", mock_config)

    mocker.patch("app.endpoints.query_langchain.check_mcp_auth", new=mocker.AsyncMock())
    mocker.patch("app.endpoints.query_langchain.check_tokens_available")
    mocker.patch("app.endpoints.query_langchain.validate_model_provider_override")
    mocker.patch("app.endpoints.query_langchain.validate_shield_ids_override")

    # Mock registry to raise exception on initialization
    mock_registry = mocker.AsyncMock()
    mock_registry.initialize = mocker.AsyncMock(
        side_effect=ValueError("Invalid provider configuration")
    )

    mocker.patch(
        "app.endpoints.query_langchain.LLMProviderRegistry",
        return_value=mock_registry,
    )

    with pytest.raises(HTTPException) as exc_info:
        await query_endpoint_handler_langchain(
            request=dummy_request,
            query_request=query_request,
            auth=MOCK_AUTH,
            mcp_headers={},
        )

    assert exc_info.value.status_code == 503
    assert "Provider initialization failed" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_model_retrieval_failure(
    dummy_request: Request,
    langchain_config: LangChainConfiguration,
    mocker: MockerFixture,
) -> None:
    """Test query when chat model retrieval fails."""
    query_request = QueryRequest(
        query="What is Kubernetes?",
        provider="openai",
        model="nonexistent-model",
    )  # pyright: ignore[reportCallIssue]

    mock_config = mocker.Mock()
    mock_config.configuration.langchain = langchain_config
    mock_config.quota_limiters = []
    mocker.patch("app.endpoints.query_langchain.configuration", mock_config)

    mocker.patch("app.endpoints.query_langchain.check_mcp_auth", new=mocker.AsyncMock())
    mocker.patch("app.endpoints.query_langchain.check_tokens_available")
    mocker.patch("app.endpoints.query_langchain.validate_model_provider_override")
    mocker.patch("app.endpoints.query_langchain.validate_shield_ids_override")

    # Mock registry to raise exception on get_chat_model
    mock_registry = mocker.AsyncMock()
    mock_registry.initialize = mocker.AsyncMock()
    mock_registry.get_chat_model = mocker.AsyncMock(
        side_effect=ValueError("Model 'nonexistent-model' not available")
    )

    mocker.patch(
        "app.endpoints.query_langchain.LLMProviderRegistry",
        return_value=mock_registry,
    )

    with pytest.raises(HTTPException) as exc_info:
        await query_endpoint_handler_langchain(
            request=dummy_request,
            query_request=query_request,
            auth=MOCK_AUTH,
            mcp_headers={},
        )

    assert exc_info.value.status_code == 503
    assert "Failed to get model" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_model_invocation_failure(
    dummy_request: Request,
    langchain_config: LangChainConfiguration,
    mocker: MockerFixture,
) -> None:
    """Test query when chat model invocation fails."""
    query_request = QueryRequest(
        query="What is Kubernetes?"
    )  # pyright: ignore[reportCallIssue]

    mock_config = mocker.Mock()
    mock_config.configuration.langchain = langchain_config
    mock_config.quota_limiters = []
    mocker.patch("app.endpoints.query_langchain.configuration", mock_config)

    mocker.patch("app.endpoints.query_langchain.check_mcp_auth", new=mocker.AsyncMock())
    mocker.patch("app.endpoints.query_langchain.check_tokens_available")
    mocker.patch("app.endpoints.query_langchain.validate_model_provider_override")
    mocker.patch("app.endpoints.query_langchain.validate_shield_ids_override")

    # Mock chat model to raise exception on invocation
    mock_chat_model = mocker.AsyncMock()
    mock_chat_model.ainvoke = mocker.AsyncMock(
        side_effect=RuntimeError("API rate limit exceeded")
    )

    mock_registry = mocker.AsyncMock()
    mock_registry.initialize = mocker.AsyncMock()
    mock_registry.get_chat_model = mocker.AsyncMock(return_value=mock_chat_model)

    mocker.patch(
        "app.endpoints.query_langchain.LLMProviderRegistry",
        return_value=mock_registry,
    )

    with pytest.raises(HTTPException) as exc_info:
        await query_endpoint_handler_langchain(
            request=dummy_request,
            query_request=query_request,
            auth=MOCK_AUTH,
            mcp_headers={},
        )

    assert exc_info.value.status_code == 503
    assert "Model invocation failed" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_query_with_attachments(
    dummy_request: Request,
    langchain_config: LangChainConfiguration,
    mocker: MockerFixture,
) -> None:
    """Test query with attachments validation."""
    query_request = QueryRequest(
        query="What is Kubernetes?",
        attachments=[
            Attachment(
                attachment_type="log",
                content_type="text/plain",
                content="log content",
            )
        ],
    )  # pyright: ignore[reportCallIssue]

    mock_config = mocker.Mock()
    mock_config.configuration.langchain = langchain_config
    mock_config.quota_limiters = []
    mocker.patch("app.endpoints.query_langchain.configuration", mock_config)

    mocker.patch("app.endpoints.query_langchain.check_mcp_auth", new=mocker.AsyncMock())
    mocker.patch("app.endpoints.query_langchain.check_tokens_available")
    mocker.patch("app.endpoints.query_langchain.validate_model_provider_override")
    mocker.patch("app.endpoints.query_langchain.validate_shield_ids_override")

    # Mock validate_attachments_metadata
    mock_validate = mocker.patch(
        "app.endpoints.query_langchain.validate_attachments_metadata"
    )

    # Mock provider registry and chat model
    mock_chat_model = mocker.AsyncMock()
    mock_ai_message = AIMessage(content="Response")
    mock_chat_model.ainvoke = mocker.AsyncMock(return_value=mock_ai_message)

    mock_registry = mocker.AsyncMock()
    mock_registry.initialize = mocker.AsyncMock()
    mock_registry.get_chat_model = mocker.AsyncMock(return_value=mock_chat_model)

    mocker.patch(
        "app.endpoints.query_langchain.LLMProviderRegistry",
        return_value=mock_registry,
    )

    mocker.patch("app.endpoints.query_langchain.consume_query_tokens")
    mocker.patch("app.endpoints.query_langchain.get_available_quotas", return_value={})
    mocker.patch("app.endpoints.query_langchain.store_query_results")

    await query_endpoint_handler_langchain(
        request=dummy_request,
        query_request=query_request,
        auth=MOCK_AUTH,
        mcp_headers={},
    )

    # Verify attachments were validated
    mock_validate.assert_called_once_with(query_request.attachments)


@pytest.mark.asyncio
async def test_query_with_conversation_id(
    dummy_request: Request,
    langchain_config: LangChainConfiguration,
    mocker: MockerFixture,
) -> None:
    """Test query with existing conversation ID."""
    conversation_id = "123e4567-e89b-12d3-a456-426614174000"
    query_request = QueryRequest(
        query="What is Kubernetes?",
        conversation_id=conversation_id,
    )  # pyright: ignore[reportCallIssue]

    mock_config = mocker.Mock()
    mock_config.configuration.langchain = langchain_config
    mock_config.quota_limiters = []
    mocker.patch("app.endpoints.query_langchain.configuration", mock_config)

    mocker.patch("app.endpoints.query_langchain.check_mcp_auth", new=mocker.AsyncMock())
    mocker.patch("app.endpoints.query_langchain.check_tokens_available")
    mocker.patch("app.endpoints.query_langchain.validate_model_provider_override")
    mocker.patch("app.endpoints.query_langchain.validate_shield_ids_override")

    # Mock normalize_conversation_id
    mock_normalize = mocker.patch(
        "app.endpoints.query_langchain.normalize_conversation_id",
        return_value="normalized-123",
    )

    # Mock provider registry and chat model
    mock_chat_model = mocker.AsyncMock()
    mock_ai_message = AIMessage(content="Response")
    mock_chat_model.ainvoke = mocker.AsyncMock(return_value=mock_ai_message)

    mock_registry = mocker.AsyncMock()
    mock_registry.initialize = mocker.AsyncMock()
    mock_registry.get_chat_model = mocker.AsyncMock(return_value=mock_chat_model)

    mocker.patch(
        "app.endpoints.query_langchain.LLMProviderRegistry",
        return_value=mock_registry,
    )

    mocker.patch("app.endpoints.query_langchain.consume_query_tokens")
    mocker.patch("app.endpoints.query_langchain.get_available_quotas", return_value={})
    mocker.patch("app.endpoints.query_langchain.store_query_results")

    response = await query_endpoint_handler_langchain(
        request=dummy_request,
        query_request=query_request,
        auth=MOCK_AUTH,
        mcp_headers={},
    )

    # Verify conversation ID was normalized
    mock_normalize.assert_called_once_with(conversation_id)
    assert response.conversation_id == "normalized-123"


@pytest.mark.asyncio
async def test_query_with_custom_provider_and_model(
    dummy_request: Request,
    langchain_config: LangChainConfiguration,
    mocker: MockerFixture,
) -> None:
    """Test query with custom provider and model override."""
    query_request = QueryRequest(
        query="What is Kubernetes?",
        provider="openai",
        model="gpt-3.5-turbo",
    )  # pyright: ignore[reportCallIssue]

    mock_config = mocker.Mock()
    mock_config.configuration.langchain = langchain_config
    mock_config.quota_limiters = []
    mocker.patch("app.endpoints.query_langchain.configuration", mock_config)

    mocker.patch("app.endpoints.query_langchain.check_mcp_auth", new=mocker.AsyncMock())
    mocker.patch("app.endpoints.query_langchain.check_tokens_available")
    mocker.patch("app.endpoints.query_langchain.validate_model_provider_override")
    mocker.patch("app.endpoints.query_langchain.validate_shield_ids_override")

    # Mock provider registry and chat model
    mock_chat_model = mocker.AsyncMock()
    mock_ai_message = AIMessage(content="Response")
    mock_chat_model.ainvoke = mocker.AsyncMock(return_value=mock_ai_message)

    mock_registry = mocker.AsyncMock()
    mock_registry.initialize = mocker.AsyncMock()
    mock_registry.get_chat_model = mocker.AsyncMock(return_value=mock_chat_model)

    mocker.patch(
        "app.endpoints.query_langchain.LLMProviderRegistry",
        return_value=mock_registry,
    )

    mocker.patch("app.endpoints.query_langchain.consume_query_tokens")
    mocker.patch("app.endpoints.query_langchain.get_available_quotas", return_value={})
    mocker.patch("app.endpoints.query_langchain.store_query_results")

    await query_endpoint_handler_langchain(
        request=dummy_request,
        query_request=query_request,
        auth=MOCK_AUTH,
        mcp_headers={},
    )

    # Verify correct model and provider were requested
    mock_registry.get_chat_model.assert_called_once_with(
        model_id="gpt-3.5-turbo",
        provider="openai",
    )
