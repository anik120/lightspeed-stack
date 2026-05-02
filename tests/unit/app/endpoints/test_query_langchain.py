# pylint: disable=redefined-outer-name

"""Unit tests for LangChain query endpoint placeholder."""

import pytest
from fastapi import HTTPException, Request

from app.endpoints.query_langchain import query_endpoint_handler_langchain
from models.config import Action
from models.requests import QueryRequest

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


@pytest.mark.asyncio
async def test_query_langchain_not_implemented(
    dummy_request: Request,
) -> None:
    """Test that LangChain query endpoint returns not implemented error.

    This is a placeholder test for the infrastructure-only migration commit.
    Full LangChain implementation will be added in a follow-up commit.
    """
    query_request = QueryRequest(
        query="What is Kubernetes?"
    )  # pyright: ignore[reportCallIssue]

    with pytest.raises(HTTPException) as exc_info:
        await query_endpoint_handler_langchain(
            dummy_request,
            query_request,
            MOCK_AUTH,
            {},
        )

    assert exc_info.value.status_code == 503
