# pylint: disable=protected-access

"""Unit tests for LangChain tools endpoint."""

from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import AnyHttpUrl, SecretStr
from pytest_mock import MockerFixture

from app.endpoints.tools_langchain import tools_langchain
from configuration import AppConfig
from models.config import (
    Configuration,
    CORSConfiguration,
    LangChainConfiguration,
    ModelContextProtocolServer,
    ServiceConfiguration,
    TLSConfiguration,
    UserDataCollection,
)
from models.responses import ToolsResponse


@pytest.fixture
def mock_configuration() -> Configuration:
    """Create a mock configuration with MCP servers for LangChain."""
    return Configuration(
        name="test",
        service=ServiceConfiguration(
            tls_config=TLSConfiguration(
                tls_certificate_path=Path("tests/configuration/server.crt"),
                tls_key_path=Path("tests/configuration/server.key"),
                tls_key_password=Path("tests/configuration/password"),
            ),
            cors=CORSConfiguration(
                allow_origins=["*"],
                allow_credentials=False,
                allow_methods=["*"],
                allow_headers=["*"],
            ),
            host="localhost",
            port=1234,
            base_url=".",
            auth_enabled=False,
            workers=1,
            color_log=True,
            access_log=True,
            root_path="/.",
        ),
        langchain=LangChainConfiguration(
            providers={
                "openai": {
                    "api_key": SecretStr("test-key"),
                    "models": ["gpt-4"],
                }
            },
            default_provider="openai",
            default_model="gpt-4",
        ),
        user_data_collection=UserDataCollection(
            transcripts_enabled=False,
            feedback_enabled=False,
            transcripts_storage=".",
            feedback_storage=".",
        ),
        mcp_servers=[
            ModelContextProtocolServer(
                name="filesystem-tools",
                provider_id="model-context-protocol",
                url="http://localhost:3000",
            ),
            ModelContextProtocolServer(
                name="git-tools",
                provider_id="model-context-protocol",
                url="http://localhost:3001",
            ),
        ],
        customization=None,
        authorization=None,
        deployment_environment=".",
    )


@pytest.mark.asyncio
async def test_tools_langchain_no_mcp_servers(mocker: MockerFixture) -> None:
    """Test LangChain tools endpoint with no MCP servers configured."""
    mock_config = Configuration(
        name="test",
        service=ServiceConfiguration(
            tls_config=TLSConfiguration(
                tls_certificate_path=Path("tests/configuration/server.crt"),
                tls_key_path=Path("tests/configuration/server.key"),
                tls_key_password=Path("tests/configuration/password"),
            ),
            cors=CORSConfiguration(
                allow_origins=["*"],
                allow_credentials=False,
                allow_methods=["*"],
                allow_headers=["*"],
            ),
            host="localhost",
            port=1234,
            base_url=".",
            auth_enabled=False,
            workers=1,
            color_log=True,
            access_log=True,
            root_path="/.",
        ),
        langchain=LangChainConfiguration(
            providers={
                "openai": {
                    "api_key": SecretStr("test-key"),
                    "models": ["gpt-4"],
                }
            },
            default_provider="openai",
            default_model="gpt-4",
        ),
        user_data_collection=UserDataCollection(
            transcripts_enabled=False,
            feedback_enabled=False,
            transcripts_storage=".",
            feedback_storage=".",
        ),
        mcp_servers=[],
        customization=None,
        authorization=None,
        deployment_environment=".",
    )
    app_config = AppConfig()
    app_config._configuration = mock_config
    mocker.patch("app.endpoints.tools_langchain.configuration", app_config)

    response = await tools_langchain({})

    assert isinstance(response, ToolsResponse)
    assert len(response.tools) == 0


@pytest.mark.asyncio
async def test_tools_langchain_with_mcp_servers_placeholder(
    mocker: MockerFixture,
    mock_configuration: Configuration,  # pylint: disable=redefined-outer-name
) -> None:
    """Test LangChain tools endpoint with MCP servers (placeholder implementation).

    Note: This test verifies the current placeholder behavior where tool discovery
    returns empty lists. Once LangChain MCP integration is implemented, this test
    should be updated to verify actual tool discovery.
    """
    app_config = AppConfig()
    app_config._configuration = mock_configuration
    mocker.patch("app.endpoints.tools_langchain.configuration", app_config)

    response = await tools_langchain({})

    assert isinstance(response, ToolsResponse)
    # Placeholder implementation returns empty list for each server
    assert len(response.tools) == 0


@pytest.mark.asyncio
async def test_tools_langchain_skips_unresolved_auth_headers(
    mocker: MockerFixture,
) -> None:
    """Test that MCP servers with unresolved auth headers are skipped."""
    mock_config = Configuration(
        name="test",
        service=ServiceConfiguration(
            tls_config=TLSConfiguration(
                tls_certificate_path=Path("tests/configuration/server.crt"),
                tls_key_path=Path("tests/configuration/server.key"),
                tls_key_password=Path("tests/configuration/password"),
            ),
            cors=CORSConfiguration(
                allow_origins=["*"],
                allow_credentials=False,
                allow_methods=["*"],
                allow_headers=["*"],
            ),
            host="localhost",
            port=1234,
            base_url=".",
            auth_enabled=False,
            workers=1,
            color_log=True,
            access_log=True,
            root_path="/.",
        ),
        langchain=LangChainConfiguration(
            providers={
                "openai": {
                    "api_key": SecretStr("test-key"),
                    "models": ["gpt-4"],
                }
            },
            default_provider="openai",
            default_model="gpt-4",
        ),
        user_data_collection=UserDataCollection(
            transcripts_enabled=False,
            feedback_enabled=False,
            transcripts_storage=".",
            feedback_storage=".",
        ),
        mcp_servers=[
            ModelContextProtocolServer(
                name="auth-required-server",
                provider_id="model-context-protocol",
                url="http://localhost:3000",
                authorization_headers={"Authorization": "client"},  # Requires auth
            ),
        ],
        customization=None,
        authorization=None,
        deployment_environment=".",
    )
    app_config = AppConfig()
    app_config._configuration = mock_config
    mocker.patch("app.endpoints.tools_langchain.configuration", app_config)

    # Pass empty headers - auth headers not resolved
    response = await tools_langchain({})

    assert isinstance(response, ToolsResponse)
    assert len(response.tools) == 0  # Server skipped due to unresolved auth


@pytest.mark.asyncio
async def test_tools_langchain_discovery_error(
    mocker: MockerFixture,
    mock_configuration: Configuration,  # pylint: disable=redefined-outer-name
) -> None:
    """Test that discovery errors are propagated as HTTP exceptions."""
    app_config = AppConfig()
    app_config._configuration = mock_configuration
    mocker.patch("app.endpoints.tools_langchain.configuration", app_config)

    # Mock tool discovery to raise an exception
    mocker.patch(
        "app.endpoints.tools_langchain._discover_tools_from_mcp_server",
        side_effect=Exception("Connection failed"),
    )

    with pytest.raises(HTTPException) as exc_info:
        await tools_langchain({})

    assert exc_info.value.status_code == 503
    detail = exc_info.value.detail
    assert isinstance(detail, dict)
    assert "MCP server" in detail["response"]  # type: ignore
