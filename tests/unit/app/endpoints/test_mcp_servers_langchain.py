# pylint: disable=protected-access,redefined-outer-name

"""Unit tests for the LangChain implementation of MCP servers endpoints."""

from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import AnyHttpUrl, SecretStr
from pytest_mock import MockerFixture

from app.endpoints.mcp_servers_langchain import (
    delete_mcp_server_langchain,
    register_mcp_server_langchain,
)
from app.endpoints.mcp_servers_llama_stack import list_mcp_servers
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
from models.requests import MCPServerRegistrationRequest
from models.responses import MCPServerListResponse, MCPServerRegistrationResponse


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
                name="static-mcp",
                provider_id="model-context-protocol",
                url="http://localhost:3000",
            ),
        ],
        customization=None,
        authorization=None,
        deployment_environment=".",
    )


def _make_app_config(mocker: MockerFixture, config: Configuration) -> AppConfig:
    """Create an AppConfig with the given configuration and patch it."""
    app_config = AppConfig()
    app_config._configuration = config
    app_config._dynamic_mcp_server_names = set()
    mocker.patch("app.endpoints.mcp_servers_langchain.configuration", app_config)
    mocker.patch("app.endpoints.mcp_servers_llama_stack.configuration", app_config)
    return app_config


@pytest.mark.asyncio
async def test_register_mcp_server_langchain_success(
    mocker: MockerFixture,
    mock_configuration: Configuration,
) -> None:
    """Test successful MCP server registration with LangChain."""
    app_config = _make_app_config(mocker, mock_configuration)

    body = MCPServerRegistrationRequest(
        name="new-mcp-server",
        url="http://localhost:8888/mcp",
        provider_id="MCP provider ID",
    )

    result = await register_mcp_server_langchain(body)

    assert isinstance(result, MCPServerRegistrationResponse)
    assert result.name == "new-mcp-server"
    assert result.url == "http://localhost:8888/mcp"
    assert result.provider_id == "MCP provider ID"
    assert "registered successfully" in result.message

    # Verify it was added to configuration
    assert app_config.is_dynamic_mcp_server("new-mcp-server")
    assert any(s.name == "new-mcp-server" for s in app_config.mcp_servers)


@pytest.mark.asyncio
async def test_register_mcp_server_langchain_duplicate_name(
    mocker: MockerFixture,
    mock_configuration: Configuration,
) -> None:
    """Test registration fails when name already exists (LangChain)."""
    _make_app_config(mocker, mock_configuration)

    body = MCPServerRegistrationRequest(
        name="static-mcp",
        url="http://localhost:9999/mcp",
        provider_id="MCP provider ID",
    )

    with pytest.raises(HTTPException) as exc_info:
        await register_mcp_server_langchain(body)
    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_delete_mcp_server_langchain_success(
    mocker: MockerFixture,
    mock_configuration: Configuration,
) -> None:
    """Test successful deletion of a dynamically registered server (LangChain)."""
    app_config = _make_app_config(mocker, mock_configuration)

    # Register a server first
    body = MCPServerRegistrationRequest(
        name="to-delete",
        url="http://localhost:7777/mcp",
        provider_id="MCP provider ID",
    )
    await register_mcp_server_langchain(body)
    assert app_config.is_dynamic_mcp_server("to-delete")

    # Delete it
    result = await delete_mcp_server_langchain("to-delete")

    assert result.name == "to-delete"
    assert "unregistered successfully" in result.message

    # Verify it was removed from configuration
    assert not app_config.is_dynamic_mcp_server("to-delete")
    assert not any(s.name == "to-delete" for s in app_config.mcp_servers)


@pytest.mark.asyncio
async def test_delete_static_mcp_server_langchain_forbidden(
    mocker: MockerFixture,
    mock_configuration: Configuration,
) -> None:
    """Test that deleting a statically configured server is forbidden (LangChain)."""
    _make_app_config(mocker, mock_configuration)

    with pytest.raises(HTTPException) as exc_info:
        await delete_mcp_server_langchain("static-mcp")
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_delete_nonexistent_mcp_server_langchain(
    mocker: MockerFixture,
    mock_configuration: Configuration,
) -> None:
    """Test that deleting a non-existent server returns 404 (LangChain)."""
    _make_app_config(mocker, mock_configuration)

    with pytest.raises(HTTPException) as exc_info:
        await delete_mcp_server_langchain("no-such-server")
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_list_mcp_servers_with_langchain(
    mocker: MockerFixture,
    mock_configuration: Configuration,
) -> None:
    """Test listing shows both static and dynamic servers (backend-agnostic)."""
    _make_app_config(mocker, mock_configuration)

    # Register a dynamic server
    body = MCPServerRegistrationRequest(
        name="dynamic-server",
        url="http://localhost:9999/mcp",
        provider_id="MCP provider ID",
    )
    await register_mcp_server_langchain(body)

    # List servers (shared implementation)
    result = await list_mcp_servers()

    assert isinstance(result, MCPServerListResponse)
    assert len(result.servers) == 2
    sources = {s.name: s.source for s in result.servers}
    assert sources["static-mcp"] == "config"
    assert sources["dynamic-server"] == "api"
