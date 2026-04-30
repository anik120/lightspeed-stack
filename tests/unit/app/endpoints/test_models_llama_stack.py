"""Unit tests for the /models REST API endpoint."""

from typing import Any

import pytest
from fastapi import HTTPException, status
from llama_stack_client import APIConnectionError
from pytest_mock import MockerFixture
from pytest_subtests import SubTests

from app.endpoints.models_llama_stack import models_llama_stack
from configuration import AppConfig
from models.requests import ModelFilter


# pylint: disable=R0903
class Model:
    """Model information returned in response."""

    def __init__(self, model_id: str, provider_id: str, model_type: str) -> None:
        """Initialize model information."""
        self.id = model_id
        self.custom_metadata = {
            "model_type": model_type,
            "provider_id": provider_id,
        }


@pytest.mark.asyncio
async def test_models_llama_stack_configuration_not_loaded(
    mocker: MockerFixture,
) -> None:
    """Test models_llama_stack when configuration is not loaded."""
    # simulate state when no configuration is loaded
    mock_config = AppConfig()
    mocker.patch("app.endpoints.models_llama_stack.configuration", mock_config)

    with pytest.raises(HTTPException) as e:
        await models_llama_stack(model_type=ModelFilter(model_type=None))
        assert e.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert e.value.detail["response"] == "Configuration is not loaded"  # type: ignore


@pytest.mark.asyncio
async def test_models_llama_stack_configuration_loaded(
    mocker: MockerFixture,
) -> None:
    """Test the models endpoint handler if configuration is loaded.

    Verify the models endpoint raises HTTP 503 when configuration is loaded but
    the Llama Stack client cannot connect.

    Loads an AppConfig from a test dictionary, patches the endpoint's
    configuration and AsyncLlamaStackClientHolder so that get_client raises
    APIConnectionError, issues a request with an authorization header, and
    asserts that calling the handler raises an HTTPException with status 503
    and a detail response of "Unable to connect to Llama Stack".
    """

    # configuration for tests
    config_dict: dict[str, Any] = {
        "name": "foo",
        "service": {
            "host": "localhost",
            "port": 8080,
            "auth_enabled": False,
            "workers": 1,
            "color_log": True,
            "access_log": True,
        },
        "llama_stack": {
            "api_key": "xyzzy",
            "url": "http://x.y.com:1234",
            "use_as_library_client": False,
        },
        "user_data_collection": {
            "feedback_enabled": False,
        },
        "customization": None,
        "authorization": {"access_rules": []},
        "authentication": {"module": "noop"},
    }
    cfg = AppConfig()
    cfg.init_from_dict(config_dict)

    mocker.patch("app.endpoints.models_llama_stack.configuration", cfg)
    mock_client_holder = mocker.patch(
        "app.endpoints.models_llama_stack.AsyncLlamaStackClientHolder"
    )
    mock_client_holder.return_value.get_client.side_effect = APIConnectionError(
        request=mocker.Mock()
    )

    with pytest.raises(HTTPException) as e:
        await models_llama_stack(model_type=ModelFilter(model_type=None))
    assert e.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert e.value.detail["response"] == "Unable to connect to Llama Stack"  # type: ignore


@pytest.mark.asyncio
async def test_models_llama_stack_unable_to_retrieve_models_list(
    mocker: MockerFixture,
) -> None:
    """Test the models endpoint handler if configuration is loaded."""

    # configuration for tests
    config_dict: dict[str, Any] = {
        "name": "foo",
        "service": {
            "host": "localhost",
            "port": 8080,
            "auth_enabled": False,
            "workers": 1,
            "color_log": True,
            "access_log": True,
        },
        "llama_stack": {
            "api_key": "xyzzy",
            "url": "http://x.y.com:1234",
            "use_as_library_client": False,
        },
        "user_data_collection": {
            "feedback_enabled": False,
        },
        "customization": None,
        "authorization": {"access_rules": []},
        "authentication": {"module": "noop"},
    }
    cfg = AppConfig()
    cfg.init_from_dict(config_dict)

    # Mock the LlamaStack client
    mock_client = mocker.AsyncMock()
    mock_client.models.list.return_value = []
    mock_lsc = mocker.patch(
        "app.endpoints.models_llama_stack.AsyncLlamaStackClientHolder.get_client"
    )
    mock_lsc.return_value = mock_client
    mock_config = mocker.Mock()
    mocker.patch("app.endpoints.models_llama_stack.configuration", mock_config)

    response = await models_llama_stack(model_type=ModelFilter(model_type=None))
    assert response is not None


@pytest.mark.asyncio
async def test_models_llama_stack_model_type_query_parameter(
    mocker: MockerFixture,
) -> None:
    """Test the models endpoint handler if model_type query parameter is specified."""

    # configuration for tests
    config_dict: dict[str, Any] = {
        "name": "foo",
        "service": {
            "host": "localhost",
            "port": 8080,
            "auth_enabled": False,
            "workers": 1,
            "color_log": True,
            "access_log": True,
        },
        "llama_stack": {
            "api_key": "xyzzy",
            "url": "http://x.y.com:1234",
            "use_as_library_client": False,
        },
        "user_data_collection": {
            "feedback_enabled": False,
        },
        "customization": None,
        "authorization": {"access_rules": []},
        "authentication": {"module": "noop"},
    }
    cfg = AppConfig()
    cfg.init_from_dict(config_dict)

    # Mock the LlamaStack client
    mock_client = mocker.AsyncMock()
    mock_client.models.list.return_value = []
    mock_lsc = mocker.patch(
        "app.endpoints.models_llama_stack.AsyncLlamaStackClientHolder.get_client"
    )
    mock_lsc.return_value = mock_client
    mock_config = mocker.Mock()
    mocker.patch("app.endpoints.models_llama_stack.configuration", mock_config)

    response = await models_llama_stack(model_type=ModelFilter(model_type="llm"))
    assert response is not None


@pytest.mark.asyncio
async def test_models_llama_stack_model_list_retrieved(
    mocker: MockerFixture,
) -> None:
    """Test the models endpoint handler if model list can be retrieved."""

    # configuration for tests
    config_dict: dict[str, Any] = {
        "name": "foo",
        "service": {
            "host": "localhost",
            "port": 8080,
            "auth_enabled": False,
            "workers": 1,
            "color_log": True,
            "access_log": True,
        },
        "llama_stack": {
            "api_key": "xyzzy",
            "url": "http://x.y.com:1234",
            "use_as_library_client": False,
        },
        "user_data_collection": {
            "feedback_enabled": False,
        },
        "customization": None,
        "authorization": {"access_rules": []},
        "authentication": {"module": "noop"},
    }
    cfg = AppConfig()
    cfg.init_from_dict(config_dict)

    # Mock the LlamaStack client
    mock_client = mocker.AsyncMock()
    mock_client.models.list.return_value = [
        Model("model1", "provider1", "llm"),
        Model("model2", "provider2", "embedding"),
        Model("model3", "provider3", "llm"),
        Model("model4", "provider4", "embedding"),
    ]
    mock_lsc = mocker.patch(
        "app.endpoints.models_llama_stack.AsyncLlamaStackClientHolder.get_client"
    )
    mock_lsc.return_value = mock_client
    mock_config = mocker.Mock()
    mocker.patch("app.endpoints.models_llama_stack.configuration", mock_config)

    response = await models_llama_stack(model_type=ModelFilter(model_type=None))
    assert response is not None
    assert len(response.models) == 4
    assert response.models[0]["identifier"] == "model1"
    assert response.models[0]["model_type"] == "llm"
    assert response.models[1]["identifier"] == "model2"
    assert response.models[1]["model_type"] == "embedding"
    assert response.models[2]["identifier"] == "model3"
    assert response.models[2]["model_type"] == "llm"
    assert response.models[3]["identifier"] == "model4"
    assert response.models[3]["model_type"] == "embedding"


@pytest.mark.asyncio
async def test_models_llama_stack_model_list_retrieved_with_query_parameter(
    mocker: MockerFixture,
    subtests: SubTests,
) -> None:
    """Test the models endpoint handler if model list can be retrieved."""

    # configuration for tests
    config_dict: dict[str, Any] = {
        "name": "foo",
        "service": {
            "host": "localhost",
            "port": 8080,
            "auth_enabled": False,
            "workers": 1,
            "color_log": True,
            "access_log": True,
        },
        "llama_stack": {
            "api_key": "xyzzy",
            "url": "http://x.y.com:1234",
            "use_as_library_client": False,
        },
        "user_data_collection": {
            "feedback_enabled": False,
        },
        "customization": None,
        "authorization": {"access_rules": []},
        "authentication": {"module": "noop"},
    }
    cfg = AppConfig()
    cfg.init_from_dict(config_dict)

    # Mock the LlamaStack client
    mock_client = mocker.AsyncMock()
    mock_client.models.list.return_value = [
        Model("model1", "provider1", "llm"),
        Model("model2", "provider2", "embedding"),
        Model("model3", "provider3", "llm"),
        Model("model4", "provider4", "embedding"),
    ]
    mock_lsc = mocker.patch(
        "app.endpoints.models_llama_stack.AsyncLlamaStackClientHolder.get_client"
    )
    mock_lsc.return_value = mock_client
    mock_config = mocker.Mock()
    mocker.patch("app.endpoints.models_llama_stack.configuration", mock_config)

    with subtests.test(msg="Model type = 'llm'"):
        response = await models_llama_stack(model_type=ModelFilter(model_type="llm"))
        assert response is not None
        assert len(response.models) == 2
        assert response.models[0]["identifier"] == "model1"
        assert response.models[0]["model_type"] == "llm"
        assert response.models[1]["identifier"] == "model3"
        assert response.models[1]["model_type"] == "llm"

    with subtests.test(msg="Model type = 'embedding'"):
        response = await models_llama_stack(
            model_type=ModelFilter(model_type="embedding")
        )
        assert response is not None
        assert len(response.models) == 2
        assert response.models[0]["identifier"] == "model2"
        assert response.models[0]["model_type"] == "embedding"
        assert response.models[1]["identifier"] == "model4"
        assert response.models[1]["model_type"] == "embedding"

    with subtests.test(msg="Model type = 'xyzzy'"):
        response = await models_llama_stack(model_type=ModelFilter(model_type="xyzzy"))
        assert response is not None
        assert len(response.models) == 0

    with subtests.test(msg="Model type is empty string"):
        response = await models_llama_stack(model_type=ModelFilter(model_type=""))
        assert response is not None
        assert len(response.models) == 0


@pytest.mark.asyncio
async def test_models_endpoint_llama_stack_connection_error(
    mocker: MockerFixture,
) -> None:
    """Test the model endpoint when LlamaStack connection fails."""

    # configuration for tests
    config_dict: dict[str, Any] = {
        "name": "foo",
        "service": {
            "host": "localhost",
            "port": 8080,
            "auth_enabled": False,
            "workers": 1,
            "color_log": True,
            "access_log": True,
        },
        "llama_stack": {
            "api_key": "xyzzy",
            "url": "http://x.y.com:1234",
            "use_as_library_client": False,
        },
        "user_data_collection": {
            "feedback_enabled": False,
        },
        "customization": None,
        "authorization": {"access_rules": []},
        "authentication": {"module": "noop"},
    }

    # mock AsyncLlamaStackClientHolder to raise APIConnectionError
    # when models.list() method is called
    mock_client = mocker.AsyncMock()
    mock_client.models.list.side_effect = APIConnectionError(request=None)  # type: ignore
    mock_client_holder = mocker.patch(
        "app.endpoints.models_llama_stack.AsyncLlamaStackClientHolder"
    )
    mock_client_holder.return_value.get_client.return_value = mock_client

    cfg = AppConfig()
    cfg.init_from_dict(config_dict)

    with pytest.raises(HTTPException) as e:
        await models_llama_stack(model_type=ModelFilter(model_type=None))
        assert e.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert e.value.detail["response"] == "Unable to connect to Llama Stack"  # type: ignore
        assert "Unable to connect to Llama Stack" in e.value.detail["cause"]  # type: ignore
