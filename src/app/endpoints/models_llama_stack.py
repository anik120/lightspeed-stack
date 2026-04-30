"""Llama Stack implementation for /models endpoint."""

from typing import Any

from fastapi import HTTPException
from llama_stack_client import APIConnectionError

from client import AsyncLlamaStackClientHolder
from configuration import configuration
from log import get_logger
from models.requests import ModelFilter
from models.responses import ModelsResponse, ServiceUnavailableResponse
from utils.endpoints import check_configuration_loaded

logger = get_logger(__name__)


def parse_llama_stack_model(model: Any) -> dict[str, Any]:
    """
    Parse llama-stack model.

    Converting the new llama-stack model format (0.4.x) with custom_metadata.

    Parameters:
        model: Model object from llama-stack (has id, custom_metadata, object fields)

    Returns:
        dict: Model in legacy format with identifier, provider_id, model_type, etc.
    """
    custom_metadata = getattr(model, "custom_metadata", {}) or {}

    model_type = str(custom_metadata.get("model_type", "unknown"))

    metadata = {
        k: v
        for k, v in custom_metadata.items()
        if k not in ("provider_id", "provider_resource_id", "model_type")
    }

    return {
        "identifier": getattr(model, "id", ""),
        "metadata": metadata,
        "api_model_type": model_type,
        "provider_id": str(custom_metadata.get("provider_id", "")),
        "type": getattr(model, "object", "model"),
        "provider_resource_id": str(custom_metadata.get("provider_resource_id", "")),
        "model_type": model_type,
    }


async def models_llama_stack(model_type: ModelFilter) -> ModelsResponse:
    """
    Retrieve available models from Llama Stack.

    Fetches the list of models from the Llama Stack service and optionally
    filters by model type (e.g., "llm", "embedding").

    Parameters:
        model_type: Optional filter to return only models matching this type.

    Returns:
        ModelsResponse: Object containing the list of available models.

    Raises:
        HTTPException: If unable to connect to Llama Stack or retrieval fails.
    """
    check_configuration_loaded(configuration)

    llama_stack_configuration = configuration.llama_stack_configuration
    logger.info("Llama stack config: %s", llama_stack_configuration)

    try:
        # try to get Llama Stack client
        client = AsyncLlamaStackClientHolder().get_client()
        # retrieve models
        models = await client.models.list()

        # parse models to legacy format
        parsed_models = [parse_llama_stack_model(model) for model in models]

        # optional filtering by model type
        if model_type.model_type is not None:
            parsed_models = [
                model
                for model in parsed_models
                if model["model_type"] == model_type.model_type
            ]

        return ModelsResponse(models=parsed_models)

    # Connection to Llama Stack server failed
    except APIConnectionError as e:
        logger.error("Unable to connect to Llama Stack: %s", e)
        response = ServiceUnavailableResponse(backend_name="Llama Stack", cause=str(e))
        raise HTTPException(**response.model_dump()) from e
