"""LangChain implementation for /models endpoint."""

from typing import Any

from fastapi import HTTPException

from configuration import configuration
from langchain_client import LLMProviderRegistry
from log import get_logger
from models.requests import ModelFilter
from models.responses import ModelsResponse, ServiceUnavailableResponse
from utils.endpoints import check_configuration_loaded

logger = get_logger(__name__)


async def models_langchain(model_type: ModelFilter) -> ModelsResponse:
    """
    Retrieve available models from LangChain providers.

    Fetches the list of models from configured LangChain providers and optionally
    filters by model type (e.g., "llm", "embedding").

    Parameters:
        model_type: Optional filter to return only models matching this type.

    Returns:
        ModelsResponse: Object containing the list of available models.

    Raises:
        HTTPException: If unable to retrieve models from LangChain providers.
    """
    check_configuration_loaded(configuration)

    langchain_config = configuration.langchain_configuration
    if langchain_config is None:
        raise HTTPException(
            status_code=503,
            detail={"backend_name": "LangChain", "response": "LangChain not configured"},
        )

    logger.info("Using LangChain provider registry for models")

    try:
        # Get the LangChain provider registry
        registry = LLMProviderRegistry()
        await registry.initialize(langchain_config)

        # Retrieve all models from all providers
        all_models = await registry.list_models_by_provider()

        # Convert to the expected format
        parsed_models: list[dict[str, Any]] = []
        for provider_name, models in all_models.items():
            for model_id in models:
                # For LangChain, all current models are LLMs
                # In the future, we may need to distinguish embedding models
                model_entry = {
                    "identifier": f"{provider_name}/{model_id}",
                    "metadata": {},
                    "api_model_type": "llm",  # Default to llm
                    "provider_id": provider_name,
                    "type": "model",
                    "provider_resource_id": model_id,
                    "model_type": "llm",  # Default to llm
                }
                parsed_models.append(model_entry)

        # Optional filtering by model type
        if model_type.model_type is not None:
            parsed_models = [
                model
                for model in parsed_models
                if model["model_type"] == model_type.model_type
            ]

        return ModelsResponse(models=parsed_models)

    except Exception as e:
        logger.error("Unable to retrieve models from LangChain: %s", e)
        response = ServiceUnavailableResponse(backend_name="LangChain", cause=str(e))
        raise HTTPException(**response.model_dump()) from e
