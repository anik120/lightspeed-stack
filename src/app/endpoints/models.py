"""Handler for REST API call to list available models.

Routes to either Llama Stack or LangChain implementation based on feature flags.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Query, Request
from fastapi.params import Depends

from authentication import get_auth_dependency
from authentication.interface import AuthTuple
from authorization.middleware import authorize
from configuration import configuration
from log import get_logger
from models.config import Action
from models.requests import ModelFilter
from models.responses import (
    UNAUTHORIZED_OPENAPI_EXAMPLES,
    ForbiddenResponse,
    InternalServerErrorResponse,
    ModelsResponse,
    ServiceUnavailableResponse,
    UnauthorizedResponse,
)
from utils.endpoints import check_configuration_loaded

logger = get_logger(__name__)
router = APIRouter(tags=["models"])


models_responses: dict[int | str, dict[str, Any]] = {
    200: ModelsResponse.openapi_response(),
    401: UnauthorizedResponse.openapi_response(examples=UNAUTHORIZED_OPENAPI_EXAMPLES),
    403: ForbiddenResponse.openapi_response(examples=["endpoint"]),
    500: InternalServerErrorResponse.openapi_response(examples=["configuration"]),
    503: ServiceUnavailableResponse.openapi_response(
        examples=["llama stack", "kubernetes api"]
    ),
}


@router.get("/models", responses=models_responses)
@authorize(Action.GET_MODELS)
async def models_endpoint_handler(
    request: Request,
    auth: Annotated[AuthTuple, Depends(get_auth_dependency())],
    model_type: Annotated[ModelFilter, Query()],
) -> ModelsResponse:
    """
    Handle requests to the /models endpoint.

    Process GET requests to the /models endpoint, returning a list of available
    models. Routes to either Llama Stack or LangChain implementation based on
    feature flags.

    It is possible to specify "model_type" query parameter that is used as a
    filter. For example, if model type is set to "llm", only LLM models will
    be returned:

        curl http://localhost:8080/v1/models?model_type=llm

    The "model_type" query parameter is optional. When not specified, all models
    will be returned.

    ### Parameters:
    - request: The incoming HTTP request (used by middleware).
    - auth: Authentication tuple from the auth dependency (used by middleware).
    - model_type: Optional filter to return only models matching this type.

    ### Raises:
    - HTTPException: If unable to connect to the backend server or if
      model retrieval fails for any reason.

    ### Returns:
    - ModelsResponse: An object containing the list of available models.
    """
    check_configuration_loaded(configuration)

    # Route based on feature flags
    feature_flags = configuration.configuration.feature_flags
    use_langchain = (
        feature_flags.use_langchain or "models" in feature_flags.langchain_endpoints
    )

    if use_langchain:
        # Import here to avoid circular dependencies
        # pylint: disable=import-outside-toplevel
        from app.endpoints.models_langchain import models_langchain

        logger.info("Routing /models to LangChain implementation")
        return await models_langchain(model_type)

    # pylint: disable=import-outside-toplevel
    from app.endpoints.models_llama_stack import models_llama_stack

    logger.info("Routing /models to Llama Stack implementation")
    return await models_llama_stack(model_type)
