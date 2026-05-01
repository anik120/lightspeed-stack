"""Info endpoint for retrieving service and backend information."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from authentication import get_auth_dependency
from authentication.interface import AuthTuple
from authorization.middleware import authorize
from configuration import configuration
from log import get_logger
from models.config import Action
from models.responses import (
    UNAUTHORIZED_OPENAPI_EXAMPLES,
    ForbiddenResponse,
    InfoResponse,
    ServiceUnavailableResponse,
    UnauthorizedResponse,
)
from utils.endpoints import check_configuration_loaded

logger = get_logger(__name__)
router = APIRouter(tags=["info"])


get_info_responses: dict[int | str, dict[str, Any]] = {
    200: InfoResponse.openapi_response(),
    401: UnauthorizedResponse.openapi_response(examples=UNAUTHORIZED_OPENAPI_EXAMPLES),
    403: ForbiddenResponse.openapi_response(examples=["endpoint"]),
    503: ServiceUnavailableResponse.openapi_response(
        examples=["backend service", "kubernetes api"]
    ),
}


@router.get("/info", responses=get_info_responses)
@authorize(Action.INFO)
async def info_endpoint_handler(
    auth: Annotated[AuthTuple, Depends(get_auth_dependency())],
    request: Request,
) -> InfoResponse:
    """
    Retrieve service information.

    Routes to either Llama Stack or LangChain implementation based on feature flags.

    Parameters:
        request: The incoming HTTP request.
        auth: Authentication tuple from the auth dependency.

    Returns:
        InfoResponse: Service name, version, and backend version.
    """
    check_configuration_loaded(configuration)

    feature_flags = configuration.configuration.feature_flags
    use_langchain = (
        feature_flags.use_langchain or "info" in feature_flags.langchain_endpoints
    )

    if use_langchain:
        from app.endpoints.info_langchain import info_langchain

        logger.info("Routing /info to LangChain implementation")
        return await info_langchain()

    from app.endpoints.info_llama_stack import info_llama_stack

    logger.info("Routing /info to Llama Stack implementation")
    return await info_llama_stack()
