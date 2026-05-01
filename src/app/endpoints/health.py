"""Handlers for health REST API endpoints.

These endpoints are used to check if service is live and prepared to accept
requests. Note that these endpoints can be accessed using GET or HEAD HTTP
methods. For HEAD HTTP method, just the HTTP response code is used.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Response

from authentication import get_auth_dependency
from authentication.interface import AuthTuple
from authorization.middleware import authorize
from configuration import configuration
from log import get_logger
from models.config import Action
from models.responses import (
    UNAUTHORIZED_OPENAPI_EXAMPLES,
    ForbiddenResponse,
    LivenessResponse,
    ReadinessResponse,
    ServiceUnavailableResponse,
    UnauthorizedResponse,
)

logger = get_logger(__name__)
router = APIRouter(tags=["health"])


get_readiness_responses: dict[int | str, dict[str, Any]] = {
    200: ReadinessResponse.openapi_response(),
    401: UnauthorizedResponse.openapi_response(examples=UNAUTHORIZED_OPENAPI_EXAMPLES),
    403: ForbiddenResponse.openapi_response(examples=["endpoint"]),
    503: ServiceUnavailableResponse.openapi_response(
        examples=["backend service", "kubernetes api"]
    ),
}

get_liveness_responses: dict[int | str, dict[str, Any]] = {
    200: LivenessResponse.openapi_response(),
    401: UnauthorizedResponse.openapi_response(examples=UNAUTHORIZED_OPENAPI_EXAMPLES),
    403: ForbiddenResponse.openapi_response(examples=["endpoint"]),
    503: ServiceUnavailableResponse.openapi_response(examples=["kubernetes api"]),
}


@router.get("/readiness", responses=get_readiness_responses)
@authorize(Action.INFO)
async def readiness_probe_get_method(
    auth: Annotated[AuthTuple, Depends(get_auth_dependency())],
    response: Response,
) -> ReadinessResponse:
    """
    Handle the readiness probe endpoint, returning service readiness.

    Routes to either Llama Stack or LangChain implementation based on feature flags.

    Parameters:
        auth: Authentication tuple from the auth dependency (used by middleware).
        response: The outgoing HTTP response (used by middleware).

    Returns:
        ReadinessResponse: Object with `ready` indicating overall readiness,
        `reason` explaining the outcome, and `providers` containing the list of
        unhealthy ProviderHealthStatus entries (empty when ready).
    """
    # Used only for authorization
    _ = auth

    feature_flags = configuration.configuration.feature_flags
    use_langchain = (
        feature_flags.use_langchain or "health" in feature_flags.langchain_endpoints
    )

    if use_langchain:
        from app.endpoints.health_langchain import readiness_langchain

        logger.info("Routing /readiness to LangChain implementation")
        return await readiness_langchain(response)

    from app.endpoints.health_llama_stack import readiness_llama_stack

    logger.info("Routing /readiness to Llama Stack implementation")
    return await readiness_llama_stack(response)


@router.get("/liveness", responses=get_liveness_responses)
@authorize(Action.INFO)
async def liveness_probe_get_method(
    auth: Annotated[AuthTuple, Depends(get_auth_dependency())],
) -> LivenessResponse:
    """
    Return the liveness status of the service.

    This endpoint does not depend on the backend, so no routing is needed.

    Parameters:
        auth: Authentication tuple from the auth dependency (used by middleware).

    Returns:
        LivenessResponse: Indicates that the service is alive.
    """
    # Used only for authorization
    _ = auth

    logger.info("Response to /v1/liveness endpoint")

    return LivenessResponse(alive=True)
