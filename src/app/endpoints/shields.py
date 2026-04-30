"""Shields endpoint for retrieving available shields."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from authentication import get_auth_dependency
from authentication.interface import AuthTuple
from authorization.middleware import authorize
from configuration import configuration
from log import get_logger
from models.config import Action
from models.responses import ShieldsResponse
from utils.endpoints import check_configuration_loaded

logger = get_logger(__name__)

router = APIRouter()

shields_responses = {
    200: {"model": ShieldsResponse},
}


@router.get("/shields", responses=shields_responses)
@authorize(Action.GET_SHIELDS)
async def shields_endpoint_handler(
    request: Request,
    auth: Annotated[AuthTuple, Depends(get_auth_dependency())],
) -> ShieldsResponse:
    """
    Retrieve available shields.

    Routes to either Llama Stack or LangChain implementation based on feature flags.

    Parameters:
        request: The incoming FastAPI request object.
        auth: Authentication tuple from auth dependency.

    Returns:
        ShieldsResponse: Object containing the list of available shields.
    """
    check_configuration_loaded(configuration)

    feature_flags = configuration.configuration.feature_flags
    use_langchain = (
        feature_flags.use_langchain or "shields" in feature_flags.langchain_endpoints
    )

    if use_langchain:
        from app.endpoints.shields_langchain import shields_langchain

        logger.info("Routing /shields to LangChain implementation")
        return await shields_langchain()

    from app.endpoints.shields_llama_stack import shields_llama_stack

    logger.info("Routing /shields to Llama Stack implementation")
    return await shields_llama_stack()
