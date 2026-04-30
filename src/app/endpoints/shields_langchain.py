"""LangChain implementation for /shields endpoint."""

from fastapi import HTTPException

from configuration import configuration
from log import get_logger
from models.responses import ServiceUnavailableResponse, ShieldsResponse
from utils.endpoints import check_configuration_loaded

logger = get_logger(__name__)


async def shields_langchain() -> ShieldsResponse:
    """
    Retrieve available shields from LangChain configuration.

    Fetches the list of configured shields for content moderation and safety.

    Returns:
        ShieldsResponse: Object containing the list of available shields.

    Raises:
        HTTPException: If unable to retrieve shields from LangChain.
    """
    check_configuration_loaded(configuration)

    langchain_config = configuration.langchain_configuration
    if langchain_config is None:
        raise HTTPException(
            status_code=503,
            detail={
                "backend_name": "LangChain",
                "response": "LangChain not configured",
            },
        )

    logger.info("Retrieving shields from LangChain configuration")

    try:
        # For now, return an empty list as shields are not yet implemented in LangChain
        # TODO: Implement shield support when LangChain content moderation is added
        shields = []
        return ShieldsResponse(shields=shields)

    except Exception as e:
        logger.error("Unable to retrieve shields from LangChain: %s", e)
        response = ServiceUnavailableResponse(backend_name="LangChain", cause=str(e))
        raise HTTPException(**response.model_dump()) from e
