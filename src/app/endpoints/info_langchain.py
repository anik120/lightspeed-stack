"""LangChain implementation for /info endpoint."""

from fastapi import HTTPException

from configuration import configuration
from log import get_logger
from models.responses import InfoResponse, ServiceUnavailableResponse
from utils.endpoints import check_configuration_loaded
from version import __version__

logger = get_logger(__name__)


async def info_langchain() -> InfoResponse:
    """
    Retrieve service info from LangChain configuration.

    Returns:
        InfoResponse: Service name, version, and LangChain version.

    Raises:
        HTTPException: If unable to retrieve info from LangChain.
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

    logger.info("Retrieving info from LangChain")

    try:
        # Get LangChain version from the installed package
        import langchain_core

        langchain_version = langchain_core.__version__

        logger.debug("Service name: %s", configuration.configuration.name)
        logger.debug("Service version: %s", __version__)
        logger.debug("LangChain version: %s", langchain_version)

        return InfoResponse(
            name=configuration.configuration.name,
            service_version=__version__,
            llama_stack_version=langchain_version,  # Using same field for backend version
        )
    except Exception as e:
        logger.error("Unable to retrieve info from LangChain: %s", e)
        response = ServiceUnavailableResponse(backend_name="LangChain", cause=str(e))
        raise HTTPException(**response.model_dump()) from e
