"""Llama Stack implementation for /info endpoint."""

from fastapi import HTTPException
from llama_stack_client import APIConnectionError

from client import AsyncLlamaStackClientHolder
from configuration import configuration
from log import get_logger
from models.responses import InfoResponse, ServiceUnavailableResponse
from version import __version__

logger = get_logger(__name__)


async def info_llama_stack() -> InfoResponse:
    """
    Retrieve service info from Llama Stack.

    Returns:
        InfoResponse: Service name, version, and Llama Stack version.

    Raises:
        HTTPException: If unable to connect to Llama Stack.
    """
    logger.info("Retrieving info from Llama Stack")

    try:
        client = AsyncLlamaStackClientHolder().get_client()
        llama_stack_version_object = await client.inspect.version()
        llama_stack_version = llama_stack_version_object.version
        logger.debug("Service name: %s", configuration.configuration.name)
        logger.debug("Service version: %s", __version__)
        logger.debug("Llama Stack version: %s", llama_stack_version)
        return InfoResponse(
            name=configuration.configuration.name,
            service_version=__version__,
            llama_stack_version=llama_stack_version,
        )
    except APIConnectionError as e:
        logger.error("Unable to connect to Llama Stack: %s", e)
        response = ServiceUnavailableResponse(backend_name="Llama Stack", cause=str(e))
        raise HTTPException(**response.model_dump()) from e
