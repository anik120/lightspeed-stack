"""Llama Stack implementation for /shields endpoint."""

from fastapi import HTTPException
from llama_stack_client import APIConnectionError

from client import AsyncLlamaStackClientHolder
from configuration import configuration
from log import get_logger
from models.responses import ServiceUnavailableResponse, ShieldsResponse
from utils.endpoints import check_configuration_loaded

logger = get_logger(__name__)


async def shields_llama_stack() -> ShieldsResponse:
    """
    Retrieve available shields from Llama Stack.

    Fetches the list of shields from the Llama Stack service.

    Returns:
        ShieldsResponse: Object containing the list of available shields.

    Raises:
        HTTPException: If unable to connect to Llama Stack or retrieval fails.
    """
    check_configuration_loaded(configuration)

    llama_stack_configuration = configuration.llama_stack_configuration
    logger.info("Llama stack config: %s", llama_stack_configuration)

    try:
        client = AsyncLlamaStackClientHolder().get_client()
        shields = await client.shields.list()
        s = [dict(s) for s in shields]
        return ShieldsResponse(shields=s)

    except APIConnectionError as e:
        logger.error("Unable to connect to Llama Stack: %s", e)
        response = ServiceUnavailableResponse(backend_name="Llama Stack", cause=str(e))
        raise HTTPException(**response.model_dump()) from e
