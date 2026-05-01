"""LangChain implementation for RAGs endpoints."""

from fastapi import HTTPException

from configuration import configuration
from log import get_logger
from models.responses import (
    NotFoundResponse,
    RAGInfoResponse,
    RAGListResponse,
)

logger = get_logger(__name__)


async def list_rags_langchain() -> RAGListResponse:
    """
    List all available RAGs using LangChain configuration.

    LangChain doesn't have a runtime RAG registry like Llama Stack,
    so this returns the configured BYOK RAGs from the configuration file.

    Returns:
        RAGListResponse: List of RAG identifiers from configuration.
    """
    logger.info("Retrieving RAGs from LangChain configuration")

    # Get BYOK RAGs from configuration
    byok_rags = configuration.configuration.byok_rag
    rag_ids = [rag.rag_id for rag in byok_rags]

    logger.info("Found %d configured RAGs", len(rag_ids))
    return RAGListResponse(rags=rag_ids)


async def get_rag_langchain(rag_id: str) -> RAGInfoResponse:
    """Retrieve a single RAG identified by its unique ID using LangChain configuration.

    LangChain doesn't have a runtime RAG registry, so this looks up
    the RAG in the configuration and returns its metadata.

    Parameters:
        rag_id: The RAG identifier.

    Returns:
        RAGInfoResponse: A single RAG's details.

    Raises:
        HTTPException: If RAG not found in configuration.
    """
    logger.info("Retrieving RAG %s from LangChain configuration", rag_id)

    # Find RAG in configuration
    byok_rags = configuration.configuration.byok_rag
    rag_config = next((rag for rag in byok_rags if rag.rag_id == rag_id), None)

    if rag_config is None:
        logger.error("RAG not found: %s", rag_id)
        response = NotFoundResponse(resource="rag", resource_id=rag_id)
        raise HTTPException(**response.model_dump())

    # Return RAG info from configuration
    # Note: LangChain doesn't track runtime stats like usage_bytes, created_at, etc.
    # so we return minimal metadata
    return RAGInfoResponse(
        id=rag_config.rag_id,
        name=rag_config.rag_id,  # Use rag_id as name if not explicitly set
        created_at=0,  # Not tracked in LangChain
        last_active_at=0,  # Not tracked in LangChain
        expires_at=None,  # Not applicable
        object="vector_store",
        status="active",  # Assume active if configured
        usage_bytes=0,  # Not tracked in LangChain
    )
