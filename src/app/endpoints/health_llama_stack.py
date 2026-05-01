"""Llama Stack implementation for health endpoints."""

from enum import Enum

from fastapi import Response, status
from llama_stack_client import APIConnectionError

from client import AsyncLlamaStackClientHolder
from log import get_logger
from models.responses import ProviderHealthStatus, ReadinessResponse

logger = get_logger(__name__)


# HealthStatus enum was removed from llama_stack in newer versions
# Defining locally for compatibility
class HealthStatus(str, Enum):
    """Health status enum for provider health checks."""

    OK = "ok"
    ERROR = "Error"
    NOT_IMPLEMENTED = "not_implemented"
    HEALTHY = "healthy"
    UNKNOWN = "unknown"


async def get_providers_health_statuses() -> list[ProviderHealthStatus]:
    """
    Retrieve the health status of all configured providers from Llama Stack.

    Returns:
        list[ProviderHealthStatus]: A list containing the health
        status of each provider. If provider health cannot be
        determined, returns a single entry indicating an error.
    """
    try:
        client = AsyncLlamaStackClientHolder().get_client()

        providers = await client.providers.list()
        logger.debug("Found %d providers", len(providers))

        return [
            ProviderHealthStatus(
                provider_id=provider.provider_id,
                status=str(provider.health.get("status", "unknown")),
                message=str(provider.health.get("message", "")),
            )
            for provider in providers
        ]

    except APIConnectionError as e:
        logger.error("Failed to check providers health: %s", e)
        return [
            ProviderHealthStatus(
                provider_id="unknown",
                status=HealthStatus.ERROR.value,
                message=f"Failed to initialize health check: {e!s}",
            )
        ]


async def readiness_llama_stack(response: Response) -> ReadinessResponse:
    """
    Check service readiness using Llama Stack provider health.

    If any provider reports an error status, responds with HTTP 503
    and details of unhealthy providers; otherwise, indicates the
    service is ready.

    Parameters:
        response: The outgoing HTTP response (status code modified if unhealthy).

    Returns:
        ReadinessResponse: Object with `ready` indicating overall readiness,
        `reason` explaining the outcome, and `providers` containing the list of
        unhealthy ProviderHealthStatus entries (empty when ready).
    """
    logger.info("Checking readiness via Llama Stack")

    provider_statuses = await get_providers_health_statuses()

    # Check if any provider is unhealthy (not counting not_implemented as unhealthy)
    unhealthy_providers = [
        p for p in provider_statuses if p.status == HealthStatus.ERROR.value
    ]

    if unhealthy_providers:
        ready = False
        unhealthy_provider_names = [p.provider_id for p in unhealthy_providers]
        reason = f"Providers not healthy: {', '.join(unhealthy_provider_names)}"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    else:
        ready = True
        reason = "All providers are healthy"

    return ReadinessResponse(ready=ready, reason=reason, providers=unhealthy_providers)
