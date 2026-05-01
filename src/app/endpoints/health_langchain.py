"""LangChain implementation for health endpoints."""

from fastapi import Response, status

from configuration import configuration
from langchain_client import LLMProviderRegistry
from log import get_logger
from models.responses import ProviderHealthStatus, ReadinessResponse

logger = get_logger(__name__)


async def readiness_langchain(response: Response) -> ReadinessResponse:
    """
    Check service readiness using LangChain provider health.

    Checks if LangChain providers can be initialized. If any provider
    fails to initialize, responds with HTTP 503 and details of
    unhealthy providers.

    Parameters:
        response: The outgoing HTTP response (status code modified if unhealthy).

    Returns:
        ReadinessResponse: Object with `ready` indicating overall readiness,
        `reason` explaining the outcome, and `providers` containing the list of
        unhealthy ProviderHealthStatus entries (empty when ready).
    """
    logger.info("Checking readiness via LangChain")

    langchain_config = configuration.langchain_configuration
    provider_statuses: list[ProviderHealthStatus] = []
    unhealthy_providers: list[ProviderHealthStatus] = []

    # Try to initialize the registry
    try:
        registry = LLMProviderRegistry()
        await registry.initialize(langchain_config)
    except Exception as e:
        logger.error("Failed to initialize LangChain registry: %s", e)
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(
            ready=False,
            reason="Failed to initialize LangChain registry",
            providers=[
                ProviderHealthStatus(
                    provider_id="langchain",
                    status="error",
                    message=f"Registry initialization failed: {e!s}",
                )
            ],
        )

    # Check health of each configured provider
    for provider_name in langchain_config.providers.keys():
        try:
            # Try to get the provider - this validates it can be initialized
            _ = await registry.get_provider(f"{provider_name}/{langchain_config.default_model}")
            provider_statuses.append(
                ProviderHealthStatus(
                    provider_id=provider_name,
                    status="healthy",
                    message="Provider initialized successfully",
                )
            )
            logger.debug("Provider %s is healthy", provider_name)
        except Exception as e:
            logger.error("Provider %s is unhealthy: %s", provider_name, e)
            unhealthy_provider = ProviderHealthStatus(
                provider_id=provider_name,
                status="error",
                message=f"Failed to initialize: {e!s}",
            )
            provider_statuses.append(unhealthy_provider)
            unhealthy_providers.append(unhealthy_provider)

    if unhealthy_providers:
        ready = False
        unhealthy_provider_names = [p.provider_id for p in unhealthy_providers]
        reason = f"Providers not healthy: {', '.join(unhealthy_provider_names)}"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    else:
        ready = True
        reason = "All providers are healthy"

    return ReadinessResponse(ready=ready, reason=reason, providers=unhealthy_providers)
