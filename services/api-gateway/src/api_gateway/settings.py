"""Settings for the api-gateway."""

from raglab_common.settings import BaseServiceSettings


class GatewaySettings(BaseServiceSettings):
    service_name: str = "api-gateway"
    port: int = 8000

    # Health-check timeout per downstream service (seconds)
    health_check_timeout: float = 3.0

    # Proxy request timeout (seconds)
    proxy_timeout: float = 120.0

    # How often to refresh downstream health cache (seconds)
    health_cache_ttl: float = 10.0

    # ── Auth / JWT (R7) ───────────────────────────────────────────────────────
    auth_enabled: bool = False        # True activates JWT validation at gateway
    auth_service_url: str = "http://auth:8012"
