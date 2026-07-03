"""Redis cache for surrogate ECL predictions (separate from simulation job queues)."""
import json
import time

import redis

from risk_engine.config import ECL_CACHE_ENABLED, ECL_CACHE_TTL, REDIS_HOST, REDIS_PORT

CACHE_KEY_PREFIX = "ecl_cache"

def format_cache_key(
    unemployment: float,
    interest_rate: float,
    housing_price_index: float,
) -> str:
    """Build a deterministic Redis key from clipped macro coordinates."""
    return (
        f"{CACHE_KEY_PREFIX}:"
        f"{round(unemployment, 2)}:"
        f"{round(interest_rate, 2)}:"
        f"{round(housing_price_index, 2)}"
    )

class ECLCache:
    """Optional Redis cache for predicted ECL values."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        ttl_seconds: int | None = None,
        redis_client: redis.Redis | None = None,
    ) -> None:
        self.enabled = ECL_CACHE_ENABLED if enabled is None else enabled
        self.ttl_seconds = ECL_CACHE_TTL if ttl_seconds is None else ttl_seconds
        self._client = redis_client
        self._available = redis_client is not None

    @classmethod
    def connect(cls, **kwargs) -> "ECLCache":
        """Create a cache instance and probe Redis when enabled."""
        cache = cls(**kwargs)
        if cache.enabled and cache._client is None:
            try:
                client = redis.Redis(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    db=0,
                    decode_responses=True,
                    socket_connect_timeout=2,
                )
                client.ping()
                cache._client = client
                cache._available = True
            except redis.RedisError:
                cache._available = False
        return cache

    @property
    def available(self) -> bool:
        return self.enabled and self._available and self._client is not None

    def get(
        self,
        unemployment: float,
        interest_rate: float,
        housing_price_index: float,
    ) -> float | None:
        if not self.available:
            return None

        key = format_cache_key(unemployment, interest_rate, housing_price_index)
        try:
            raw = self._client.get(key)
        except redis.RedisError:
            self._available = False
            return None

        if raw is None:
            return None

        try:
            payload = json.loads(raw)
            return float(payload["predicted_ecl"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def set(
        self,
        unemployment: float,
        interest_rate: float,
        housing_price_index: float,
        predicted_ecl: float,
    ) -> None:
        if not self.available:
            return

        key = format_cache_key(unemployment, interest_rate, housing_price_index)
        payload = json.dumps(
            {
                "predicted_ecl": predicted_ecl,
                "timestamp": time.time(),
            }
        )
        try:
            self._client.setex(key, self.ttl_seconds, payload)
        except redis.RedisError:
            self._available = False
