"""Tests for Redis ECL prediction cache."""
import json
import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from ai_surrogate.cache import ECLCache, format_cache_key
from conftest import FakeRedis


def test_format_cache_key_is_deterministic():
    assert format_cache_key(6.5, 5.25, 95.0) == "ecl_cache:6.5:5.25:95.0"
    assert format_cache_key(6.501, 5.249, 95.004) == "ecl_cache:6.5:5.25:95.0"


def test_cache_get_miss_when_empty():
    cache = ECLCache(enabled=True, ttl_seconds=3600, redis_client=FakeRedis())
    assert cache.get(4.0, 3.0, 100.0) is None


def test_cache_set_and_get_round_trip():
    fake = FakeRedis()
    cache = ECLCache(enabled=True, ttl_seconds=86400, redis_client=fake)

    cache.set(6.5, 5.25, 95.0, 4_307_526_656.0)
    assert cache.get(6.5, 5.25, 95.0) == 4_307_526_656.0

    key = format_cache_key(6.5, 5.25, 95.0)
    payload = json.loads(fake.store[key])
    assert payload["predicted_ecl"] == 4_307_526_656.0
    assert "timestamp" in payload
    assert fake.ttl[key] == 86400


def test_cache_disabled_skips_reads_and_writes():
    fake = FakeRedis()
    cache = ECLCache(enabled=False, ttl_seconds=86400, redis_client=fake)

    cache.set(4.0, 3.0, 100.0, 1.0)
    assert cache.get(4.0, 3.0, 100.0) is None
    assert fake.store == {}
