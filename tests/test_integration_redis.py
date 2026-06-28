"""Optional integration tests that require a live Redis instance."""
import sys
from pathlib import Path
from uuid import uuid4

import pytest

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from ai_surrogate.cache import ECLCache, format_cache_key


pytestmark = pytest.mark.integration


def _redis_available() -> bool:
    cache = ECLCache.connect()
    return cache.available


@pytest.mark.skipif(not _redis_available(), reason="Redis is not running")
def test_ecl_cache_round_trip_against_live_redis():
    cache = ECLCache.connect()
    unemployment = 4.0 + (uuid4().int % 100) / 1000
    interest = 3.0 + (uuid4().int % 100) / 1000
    hpi = 100.0 + (uuid4().int % 100) / 10

    assert cache.get(unemployment, interest, hpi) is None
    cache.set(unemployment, interest, hpi, 1_234_567.89)
    assert cache.get(unemployment, interest, hpi) == pytest.approx(1_234_567.89)

    key = format_cache_key(unemployment, interest, hpi)
    assert key.startswith("ecl_cache:")
