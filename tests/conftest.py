from collections.abc import Iterator

import pytest
import structlog


@pytest.fixture(autouse=True)
def reset_structlog() -> Iterator[None]:
    # configure_logging() binds to the current sys.stderr, which pytest swaps per test.
    yield
    structlog.reset_defaults()
