import pytest


@pytest.fixture(autouse=True)
def normal_limits_by_default(monkeypatch):
    """Tests assume normal limits even if the developer's real .env sets LITE_MODE=1 (dotenv never overrides existing env vars)."""
    monkeypatch.setenv("LITE_MODE", "0")
