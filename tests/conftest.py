import pytest


@pytest.fixture(autouse=True)
def normal_limits_by_default(monkeypatch):
    """Tests assume normal limits even if the developer's real .env sets LITE_MODE=1 (dotenv never overrides existing env vars)."""
    monkeypatch.setenv("LITE_MODE", "0")


@pytest.fixture(autouse=True)
def no_real_tavily_key_by_default(monkeypatch):
    """
    Tests must never make a live Tavily call with a developer's real key from .env.

    setenv to "" (not delenv): dotenv only fills in a var that is absent from os.environ, so
    delenv would let ``tools.cve``'s own ``load_dotenv()`` reload a real key. Tests that need a
    key set it explicitly with ``monkeypatch.setenv("TAVILY_API_KEY", ...)``, which overrides this.
    """
    monkeypatch.setenv("TAVILY_API_KEY", "")
