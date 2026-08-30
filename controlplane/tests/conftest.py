import os
import tempfile

os.environ.setdefault("CP_LLM_MOCK", "1")
os.environ.setdefault("CP_CACHE_ENABLED", "true")
# tests are functional, not latency benchmarks - give branches room for cold loads
os.environ.setdefault("CP_PERF_BUDGET_S", "60")
os.environ.setdefault("CP_RESP_BUDGET_S", "60")
# isolate the semantic cache from the on-disk demo cache
os.environ.setdefault("CP_CACHE_DIR", tempfile.mkdtemp(prefix="cp_test_cache_"))
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_cache():
    from controlplane.cache import get_cache

    get_cache().clear()
    yield
    get_cache().clear()


def pytest_sessionfinish(session, exitstatus):
    """torch/transformers/faiss can segfault during interpreter shutdown on Windows
    AFTER every test has run. Print an explicit result line and hard-exit past
    those atexit handlers on a clean pass. On failure, shut down normally so
    tracebacks are not truncated."""
    import os
    import sys

    tr = session.config.pluginmanager.get_plugin("terminalreporter")
    if tr is not None:
        p = len(tr.stats.get("passed", []))
        f = len(tr.stats.get("failed", [])) + len(tr.stats.get("error", []))
        print(f"\n[conftest] {p} passed, {f} failed  (exitstatus={exitstatus})")
    sys.stdout.flush()
    sys.stderr.flush()
    if sys.platform.startswith("win") and int(exitstatus) == 0:
        os._exit(0)
