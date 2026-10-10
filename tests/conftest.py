"""Test-suite settings shared by every test.

Locally, tests that need something absent (Spark on some Windows setups, the
lake when Docker isn't running) skip themselves. In CI everything is provided,
so a skip means something is broken: with HMIS_FAIL_ON_SKIP=1 set, any skipped
test fails the run.
"""

import os

import pytest


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not os.environ.get("HMIS_FAIL_ON_SKIP"):
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = reporter.stats.get("skipped", []) if reporter else []
    if skipped:
        if reporter:
            reporter.write_line(
                f"HMIS_FAIL_ON_SKIP: {len(skipped)} test(s) skipped, which CI doesn't allow",
                red=True,
            )
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
