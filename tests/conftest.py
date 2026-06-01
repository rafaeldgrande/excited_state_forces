"""
Pytest configuration: add module directories to sys.path, and provide a shared
fixture that resets the global config dict between tests.
"""
import sys
import os
import pytest

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
for subdir in ('main', 'common', 'elph'):
    sys.path.insert(0, os.path.abspath(os.path.join(REPO_ROOT, subdir)))
# Also insert repo root so `from common import ...` works for elph scripts
sys.path.insert(0, os.path.abspath(REPO_ROOT))


@pytest.fixture(autouse=True)
def reset_config():
    """Restore the global config dict to its original state after every test."""
    from excited_forces_config import config
    saved = config.copy()
    yield
    config.clear()
    config.update(saved)
