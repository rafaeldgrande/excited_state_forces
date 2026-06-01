"""
Pytest configuration: add main/ and common/ to sys.path so their modules are importable,
and provide a shared fixture that resets the global config dict between tests.
"""
import sys
import os
import pytest

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.abspath(os.path.join(REPO_ROOT, 'main')))
sys.path.insert(0, os.path.abspath(os.path.join(REPO_ROOT, 'common')))


@pytest.fixture(autouse=True)
def reset_config():
    """Restore the global config dict to its original state after every test."""
    from excited_forces_config import config
    saved = config.copy()
    yield
    config.clear()
    config.update(saved)
