"""The API reference lists every public name exactly once."""

import importlib.util
from collections import Counter
from pathlib import Path

import peyesim

BUILD_DOCS = Path(__file__).resolve().parents[1] / "tools" / "build_docs.py"


def _sections():
    spec = importlib.util.spec_from_file_location("build_docs", BUILD_DOCS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.REFERENCE_SECTIONS


def test_reference_sections_cover_public_api_once():
    listed = Counter(name for *_, names in _sections() for name in names)
    public = set(peyesim.__all__) - {"__version__"}
    assert sorted(public - set(listed)) == []
    assert sorted(set(listed) - public) == []
    assert [name for name, n in listed.items() if n > 1] == []
