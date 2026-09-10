"""Structural privacy guarantee: the OCR and grading packages cannot talk to the network."""
import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
AIR_GAPPED_PACKAGES = ["ocr", "grading"]
NETWORK_MODULES = {
    "requests", "httpx", "urllib", "urllib3", "http", "socket", "aiohttp",
    "litellm", "ollama", "huggingface_hub",
}


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("package", AIR_GAPPED_PACKAGES)
def test_no_network_imports(package):
    files = sorted((SRC / package).rglob("*.py"))
    offenders = {
        str(f.relative_to(SRC)): sorted(_imported_roots(f) & NETWORK_MODULES)
        for f in files
    }
    assert not {k: v for k, v in offenders.items() if v}
