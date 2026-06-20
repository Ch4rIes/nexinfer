import tomllib
from pathlib import Path


def test_optional_dependencies_expose_safetensors_loader_extra() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    optional_dependencies = pyproject["project"]["optional-dependencies"]

    assert "safetensors>=0.4" in optional_dependencies["safetensors"]
    assert "safetensors>=0.4" in optional_dependencies["torch"]
