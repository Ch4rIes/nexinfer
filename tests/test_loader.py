from pathlib import Path
from typing import Any

from nexinfer import default_weight_loader, load_model


class FakeData:
    def __init__(self) -> None:
        self.copied: list[Any] = []

    def copy_(self, value: Any) -> None:
        self.copied.append(value)


class FakeParam:
    def __init__(self, weight_loader: Any | None = None) -> None:
        self.data = FakeData()
        if weight_loader is not None:
            self.weight_loader = weight_loader


class FakeModel:
    def __init__(self) -> None:
        self.params: dict[str, FakeParam] = {}
        self.packed_modules_mapping: dict[str, tuple[str, str]] = {}
        self.requested_params: list[str] = []

    def get_parameter(self, name: str) -> FakeParam:
        self.requested_params.append(name)
        return self.params[name]


class FakeSafeTensorFile:
    def __init__(self, tensors: dict[str, Any]) -> None:
        self.tensors = tensors

    def __enter__(self) -> "FakeSafeTensorFile":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        pass

    def keys(self) -> list[str]:
        return list(self.tensors)

    def get_tensor(self, name: str) -> Any:
        return self.tensors[name]


def test_default_weight_loader_copies_into_parameter_data() -> None:
    param = FakeParam()

    default_weight_loader(param, "tensor")

    assert param.data.copied == ["tensor"]


def test_load_model_loads_safetensors_parameters(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").touch()
    model = FakeModel()
    model.params["layers.0.weight"] = FakeParam()
    opened_files: list[tuple[str, str, str]] = []

    def safe_open(path: str, framework: str, device: str) -> FakeSafeTensorFile:
        opened_files.append((Path(path).name, framework, device))
        return FakeSafeTensorFile({"layers.0.weight": "tensor"})

    load_model(model, str(tmp_path), safe_open_fn=safe_open)

    assert opened_files == [("model.safetensors", "pt", "cpu")]
    assert model.requested_params == ["layers.0.weight"]
    assert model.params["layers.0.weight"].data.copied == ["tensor"]


def test_load_model_uses_packed_module_weight_loader(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").touch()
    loaded: list[tuple[FakeParam, Any, str]] = []

    def packed_loader(param: FakeParam, tensor: Any, shard_id: str) -> None:
        loaded.append((param, tensor, shard_id))

    model = FakeModel()
    model.packed_modules_mapping = {"qkv_proj": ("q_proj", "q")}
    model.params["layers.0.q_proj.weight"] = FakeParam(weight_loader=packed_loader)

    def safe_open(path: str, framework: str, device: str) -> FakeSafeTensorFile:
        return FakeSafeTensorFile({"layers.0.qkv_proj.weight": "packed-tensor"})

    load_model(model, str(tmp_path), safe_open_fn=safe_open)

    param = model.params["layers.0.q_proj.weight"]
    assert model.requested_params == ["layers.0.q_proj.weight"]
    assert loaded == [(param, "packed-tensor", "q")]
    assert param.data.copied == []
