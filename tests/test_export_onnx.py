import importlib.util
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def load_export_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "export_onnx.py"
    spec = importlib.util.spec_from_file_location("export_onnx", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_main_resolves_default_paths_from_project_root_when_cwd_changes(monkeypatch, tmp_path):
    export_onnx = load_export_module()
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)

    checkpoint = {
        "args": {"class_names": ["forklift_with_load", "forklift_empty"]},
        "model": {},
    }
    model = SimpleNamespace(
        load_state_dict=lambda *args, **kwargs: ([], []),
        export=lambda: None,
        eval=lambda: None,
    )
    onnx_model = object()

    with patch.object(
        export_onnx.torch, "load", return_value=checkpoint
    ) as load, patch.object(
        export_onnx, "build_args", return_value=Namespace()
    ), patch.dict(
        sys.modules,
        {"rfdetr.models.lwdetr": SimpleNamespace(build_model=lambda args: model)},
    ), patch.object(
        export_onnx.torch, "randn", return_value=object()
    ), patch.object(export_onnx.torch.onnx, "export") as export, patch.dict(
        sys.modules,
        {
            "onnx": SimpleNamespace(
                load=lambda path: onnx_model,
                checker=SimpleNamespace(check_model=lambda model: None),
            )
        },
    ):
        export_onnx.main()

    load.assert_called_once()
    assert (
        Path(load.call_args.args[0])
        == project_root / "models" / "checkpoint_best_regular.pth"
    )
    export.assert_called_once()
    assert Path(export.call_args.args[2]) == project_root / "models" / "inference_model.onnx"
