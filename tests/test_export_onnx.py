import importlib.util
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


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
    assert export.call_args.kwargs["input_names"] == ["input"]
    assert export.call_args.kwargs["output_names"] == ["pred_boxes", "pred_logits"]
    assert "target_sizes" not in export.call_args.kwargs["dynamic_axes"]


def test_load_model_uses_rfdetr_small_compatible_defaults():
    export_onnx = load_export_module()
    checkpoint = {"args": {}, "model": {}}
    captured_args = {}
    model = SimpleNamespace(
        load_state_dict=lambda *args, **kwargs: ([], []),
        export=lambda: None,
        eval=lambda: None,
    )

    def fake_build_model(args):
        captured_args["args"] = args
        return model

    with patch.object(export_onnx.torch, "load", return_value=checkpoint), patch.dict(
        sys.modules,
        {"rfdetr.models.lwdetr": SimpleNamespace(build_model=fake_build_model)},
    ):
        export_onnx.load_model(Path("/tmp/checkpoint.pth"))

    args = captured_args["args"]
    assert args.encoder == "dinov2_windowed_small"
    assert args.out_feature_indexes == [3, 6, 9, 12]
    assert args.patch_size == 16
    assert args.num_windows == 2
    assert args.positional_encoding_size == 32
    assert args.pretrain_weights == "/tmp/checkpoint.pth"
    assert args.shape == (export_onnx.RESOLUTION, export_onnx.RESOLUTION)


def test_main_calls_model_export_before_torch_onnx_export(tmp_path, monkeypatch):
    export_onnx = load_export_module()
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)

    model = SimpleNamespace(
        load_state_dict=lambda *args, **kwargs: ([], []),
        eval=lambda: None,
    )
    model.export = Mock()

    checkpoint = {"args": {}, "model": {}}

    with patch.object(export_onnx.torch, "load", return_value=checkpoint), patch.object(
        export_onnx, "build_args", return_value=Namespace()
    ), patch.dict(
        sys.modules,
        {"rfdetr.models.lwdetr": SimpleNamespace(build_model=lambda args: model)},
    ), patch.object(
        export_onnx.torch, "randn", return_value=object()
    ), patch.object(export_onnx.torch.onnx, "export") as export:
        export_onnx.main()

    model.export.assert_called_once_with()
    assert Path(export.call_args.args[2]) == project_root / "models" / "inference_model.onnx"


def test_patch_projector_layer_norm_for_onnx_uses_static_normalized_shape():
    export_onnx = load_export_module()

    class FakeLayerNorm:
        def __init__(self, channels, eps=1e-6):
            self.weight = torch.ones(channels)
            self.bias = torch.zeros(channels)
            self.eps = eps
            self.normalized_shape = (channels,)

        def forward(self, x):
            raise AssertionError("expected patched forward")

    fake_projector = SimpleNamespace(LayerNorm=FakeLayerNorm)
    layer_norm = FakeLayerNorm(4)
    x = torch.randn(1, 4, 2, 2)

    with patch.dict(sys.modules, {"rfdetr.models.backbone.projector": fake_projector}), patch.object(
        export_onnx.F, "layer_norm", side_effect=lambda x, shape, *rest: x
    ) as layer_norm_fn:
        export_onnx._patch_projector_layer_norm_for_onnx()
        fake_projector.LayerNorm.forward(layer_norm, x)

    assert layer_norm_fn.call_args.args[1] == layer_norm.normalized_shape
