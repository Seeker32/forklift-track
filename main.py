from __future__ import annotations

import argparse
import sys

from src.line_tool import select_line_from_source
from src.pipeline import load_config, run


def main(argv: list[str] | None = None) -> None:
    _preload_onnxruntime_dlls()

    parser = argparse.ArgumentParser(description="Forklift entry/exit direction tracking")
    parser.add_argument("--config", default="config/cameras.yaml", help="Path to camera configuration file")
    parser.add_argument("--model-path", help="Override detector model path for all configured cameras")
    parser.add_argument("--trt-fp16", action="store_true", help="Enable TensorRT FP16 execution for ONNX models")
    parser.add_argument(
        "--trt-max-workspace-size",
        type=int,
        help="Override TensorRT max workspace size in bytes for ONNX models",
    )
    parser.add_argument(
        "--trt-builder-optimization-level",
        type=int,
        help="Override TensorRT builder optimization level for ONNX models",
    )
    parser.add_argument("--select-line", help="Open a video source and click two points to print line config")
    parser.add_argument("--debug", action="store_true", help="Print frame, track, and crossing-zone diagnostics")
    parser.add_argument("--debug-every", type=int, default=1, help="Print debug summaries every N frames")
    parser.add_argument("--debug-video", action="store_true", help="Write annotated full-length debug videos to outputs/")
    args = parser.parse_args(argv)

    if args.select_line:
        config = load_config(args.config)
        line_width = _line_width_for_source(config, args.select_line)
        select_line_from_source(args.select_line, line_width=line_width)
        return

    run_kwargs = {}
    if args.model_path is not None:
        run_kwargs["model_path"] = args.model_path
    if args.trt_fp16:
        run_kwargs["trt_fp16"] = True
    if args.trt_max_workspace_size is not None:
        run_kwargs["trt_max_workspace_size"] = args.trt_max_workspace_size
    if args.trt_builder_optimization_level is not None:
        run_kwargs["trt_builder_optimization_level"] = args.trt_builder_optimization_level

    try:
        if args.debug and args.debug_video:
            run(args.config, debug=True, debug_every=args.debug_every, debug_video=True, **run_kwargs)
        elif args.debug:
            run(args.config, debug=True, debug_every=args.debug_every, **run_kwargs)
        elif args.debug_video:
            run(args.config, debug_video=True, **run_kwargs)
        else:
            run(args.config, **run_kwargs)
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        sys.exit(0)


def _line_width_for_source(config: dict, source: str) -> float:
    cameras = config["cameras"]
    for camera in cameras:
        if camera["source"] == source:
            return float(camera["line_width"])
    return float(cameras[0]["line_width"])


def _preload_onnxruntime_dlls() -> None:
    try:
        ort = __import__("onnxruntime")
    except ModuleNotFoundError:
        return

    preload_dlls = getattr(ort, "preload_dlls", None)
    if callable(preload_dlls):
        preload_dlls()


if __name__ == "__main__":
    main()
