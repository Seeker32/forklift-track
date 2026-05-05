from __future__ import annotations

import argparse
import sys

from src.line_tool import select_line_from_source
from src.pipeline import load_config, run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Forklift entry/exit direction tracking")
    parser.add_argument("--config", default="config/cameras.yaml", help="Path to camera configuration file")
    parser.add_argument("--model-path", help="Override detector model path for all configured cameras")
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


if __name__ == "__main__":
    main()
