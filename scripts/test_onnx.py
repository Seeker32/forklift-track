"""Test RF-DETR ONNX model on a video.

Usage: uv run python scripts/test_onnx.py [--video PATH] [--frames N]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.detector_onnx import ONNXForkliftDetector


def draw_detections(frame: np.ndarray, detections: list[dict], color_map: dict) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        bbox = det["bbox"]
        class_name = det["class_name"]
        score = det["score"]
        color = color_map.get(class_name, (0, 255, 0))
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(out, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Test RF-DETR ONNX model")
    parser.add_argument("--video", default="videos/Videos_20260430185002.mp4")
    parser.add_argument("--frames", type=int, default=0, help="Max frames to process (0=all)")
    parser.add_argument("--output", default="outputs/onnx_test_output.mp4")
    parser.add_argument("--confidence", type=float, default=0.3)
    args = parser.parse_args()

    onnx_path = "models/inference_model.onnx"
    if not Path(onnx_path).exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    print(f"Loading ONNX model: {onnx_path}")
    detector = ONNXForkliftDetector(
        onnx_path=onnx_path,
        confidence=args.confidence,
        class_names=["forklift_with_load", "forklift_empty"],
    )
    print(f"Classes: {detector.class_names}")

    print(f"Opening video: {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {width}x{height} @ {fps:.1f} fps")

    Path("outputs").mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps or 30.0, (width, height))

    color_map = {
        "forklift_with_load": (0, 140, 255),  # orange
        "forklift_empty": (0, 255, 0),  # green
    }

    frame_count = 0
    total_detections = 0
    class_counts = {"forklift_with_load": 0, "forklift_empty": 0}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1
            detections = detector.detect(frame)
            total_detections += len(detections)

            for d in detections:
                class_counts[d["class_name"]] = class_counts.get(d["class_name"], 0) + 1

            annotated = draw_detections(frame, detections, color_map)
            writer.write(annotated)

            if frame_count % 30 == 0 or detections:
                names = ", ".join(f"{d['class_name']}:{d['score']:.2f}" for d in detections)
                print(f"  frame {frame_count}: {len(detections)} detections: {names}")

            if args.frames and frame_count >= args.frames:
                break
    finally:
        cap.release()
        writer.release()

    print(f"\nDone. Processed {frame_count} frames.")
    print(f"Total detections: {total_detections}")
    for name, count in class_counts.items():
        print(f"  {name}: {count}")
    print(f"Output video: {args.output}")


if __name__ == "__main__":
    main()
