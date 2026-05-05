from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


DEFAULT_MODEL_PATH = "models/model.onnx"
DEFAULT_VIDEO_PATH = "42346d5303cc0df1099faa1821945c25.mp4"
DEFAULT_IMAGE_PATH = "input/forklift-operator.jpg"
DEFAULT_OUTPUT_PATH = "output/detected_42346d5303cc0df1099faa1821945c25.mp4"
DEFAULT_IMAGE_OUTPUT_PATH = "output/detected_forklift-operator.jpg"
DEFAULT_RESOLUTION = 512
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_PROGRESS_INTERVAL = 100
DEFAULT_NUM_SELECT = 300
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_NAMES = {0: "forklift_with_load", 1: "forklift_empty"}
COLORS = {0: (0, 255, 0), 1: (0, 0, 255)}


@dataclass
class InferenceConfig:
    model_path: str = DEFAULT_MODEL_PATH
    video_path: str = DEFAULT_VIDEO_PATH
    image_path: str = DEFAULT_IMAGE_PATH
    output_path: str = DEFAULT_OUTPUT_PATH
    resolution: int = DEFAULT_RESOLUTION
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    progress_interval: int = DEFAULT_PROGRESS_INTERVAL


ort.preload_dlls()

def create_session(model_path: str) -> ort.InferenceSession:
    print(f"Loading ONNX model: {model_path}")
    return ort.InferenceSession(
        model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )


def preprocess(frame_bgr: np.ndarray, resolution: int) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    img = resized.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)


def box_cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    cx = boxes[:, 0]
    cy = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    return np.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), axis=1)


def postprocess_deploy_outputs(
    scores: np.ndarray,
    labels: np.ndarray,
    boxes: np.ndarray,
    orig_w: int,
    orig_h: int,
    confidence_threshold: float,
) -> list[tuple[int, int, int, int, float, int]]:
    batch_scores = scores[0]
    batch_labels = labels[0]
    batch_boxes = boxes[0]

    detections = []
    for score, cls, box in zip(batch_scores, batch_labels, batch_boxes):
        if score < confidence_threshold:
            continue

        x1 = int(np.clip(box[0], 0, orig_w - 1))
        y1 = int(np.clip(box[1], 0, orig_h - 1))
        x2 = int(np.clip(box[2], 0, orig_w - 1))
        y2 = int(np.clip(box[3], 0, orig_h - 1))
        detections.append((x1, y1, x2, y2, float(score), int(cls)))

    return detections


def postprocess(
    pred_logits: np.ndarray,
    pred_boxes: np.ndarray,
    orig_w: int,
    orig_h: int,
    confidence_threshold: float,
) -> list[tuple[int, int, int, int, float, int]]:
    logits = pred_logits[0]
    boxes = pred_boxes[0]
    probs = 1.0 / (1.0 + np.exp(-logits))
    num_classes = len(CLASS_NAMES)

    if probs.shape[1] > num_classes:
        probs = probs[:, :num_classes]

    flat_probs = probs.reshape(-1)
    num_select = min(DEFAULT_NUM_SELECT, flat_probs.shape[0])
    topk_indices = np.argpartition(-flat_probs, num_select - 1)[:num_select]
    topk_scores = flat_probs[topk_indices]
    order = np.argsort(-topk_scores)
    topk_indices = topk_indices[order]
    topk_scores = topk_scores[order]

    topk_boxes = topk_indices // probs.shape[1]
    labels = topk_indices % probs.shape[1]
    selected_boxes = box_cxcywh_to_xyxy(boxes[topk_boxes])

    detections = []
    for score, cls, box in zip(topk_scores, labels, selected_boxes):
        if score < confidence_threshold:
            continue

        x1 = int(np.clip(box[0] * orig_w, 0, orig_w - 1))
        y1 = int(np.clip(box[1] * orig_h, 0, orig_h - 1))
        x2 = int(np.clip(box[2] * orig_w, 0, orig_w - 1))
        y2 = int(np.clip(box[3] * orig_h, 0, orig_h - 1))
        detections.append((x1, y1, x2, y2, float(score), int(cls)))

    return detections


def run_session(
    session: ort.InferenceSession,
    inp: np.ndarray,
    orig_w: int,
    orig_h: int,
    confidence_threshold: float,
) -> list[tuple[int, int, int, int, float, int]]:
    output_names = {output.name for output in session.get_outputs()}

    if {"scores", "labels", "boxes"}.issubset(output_names):
        scores, labels, boxes = session.run(
            ["scores", "labels", "boxes"],
            {
                "input": inp,
                "target_sizes": np.array([[orig_h, orig_w]], dtype=np.int64),
            },
        )
        return postprocess_deploy_outputs(
            scores,
            labels,
            boxes,
            orig_w,
            orig_h,
            confidence_threshold,
        )

    pred_logits, pred_boxes = session.run(["pred_logits", "pred_boxes"], {"input": inp})
    return postprocess(
        pred_logits,
        pred_boxes,
        orig_w,
        orig_h,
        confidence_threshold,
    )


def draw_detections(
    frame: np.ndarray,
    detections: list[tuple[int, int, int, int, float, int]],
) -> np.ndarray:
    for x1, y1, x2, y2, score, cls in detections:
        color = COLORS.get(cls, (255, 255, 255))
        label = f"{CLASS_NAMES.get(cls, 'unknown')} {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_top = max(y1 - th - 4, 0)
        cv2.rectangle(frame, (x1, label_top), (x1 + tw + 4, max(y1, th + 4)), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 2, max(y1 - 4, th)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    return frame


def format_console_output(
    frame_idx: int,
    detections: list[tuple[int, int, int, int, float, int]],
) -> str:
    if not detections:
        return f"Frame {frame_idx}: no detections"

    parts = []
    for x1, y1, x2, y2, score, cls in detections:
        class_name = CLASS_NAMES.get(cls, "unknown")
        parts.append(f"{class_name}@{score:.2f}[{x1},{y1},{x2},{y2}]")
    return f"Frame {frame_idx}: " + "; ".join(parts)


def run_video_inference(config: InferenceConfig) -> str:
    session = create_session(config.model_path)
    cap = cv2.VideoCapture(config.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {config.video_path}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {orig_w}x{orig_h}, {fps:.1f}fps, {total_frames} frames")

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (orig_w, orig_h))

    print("Running inference in video mode...")
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            inp = preprocess(frame, config.resolution)
            detections = run_session(
                session,
                inp,
                orig_w,
                orig_h,
                config.confidence_threshold,
            )
            writer.write(draw_detections(frame, detections))

            frame_idx += 1
            if config.progress_interval > 0 and frame_idx % config.progress_interval == 0:
                print(f"  Frame {frame_idx}/{total_frames}")
    finally:
        cap.release()
        writer.release()

    print(f"Done. Output saved to: {output_path}")
    return str(output_path)


def run_console_inference(config: InferenceConfig) -> None:
    session = create_session(config.model_path)
    cap = cv2.VideoCapture(config.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {config.video_path}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {orig_w}x{orig_h}, {fps:.1f}fps, {total_frames} frames")
    print("Running inference in console mode...")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            inp = preprocess(frame, config.resolution)
            detections = run_session(
                session,
                inp,
                orig_w,
                orig_h,
                config.confidence_threshold,
            )

            frame_idx += 1
            print(format_console_output(frame_idx, detections))
    finally:
        cap.release()

    print("Done.")


def run_image_inference(config: InferenceConfig) -> str:
    session = create_session(config.model_path)
    image_path = Path(config.image_path)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Unable to open image: {config.image_path}")

    orig_h, orig_w = frame.shape[:2]
    print(f"Image: {orig_w}x{orig_h}")
    print("Running inference in image mode...")

    inp = preprocess(frame, config.resolution)
    detections = run_session(
        session,
        inp,
        orig_w,
        orig_h,
        config.confidence_threshold,
    )
    print(format_console_output(1, detections))

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = draw_detections(frame.copy(), detections)
    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(f"Unable to write image: {output_path}")

    print(f"Done. Output saved to: {output_path}")
    return str(output_path)
