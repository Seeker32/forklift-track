"""ONNX Runtime based forklift detector for RF-DETR models."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np


class ONNXForkliftDetector:
    """RF-DETR forklift detector backed by ONNX Runtime.

    Input resolution: 512x512.
    Output classes: forklift_with_load, forklift_empty.
    """

    MEANS = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STDS = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    DEFAULT_TRT_ENGINE_CACHE_PATH = str(Path(__file__).resolve().parents[1] / "models" / "trt_engine_cache")

    def __init__(
        self,
        onnx_path: str,
        confidence: float = 0.4,
        class_names: list[str] | None = None,
        allowed_class_names: list[str] | None = None,
        num_select: int = 300,
        providers: list[str | tuple[str, dict[str, str]]] | None = None,
        trt_engine_cache_path: str | None = None,
        trt_fp16: bool = False,
        trt_max_workspace_size: int | None = None,
        trt_builder_optimization_level: int | None = None,
        ort_profile: bool = False,
        ort_profile_prefix: str | None = None,
        ort_verbose: bool = False,
        runtime_module: Any | None = None,
        runtime_loader: Any | None = None,
        clock: Any = time.perf_counter,
    ) -> None:
        self.confidence = confidence
        self.class_names = class_names or ["forklift_with_load", "forklift_empty"]
        self.allowed_class_names = allowed_class_names
        self.num_select = num_select
        self._clock = clock
        self._stage_timing_sink: Any | None = None
        ort = runtime_module
        if ort is None:
            try:
                ort = (runtime_loader or _load_onnxruntime)()
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "ONNX Runtime is unavailable. Install onnxruntime or onnxruntime-gpu to load .onnx models."
                ) from exc
        sess_options = self._session_options(
            ort,
            ort_profile=ort_profile,
            ort_profile_prefix=ort_profile_prefix,
            ort_verbose=ort_verbose,
        )
        resolved_providers = self._resolved_providers(
            providers=providers,
            trt_engine_cache_path=trt_engine_cache_path,
            trt_fp16=trt_fp16,
            trt_max_workspace_size=trt_max_workspace_size,
            trt_builder_optimization_level=trt_builder_optimization_level,
        )
        self._session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=resolved_providers,
        )
        self._output_names = tuple(output.name for output in self._session.get_outputs())
        output_name_set = set(self._output_names)
        self._is_raw_output_format = {"pred_boxes", "pred_logits"}.issubset(output_name_set)
        self._is_deploy_format = {"scores", "labels", "boxes"}.issubset(output_name_set)

    @staticmethod
    def _default_providers(
        *,
        trt_engine_cache_path: str | None,
        trt_fp16: bool,
        trt_max_workspace_size: int | None = None,
        trt_builder_optimization_level: int | None = None,
    ) -> list[str | tuple[str, dict[str, str]]]:
        tensorrt_options = {
            "trt_engine_cache_enable": "True",
            "trt_engine_cache_path": trt_engine_cache_path or ONNXForkliftDetector.DEFAULT_TRT_ENGINE_CACHE_PATH,
        }
        if trt_fp16:
            tensorrt_options["trt_fp16_enable"] = "True"
        if trt_max_workspace_size is not None:
            tensorrt_options["trt_max_workspace_size"] = str(int(trt_max_workspace_size))
        if trt_builder_optimization_level is not None:
            tensorrt_options["trt_builder_optimization_level"] = str(int(trt_builder_optimization_level))
        return [
            ("TensorrtExecutionProvider", tensorrt_options),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    @staticmethod
    def _resolved_providers(
        *,
        providers: list[str | tuple[str, dict[str, str]]] | None,
        trt_engine_cache_path: str | None,
        trt_fp16: bool,
        trt_max_workspace_size: int | None = None,
        trt_builder_optimization_level: int | None = None,
    ) -> list[str | tuple[str, dict[str, str]]]:
        default_providers = ONNXForkliftDetector._default_providers(
            trt_engine_cache_path=trt_engine_cache_path,
            trt_fp16=trt_fp16,
            trt_max_workspace_size=trt_max_workspace_size,
            trt_builder_optimization_level=trt_builder_optimization_level,
        )
        if providers is None:
            return default_providers

        tensorrt_name, tensorrt_options = default_providers[0]
        resolved: list[str | tuple[str, dict[str, str]]] = []
        for provider in providers:
            if provider == tensorrt_name:
                resolved.append((tensorrt_name, dict(tensorrt_options)))
                continue
            if isinstance(provider, tuple) and provider[0] == tensorrt_name:
                resolved.append((tensorrt_name, {**tensorrt_options, **provider[1]}))
                continue
            resolved.append(provider)
        return resolved

    @staticmethod
    def _session_options(
        ort: Any,
        *,
        ort_profile: bool,
        ort_profile_prefix: str | None,
        ort_verbose: bool,
    ) -> Any | None:
        session_options_factory = getattr(ort, "SessionOptions", None)
        if session_options_factory is None:
            return None
        sess_options = session_options_factory()
        if ort_profile:
            sess_options.enable_profiling = True
            if ort_profile_prefix:
                sess_options.profile_file_prefix = ort_profile_prefix
        if ort_verbose:
            sess_options.log_severity_level = 0
            sess_options.log_verbosity_level = max(1, int(getattr(sess_options, "log_verbosity_level", 0)))
        return sess_options

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        h, w = frame.shape[:2]

        preprocess_start = self._clock()
        input_tensor = self._preprocess(frame)
        preprocess_end = self._clock()

        inference_start = preprocess_end
        outputs = self._session.run(None, {"input": input_tensor})
        inference_end = self._clock()
        output_map = {name: value for name, value in zip(self._output_names, outputs)}

        postprocess_start = inference_end
        if self._is_raw_output_format:
            detections = self._postprocess(
                dets=np.asarray(self._require_output(output_map, "pred_boxes")),
                labels=np.asarray(self._require_output(output_map, "pred_logits")),
                orig_h=h,
                orig_w=w,
            )
        elif self._is_deploy_format:
            detections = self._postprocess_deploy(
                scores=np.asarray(self._require_output(output_map, "scores")),
                labels=np.asarray(self._require_output(output_map, "labels")),
                boxes=np.asarray(self._require_output(output_map, "boxes")),
                orig_h=h,
                orig_w=w,
            )
        else:
            detections = self._postprocess(np.asarray(outputs[0]), np.asarray(outputs[1]), h, w)

        if self.allowed_class_names is not None:
            detections = [d for d in detections if d.get("class_name") in self.allowed_class_names]
        postprocess_end = self._clock()
        self._record_stage_timings(
            preprocess_ms=(preprocess_end - preprocess_start) * 1000.0,
            inference_ms=(inference_end - inference_start) * 1000.0,
            postprocess_ms=(postprocess_end - postprocess_start) * 1000.0,
        )
        return detections

    def set_stage_timing_sink(self, sink: Any | None) -> None:
        self._stage_timing_sink = sink

    def end_profiling(self) -> str | None:
        end_profiling = getattr(self._session, "end_profiling", None)
        if not callable(end_profiling):
            return None
        return end_profiling()

    def _record_stage_timings(self, *, preprocess_ms: float, inference_ms: float, postprocess_ms: float) -> None:
        if self._stage_timing_sink is None:
            return
        self._stage_timing_sink(
            {
                "preprocess": preprocess_ms,
                "inference": inference_ms,
                "postprocess": postprocess_ms,
            }
        )

    @staticmethod
    def _require_output(output_map: dict[str, Any], name: str) -> Any:
        value = output_map.get(name)
        if value is None:
            raise RuntimeError(f"ONNX Runtime returned no data for required output: {name}")
        return value

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        image = resized.astype(np.float32) / 255.0
        image = (image - self.MEANS) / self.STDS
        return np.transpose(image, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

    def _postprocess(
        self, dets: np.ndarray, labels: np.ndarray, orig_h: int, orig_w: int
    ) -> list[dict[str, Any]]:
        # dets: [1, num_queries, 4] — cxcywh normalized to [0,1]
        # labels: [1, num_queries, num_classes] — raw logits
        boxes = dets[0]
        class_logits = labels[0]
        num_classes = len(self.class_names)

        if class_logits.shape[1] < num_classes:
            raise RuntimeError(
                f"Model returned {class_logits.shape[1]} class logits, expected at least {num_classes}."
            )

        # RF-DETR library export returns raw logits; keep only the application classes.
        probs = 1.0 / (1.0 + np.exp(-class_logits[:, :num_classes]))

        flat_probs = probs.reshape(-1)
        num_select = min(self.num_select, flat_probs.shape[0])
        topk_indices = np.argpartition(-flat_probs, num_select - 1)[:num_select]
        topk_scores = flat_probs[topk_indices]
        order = np.argsort(-topk_scores)
        topk_indices = topk_indices[order]
        topk_scores = topk_scores[order]

        box_indices = topk_indices // num_classes
        class_ids = topk_indices % num_classes

        selected_boxes = boxes[box_indices]
        cx, cy, bw, bh = selected_boxes[:, 0], selected_boxes[:, 1], selected_boxes[:, 2], selected_boxes[:, 3]
        x1 = np.clip((cx - bw / 2.0) * orig_w, 0, orig_w - 1)
        y1 = np.clip((cy - bh / 2.0) * orig_h, 0, orig_h - 1)
        x2 = np.clip((cx + bw / 2.0) * orig_w, 0, orig_w - 1)
        y2 = np.clip((cy + bh / 2.0) * orig_h, 0, orig_h - 1)

        detections: list[dict[str, Any]] = []
        for i in range(len(topk_scores)):
            score = topk_scores[i]
            if score < self.confidence:
                continue
            cls_id = int(class_ids[i])
            detections.append(
                {
                    "bbox": [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
                    "score": float(score),
                    "class_name": self.class_names[cls_id],
                }
            )

        return detections

    def _postprocess_deploy(
        self, scores: np.ndarray, labels: np.ndarray, boxes: np.ndarray,
        orig_h: int, orig_w: int,
    ) -> list[dict[str, Any]]:
        batch_scores = scores[0]
        batch_labels = labels[0]
        batch_boxes = boxes[0]

        detections: list[dict[str, Any]] = []
        for i in range(len(batch_scores)):
            score = float(batch_scores[i])
            if score < self.confidence:
                continue
            cls = int(batch_labels[i])
            if cls < 0 or cls >= len(self.class_names):
                continue
            box = batch_boxes[i]
            detections.append(
                {
                    "bbox": [
                        float(np.clip(box[0], 0, orig_w - 1)),
                        float(np.clip(box[1], 0, orig_h - 1)),
                        float(np.clip(box[2], 0, orig_w - 1)),
                        float(np.clip(box[3], 0, orig_h - 1)),
                    ],
                    "score": score,
                    "class_name": self.class_names[cls],
                }
            )

        return detections


def _load_onnxruntime() -> Any:
    import onnxruntime as ort

    return ort
