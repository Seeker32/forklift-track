"""TensorRT based forklift detector for RF-DETR deploy-format engines."""

from __future__ import annotations

from typing import Any

import numpy as np


class TensorRTForkliftDetector:
    """RF-DETR forklift detector backed by a serialized TensorRT engine."""

    MEANS = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STDS = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    EXPECTED_INPUT_SHAPE = (1, 3, 512, 512)
    REQUIRED_OUTPUTS = ("scores", "labels", "boxes")

    def __init__(
        self,
        engine_path: str,
        confidence: float = 0.4,
        class_names: list[str] | None = None,
        allowed_class_names: list[str] | None = None,
        runtime_loader: Any | None = None,
    ) -> None:
        self.confidence = confidence
        self.class_names = class_names or ["forklift_with_load", "forklift_empty"]
        self.allowed_class_names = allowed_class_names

        try:
            runtime_factory = runtime_loader or _TensorRTRuntime
            self._runtime = runtime_factory(engine_path)
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT runtime is unavailable. Install TensorRT and CUDA Python runtime before loading .engine models."
            ) from exc

        input_shape = tuple(int(dim) for dim in self._runtime.get_input_shape())
        if input_shape != self.EXPECTED_INPUT_SHAPE:
            raise ValueError(
                "TensorRT detector only supports RF-DETR engines with input shape 1x3x512x512; "
                f"got {input_shape!r}."
            )

        output_names = tuple(self._runtime.get_output_names())
        if any(name not in output_names for name in self.REQUIRED_OUTPUTS):
            raise ValueError(
                "TensorRT detector only supports deploy-format RF-DETR engines with outputs scores, labels, boxes."
            )

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        h, w = frame.shape[:2]
        outputs = self._runtime.infer({"input": self._preprocess(frame)})
        detections = self._postprocess_deploy(
            scores=np.asarray(outputs["scores"]),
            labels=np.asarray(outputs["labels"]),
            boxes=np.asarray(outputs["boxes"]),
            orig_h=h,
            orig_w=w,
        )
        if self.allowed_class_names is not None:
            detections = [d for d in detections if d.get("class_name") in self.allowed_class_names]
        return detections

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        image = resized.astype(np.float32) / 255.0
        image = (image - self.MEANS) / self.STDS
        return np.transpose(image, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

    def _postprocess_deploy(
        self, *, scores: np.ndarray, labels: np.ndarray, boxes: np.ndarray, orig_h: int, orig_w: int
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


class _TensorRTRuntime:
    def __init__(self, engine_path: str) -> None:
        import tensorrt as trt
        from cuda import cudart

        self._trt = trt
        self._cudart = cudart
        self._logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as engine_file:
            engine_bytes = engine_file.read()

        runtime = trt.Runtime(self._logger)
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        self._runtime = runtime
        self._engine = engine
        self._context = engine.create_execution_context()
        if self._context is None:
            raise RuntimeError(f"Failed to create TensorRT execution context: {engine_path}")

        self._input_names, self._output_names = self._discover_io_names()
        self._input_buffers = {
            name: self._allocate_buffer(self._tensor_shape(name), np.float32) for name in self._input_names
        }
        self._output_buffers = {
            name: self._allocate_buffer(self._tensor_shape(name), self._output_dtype(name)) for name in self._output_names
        }

    def get_input_shape(self) -> tuple[int, ...]:
        if len(self._input_names) != 1:
            raise ValueError(f"Expected exactly one TensorRT input, got {self._input_names!r}")
        return tuple(int(dim) for dim in self._tensor_shape(self._input_names[0]))

    def get_output_names(self) -> tuple[str, ...]:
        return self._output_names

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        for name, array in inputs.items():
            if name not in self._input_buffers:
                raise KeyError(f"Unexpected TensorRT input: {name}")
            host_array = np.ascontiguousarray(array.astype(self._input_buffers[name]["host"].dtype, copy=False))
            if host_array.shape != self._input_buffers[name]["host"].shape:
                raise ValueError(
                    f"TensorRT input {name} expected shape {self._input_buffers[name]['host'].shape!r}, "
                    f"got {host_array.shape!r}"
                )
            np.copyto(self._input_buffers[name]["host"], host_array)
            self._memcpy_host_to_device(self._input_buffers[name]["device"], self._input_buffers[name]["host"])

        self._execute()

        outputs: dict[str, np.ndarray] = {}
        for name, buffer in self._output_buffers.items():
            self._memcpy_device_to_host(buffer["host"], buffer["device"])
            outputs[name] = buffer["host"].copy()
        return outputs

    def _discover_io_names(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        trt = self._trt
        if hasattr(self._engine, "num_io_tensors"):
            input_names = []
            output_names = []
            for index in range(self._engine.num_io_tensors):
                name = self._engine.get_tensor_name(index)
                mode = self._engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    input_names.append(name)
                else:
                    output_names.append(name)
            return tuple(input_names), tuple(output_names)

        input_names = []
        output_names = []
        for index in range(self._engine.num_bindings):
            name = self._engine.get_binding_name(index)
            if self._engine.binding_is_input(index):
                input_names.append(name)
            else:
                output_names.append(name)
        return tuple(input_names), tuple(output_names)

    def _tensor_shape(self, name: str) -> tuple[int, ...]:
        if hasattr(self._engine, "get_tensor_shape"):
            shape = tuple(int(dim) for dim in self._engine.get_tensor_shape(name))
        else:
            index = self._engine.get_binding_index(name)
            shape = tuple(int(dim) for dim in self._engine.get_binding_shape(index))
        if any(dim < 0 for dim in shape):
            raise ValueError(f"Dynamic TensorRT shapes are not supported for tensor {name}: {shape!r}")
        return shape

    def _output_dtype(self, name: str) -> np.dtype[Any]:
        trt = self._trt
        if hasattr(self._engine, "get_tensor_dtype"):
            return np.dtype(trt.nptype(self._engine.get_tensor_dtype(name)))
        index = self._engine.get_binding_index(name)
        return np.dtype(trt.nptype(self._engine.get_binding_dtype(index)))

    def _allocate_buffer(self, shape: tuple[int, ...], dtype: np.dtype[Any]) -> dict[str, Any]:
        host = np.empty(shape, dtype=dtype)
        size_in_bytes = int(host.nbytes)
        status, device_ptr = self._cudart.cudaMalloc(size_in_bytes)
        self._check_cuda_status(status, "cudaMalloc")
        return {"host": host, "device": device_ptr}

    def _memcpy_host_to_device(self, device_ptr: int, host: np.ndarray) -> None:
        status = self._cudart.cudaMemcpy(
            device_ptr,
            host.ctypes.data,
            host.nbytes,
            self._cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )[0]
        self._check_cuda_status(status, "cudaMemcpyHostToDevice")

    def _memcpy_device_to_host(self, host: np.ndarray, device_ptr: int) -> None:
        status = self._cudart.cudaMemcpy(
            host.ctypes.data,
            device_ptr,
            host.nbytes,
            self._cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )[0]
        self._check_cuda_status(status, "cudaMemcpyDeviceToHost")

    def _execute(self) -> None:
        if hasattr(self._context, "set_tensor_address"):
            for name, buffer in self._input_buffers.items():
                self._context.set_tensor_address(name, buffer["device"])
            for name, buffer in self._output_buffers.items():
                self._context.set_tensor_address(name, buffer["device"])
            if not self._context.execute_async_v3(0):
                raise RuntimeError("TensorRT execute_async_v3 returned failure")
            return

        bindings: list[int] = [0] * self._engine.num_bindings
        for name, buffer in self._input_buffers.items():
            bindings[self._engine.get_binding_index(name)] = int(buffer["device"])
        for name, buffer in self._output_buffers.items():
            bindings[self._engine.get_binding_index(name)] = int(buffer["device"])
        if not self._context.execute_v2(bindings):
            raise RuntimeError("TensorRT execute_v2 returned failure")

    def _check_cuda_status(self, status: Any, operation: str) -> None:
        cuda_error_enum = getattr(self._cudart, "cudaError_t", None)
        if cuda_error_enum is not None and status == cuda_error_enum.cudaSuccess:
            return
        if status == 0:
            return
        raise RuntimeError(f"{operation} failed with CUDA status {status}")
