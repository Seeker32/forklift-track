import time
import numpy as np
import onnxruntime as ort


MODEL_PATH = "../models/model.onnx"
RESOLUTION = 512
BATCH = 1
WARMUP = 50
REPEAT = 300


def main():
    providers = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=sess_options,
        providers=providers,
    )

    print("Providers:", session.get_providers())
    print()

    print("Inputs:")
    for inp in session.get_inputs():
        print(" ", inp.name, inp.shape, inp.type)

    print()
    print("Outputs:")
    for out in session.get_outputs():
        print(" ", out.name, out.shape, out.type)

    x = np.random.randn(BATCH, 3, RESOLUTION, RESOLUTION).astype(np.float32)
    target_sizes = np.array(
        [[RESOLUTION, RESOLUTION]] * BATCH,
        dtype=np.int64,
    )

    feed = {
        "input": x,
        "target_sizes": target_sizes,
    }

    # warmup
    for _ in range(WARMUP):
        session.run(None, feed)

    # benchmark
    times = []

    for _ in range(REPEAT):
        t0 = time.perf_counter()
        session.run(None, feed)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times = np.array(times)

    print()
    print("===== ONNX Runtime Benchmark =====")
    print(f"resolution: {RESOLUTION}x{RESOLUTION}")
    print(f"batch: {BATCH}")
    print(f"mean latency: {times.mean():.2f} ms")
    print(f"p50 latency:  {np.percentile(times, 50):.2f} ms")
    print(f"p95 latency:  {np.percentile(times, 95):.2f} ms")
    print(f"p99 latency:  {np.percentile(times, 99):.2f} ms")
    print(f"fps:          {1000 / times.mean():.2f}")


if __name__ == "__main__":
    main()