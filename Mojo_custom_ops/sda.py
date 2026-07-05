from pathlib import Path
import time
import ctypes
import subprocess
import numpy as np

from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def flush_cache():
    """Flush CPU cache by reading a large array that exceeds LLC size."""
    # Typical L3 cache is 8–64 MB; 128 MB is enough to evict everything
    cache_flush_size = 128 * 1024 * 1024  # 128 MB
    flush_array = np.ones(cache_flush_size // 8, dtype=np.float64)
    _ = np.sum(flush_array)  # Force actual memory reads
    del flush_array


def numpy_attention(Q, K, V):
    d_k = Q.shape[1]
    scores = Q @ K.T
    scores = scores / np.sqrt(d_k)
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs @ V


def test_attention(kernel_name, seq_len, d_k, fnc_choice):
    mojo_kernels = Path("./kernels/SDPA")

    dtype = DType.float32
    device = CPU() if accelerator_count() == 0 else Accelerator()

    graph = Graph(
        "attention",
        forward=lambda Q, K, V: ops.custom(
            name=kernel_name,
            device=DeviceRef.from_device(device),
            values=[Q, K, V],
            out_types=[
                TensorType(
                    dtype=dtype,
                    shape=[seq_len, d_k],
                    device=DeviceRef.from_device(device),
                )
            ],
            parameters={"flag": fnc_choice},
        )[0].tensor,
        input_types=[
            TensorType(dtype, shape=[seq_len, d_k], device=DeviceRef.from_device(device)),
            TensorType(dtype, shape=[seq_len, d_k], device=DeviceRef.from_device(device)),
            TensorType(dtype, shape=[seq_len, d_k], device=DeviceRef.from_device(device)),
        ],
        custom_extensions=[mojo_kernels],
    )

    session = InferenceSession(devices=[device])
    model = session.load(graph)

    Q_np = np.random.rand(seq_len, d_k).astype(np.float32)
    K_np = np.random.rand(seq_len, d_k).astype(np.float32)
    V_np = np.random.rand(seq_len, d_k).astype(np.float32)

    Q = Buffer.from_numpy(Q_np).to(device)
    K = Buffer.from_numpy(K_np).to(device)
    V = Buffer.from_numpy(V_np).to(device)

    for _ in range(3):
        _ = model.execute(Q, K, V)

    flush_cache()

    res = 0
    for _ in range(10):
        st = time.perf_counter()
        result = model.execute(Q, K, V)[0]
        et = time.perf_counter()
        res += (et - st) * 1e3

    time_taken = res / 10

    result = result.to(CPU()).to_numpy()
    expected = numpy_attention(
        Q_np.astype(np.float32),
        K_np.astype(np.float32),
        V_np.astype(np.float32),
    ).astype(np.float16)

    if not np.allclose(result, expected, atol=1e-3):
        diff = np.abs(expected - result)
        print("Max abs diff:", diff.max())
        print("Max diff location:", np.unravel_index(diff.argmax(), diff.shape))
        print("Built-in first 5:", expected.flatten()[:5])
        print("Custom   first 5:", result.flatten()[:5])
        print("Any NaN in custom:", np.isnan(result).any())
        print("Any Inf in custom:", np.isinf(result).any())
        print("Rows with NaN:", np.where(np.isnan(result).any(axis=1))[0])
        return [False, time_taken]

    return [True, time_taken]


if __name__ == "__main__":
    seq_lens = [1024]
    d_ks     = [1024]

    kernel_names = [
        "Naive Kernel",
        "Loop Reordered Kernel",
        "Loop Tiling Kernel",
        "Loop Tiling SIMD Kernel",
        "Loop Tiling SIMD Online Softmax Kernel",
    ]


    flush_cache()
    print("Cache cleared. Starting benchmark.\n")

    for seq_len, d_k in zip(seq_lens, d_ks):
        print(f"{'─' * 55}")
        print(f"  seq_len={seq_len}  d_k={d_k}")
        print(f"{'─' * 55}")


        flush_cache()

        for j, name in enumerate(kernel_names):
            flush_cache()

            res = test_attention("sda-custom-ops", seq_len, d_k, j)
            status = "Passed " if res[0] else "Failed "
            print(f"  {name:<28} {status}   {res[1]:.3f} ms")

            time.sleep(2)

        print()