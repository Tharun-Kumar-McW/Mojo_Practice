from pathlib import Path
import time
import numpy as np

from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def numpy_attention(Q, K, V):
    d_k = Q.shape[1]
    scores = Q @ K.T
    scores = scores / np.sqrt(d_k)

    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return probs @ V


def test_attention(kernel_name, seq_len, d_k):
    mojo_kernels = Path("./kernels")

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

    # Random Q, K, V
    Q_np = np.random.rand(seq_len, d_k).astype(np.float32)
    K_np = np.random.rand(seq_len, d_k).astype(np.float32)
    V_np = np.random.rand(seq_len, d_k).astype(np.float32)

    Q = Buffer.from_numpy(Q_np).to(device)
    K = Buffer.from_numpy(K_np).to(device)
    V = Buffer.from_numpy(V_np).to(device)

    # Warmup
    _ = model.execute(Q, K, V)

    st = time.time()
    result = model.execute(Q, K, V)[0]
    et = time.time()

    time_taken = (et - st) * 1e3

    result = result.to(CPU()).to_numpy()
    expected = numpy_attention(Q_np, K_np, V_np)
    if not np.allclose(result, expected, atol=1e-4):
        return [False, time_taken]

    return [True, time_taken]


if __name__ == "__main__":
    seq_len = 128
    d_k = 128

    res_1 = test_attention("attention-serial", seq_len, d_k)
    res_2 = test_attention("attention-Loop-reordered", seq_len, d_k)
    print(f"Serial Attention: {'Passed' if res_1[0] else 'Failed'}, Time: {res_1[1]:.3f} ms")
    print(f"Loop-Reordered Attention: {'Passed' if res_2[0] else 'Failed'}, Time: {res_2[1]:.3f} ms")
    print(f"Speedup: {res_1[1] / res_2[1]:.2f}x")