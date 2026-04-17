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


def test_attention(kernel_name, seq_len, d_k, fnc_choice):
    mojo_kernels = Path("./kernels/SDPA")

    dtype = DType.float16
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

    # Random Q, K, V
    Q_np = np.random.rand(seq_len, d_k).astype(np.float16)
    K_np = np.random.rand(seq_len, d_k).astype(np.float16)
    V_np = np.random.rand(seq_len, d_k).astype(np.float16)

    Q = Buffer.from_numpy(Q_np).to(device)
    K = Buffer.from_numpy(K_np).to(device)
    V = Buffer.from_numpy(V_np).to(device)

    # Warmup
    _ = model.execute(Q, K, V)

    st = time.perf_counter()
    result = model.execute(Q, K, V)[0]
    et = time.perf_counter()

    time_taken = (et - st) * 1e3

    result = result.to(CPU()).to_numpy()
    expected = numpy_attention(
        Q_np.astype(np.float32),
        K_np.astype(np.float32),
        V_np.astype(np.float32)
    ).astype(np.float16)
    if not np.allclose(result, expected, atol=1e-3):
        return [False, time_taken]

    return [True, time_taken]


if __name__ == "__main__":
    seq_len = [32,64, 128, 256, 512, 1024]
    d_k = [32, 64, 128, 256, 512, 1024]
    for i in range(len(seq_len)):
        print(f"Testing with size {seq_len[i]} and {d_k[i]}")
        print()
        for j in range(0,3):
            res = test_attention("sda-custom-ops", seq_len[i], d_k[i], j)
            if(j == 0):
                print(f"Naive Kernel : {'Passed' if res[0] else 'Failed'}, Time: {res[1]:.3f} ms")
            elif(j == 1):
                print(f"Loop Reordered Kernel : {'Passed' if res[0] else 'Failed'}, Time: {res[1]:.3f} ms")
            else:
                print(f"Loop Tiling Kernel : {'Passed' if res[0] else 'Failed'}, Time: {res[1]:.3f} ms")
        print()