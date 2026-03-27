from pathlib import Path

import numpy as np
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

if __name__ == "__main__":
    mojo_kernels = Path("./kernels")
    # mxk kxn => mxn
    m = 4
    n = 3
    k = 3
    dtype = DType.float32
    device = CPU() if accelerator_count() == 0 else Accelerator()
    print(f"Using device: {device}")
    # Configure our simple one-operation graph.
    graph = Graph(
        "matmultiplication",
        forward=lambda a, b: ops.custom(
            name="matmultiply",
            device=DeviceRef.from_device(device),
            values=[a, b],
            out_types=[
                TensorType(
                    dtype=a.dtype,
                    shape=[m, n],
                    device=DeviceRef.from_device(device),
                )
            ],
        )[0].tensor,
        input_types=[
            TensorType(
                dtype,
                shape=[m, k],
                device=DeviceRef.from_device(device),
            ),
            TensorType(
                dtype,
                shape=[k, n],
                device=DeviceRef.from_device(device),
            ),
        ],
        custom_extensions=[mojo_kernels],
    )

    session = InferenceSession(
        devices=[device],
    )

    model = session.load(graph)

    # Fill an input matrix with random values.
    a_mat = np.random.uniform(low=1.0, high=2.0, size=(m, k)).astype(np.float32)
    b_mat = np.random.uniform(low=1.0, high=2.0, size=(k, n)).astype(np.float32)

    # Create tensors and move them to the device (CPU or GPU).
    a = Buffer.from_numpy(a_mat).to(device)
    b = Buffer.from_numpy(b_mat).to(device)

    # Run inference with the input tensor.
    result = model.execute(a, b)[0]

    # Copy values back to the CPU to be read.
    assert isinstance(result, Buffer)
    result = result.to(CPU())

    print("--- Original Input ---\nA matrix:")
    print(a_mat)
    print("\nB matrix:")
    print(b_mat)
    print("\n--- Graph result (Mojo MatMul) ---")
    print(result.to_numpy())
    print("\n--- Expected result (NumPy MatMul) ---")
    print(a_mat @ b_mat)