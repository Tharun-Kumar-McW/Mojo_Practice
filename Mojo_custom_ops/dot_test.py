from pathlib import Path
import time

import numpy as np
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

if __name__ == "__main__":
    def test_graph(kernel_name, n):
        mojo_kernels = Path("./kernels")
        dtype = DType.int32
        device = CPU() if accelerator_count() == 0 else Accelerator()
        input_size = n

        graph = Graph(
            "dot_product",
            forward=lambda a, b: ops.custom(
                name=kernel_name,
                device=DeviceRef.from_device(device),
                values=[a, b],
                out_types=[
                    TensorType(
                        dtype= dtype,
                        shape=[],
                        device=DeviceRef.from_device(device),
                    )
                ],
            )[0].tensor,
            input_types=[
                TensorType(
                    dtype,
                    shape=[input_size],
                    device=DeviceRef.from_device(device),
                ),
                TensorType(
                    dtype,
                    shape=[input_size],
                    device=DeviceRef.from_device(device),
                ),
            ],
            custom_extensions=[mojo_kernels],
        )

        session = InferenceSession(
            devices=[device],
        )

        model = session.load(graph)

        a_values = np.random.uniform(low=1, high=5, size=(input_size)).astype(np.int32)
        b_values = np.random.uniform(low=1, high=5, size=(input_size)).astype(np.int32)
        a = Buffer.from_numpy(a_values).to(device)
        b = Buffer.from_numpy(b_values).to(device) 
        result = model.execute(a, b)[0]
        st = time.time()
        result = model.execute(a, b)[0]
        et = time.time()
        time_taken = (et - st) * 1e3
        assert isinstance(result, Buffer)
        result = result.to(CPU())
        result = result.to_numpy()
        actual = np.dot(a_values, b_values)
        if(result != actual):
            return [False, time_taken]
        return [True, time_taken]
    n_size = 10**8
    res_1 = test_graph("dot-serial", n_size)
    res_2 = test_graph("dot-parallel", n_size)
    res_3 = test_graph("dot-parallel-simd", n_size)
    print(f"Serial Dot Product: {'Passed' if res_1[0] else 'Failed'}, Time taken: {res_1[1]:.3f} ms")
    print(f"Parallel Dot Product: {'Passed' if res_2[0] else 'Failed'}, Time taken: {res_2[1]:.3f} ms")
    print(f"Parallel SIMD Dot Product: {'Passed' if res_3[0] else 'Failed'}, Time taken: {res_3[1]:.3f} ms")