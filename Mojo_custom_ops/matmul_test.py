from pathlib import Path
import time
import numpy as np
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

if __name__ == "__main__":
    mojo_kernels = Path("./kernels")
    # mxk kxn => mxn
    m = 64
    n = 64
    k = 64
    dtype = DType.int32
    device = CPU() if accelerator_count() == 0 else Accelerator()
    print(f"Using device: {device}")
    # Configure our simple one-operation graph.

    def func(kernel_name):
        print(f"Testing kernel: {kernel_name}")
        graph = Graph(
            "matmultiplication",
            forward=lambda a, b: ops.custom(
                name=kernel_name,
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
        a_mat = np.random.uniform(low=1, high=5, size=(m, k)).astype(np.int32)
        b_mat = np.random.uniform(low=1, high=5, size=(k, n)).astype(np.int32)

        a = Buffer.from_numpy(a_mat).to(device)
        b = Buffer.from_numpy(b_mat).to(device)

        # Run inference with the input tensor.
        result = model.execute(a, b)[0]

        start = time.time()
        result = model.execute(a, b)[0]
        end = time.time()
        print(f"Inference time: {end - start:.6f} seconds for kernel {kernel_name}")

        assert isinstance(result, Buffer)
        result = result.to(CPU())
        # return result.to_numpy()

        print("--- Original Input ---\nA matrix:")
        print(a_mat)
        print("\nB matrix:")
        print(b_mat)
        print(f"\n--- Graph result (Mojo MatMul) {kernel_name} ---")
        print(result.to_numpy())
        print("\n--- Expected result (NumPy MatMul) ---")
        print(a_mat @ b_mat)
    
    func("matmultiply-native")
    func("matmultiply-vectorized")
    # print("\n--- Actual NumPy matmul result for reference ---")
    # print(a_mat @ b_mat)
    # print("\n--- Verifying results between kernels ---")
    # print("Result from matmultiply-native:")
    # print(res1)
    # print("\nResult from matmultiply-vectorized:")
    # print(res2)
    # np.testing.assert_allclose(res1, res2, rtol=1e-5, err_msg="Kernels produced different math!")
    # print("Math verified! Both kernels match.")