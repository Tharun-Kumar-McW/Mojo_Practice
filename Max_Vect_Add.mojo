# vec_add.mojo
from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor
from std.math import ceildiv
from std.sys import has_accelerator

comptime float_dtype  = DType.float32
comptime vector_size  = 8
comptime layout       = Layout.row_major(vector_size)
comptime block_size   = 4
comptime num_blocks   = ceildiv(vector_size, block_size)

# ── Kernel ────────────────────────────────────────────────────────────────────
fn vector_addition(
    lhs:    LayoutTensor[float_dtype, layout, MutAnyOrigin],
    rhs:    LayoutTensor[float_dtype, layout, MutAnyOrigin],
    result: LayoutTensor[float_dtype, layout, MutAnyOrigin],
):
    var i = block_dim.x * block_idx.x + thread_idx.x
    if i < vector_size:
        result[i] = lhs[i] + rhs[i]

# ── Host ──────────────────────────────────────────────────────────────────────
def main() raises:
    if not has_accelerator():
        print("No GPU found")
        return

    with DeviceContext() as ctx:

        # 1. Allocate CPU (host) buffers and fill with data
        var lhs_host = ctx.enqueue_create_host_buffer[float_dtype](vector_size)
        var rhs_host = ctx.enqueue_create_host_buffer[float_dtype](vector_size)
        ctx.synchronize()

        for i in range(vector_size):
            lhs_host[i] = Float32(i + 1)     # [1.0, 2.0 ... 8.0]
            rhs_host[i] = Float32(10.0)      # [10.0, 10.0 ... 10.0]

        print("LHS:", lhs_host)
        print("RHS:", rhs_host)

        # 2. Allocate GPU (device) buffers
        var lhs_dev    = ctx.enqueue_create_buffer[float_dtype](vector_size)
        var rhs_dev    = ctx.enqueue_create_buffer[float_dtype](vector_size)
        var result_dev = ctx.enqueue_create_buffer[float_dtype](vector_size)

        # 3. Copy CPU → GPU
        ctx.enqueue_copy(lhs_dev, lhs_host)
        ctx.enqueue_copy(rhs_dev, rhs_host)

        # 4. Wrap device buffers as LayoutTensors
        var lhs_tensor    = LayoutTensor[float_dtype, layout](lhs_dev)
        var rhs_tensor    = LayoutTensor[float_dtype, layout](rhs_dev)
        var result_tensor = LayoutTensor[float_dtype, layout](result_dev)

        # 5. Dispatch kernel — kernel passed TWICE (compile-time + type-check)
        ctx.enqueue_function[vector_addition, vector_addition](
            lhs_tensor,
            rhs_tensor,
            result_tensor,
            grid_dim  = num_blocks,
            block_dim = block_size,
        )

        # 6. Copy result GPU → CPU and print
        var result_host = ctx.enqueue_create_host_buffer[float_dtype](vector_size)
        ctx.enqueue_copy(result_host, result_dev)
        ctx.synchronize()

        print("Result:", result_host)