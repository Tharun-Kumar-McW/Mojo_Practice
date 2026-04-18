from std.math import exp, sqrt, ceildiv
from std.gpu import block_dim, block_idx, thread_idx
from std.runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from std.utils.index import IndexList


def _sda_gpu_naive(
    output: OutputTensor[dtype = DType.float16, rank = 2, ...],
    Q: InputTensor[dtype = output.dtype, rank = 2, ...],
    K: InputTensor[dtype = output.dtype, rank = 2, ...],
    V: InputTensor[dtype = output.dtype, rank = 2, ...],
    ctx: DeviceContextPtr,
) raises:
    comptime BLOCK_SIZE = 64

    var seq_len = output.dim_size(0)
    var d_k     = output.dim_size(1)
    var gpu_ctx = ctx.get_device_context()

    # fn + capturing (not def) — enqueue_function_experimental rejects `raises`
    @parameter
    fn _kernel(seq_len_: Int, d_k_: Int) capturing -> None:
        var i = Int(block_idx.x)
        var k = Int(thread_idx.x)

        if i >= seq_len_ or k >= d_k_:
            return

        var scale = Float32(1.0) / sqrt(Float32(d_k_))

        # ── Pass 1: running max ───────────────────────────────────────────
        var max_score = Float32(-3.4028235e+38)
        for j in range(seq_len_):
            var dot = Float32(0.0)
            for kk in range(d_k_):
                dot += (
                    Float32(Q.load[2](IndexList[2](i, kk))[0]) *
                    Float32(K.load[2](IndexList[2](j, kk))[0])
                )
            dot *= scale
            if dot > max_score:
                max_score = dot

        # ── Pass 2: softmax denominator ───────────────────────────────────
        var sum_exp = Float32(0.0)
        for j in range(seq_len_):
            var dot = Float32(0.0)
            for kk in range(d_k_):
                dot += (
                    Float32(Q.load[2](IndexList[2](i, kk))[0]) *
                    Float32(K.load[2](IndexList[2](j, kk))[0])
                )
            sum_exp += exp(dot * scale - max_score)

        # ── Pass 3: weighted sum → output[i, k] ───────────────────────────
        var val = Float32(0.0)
        for j in range(seq_len_):
            var dot = Float32(0.0)
            for kk in range(d_k_):
                dot += (
                    Float32(Q.load[2](IndexList[2](i, kk))[0]) *
                    Float32(K.load[2](IndexList[2](j, kk))[0])
                )
            var weight = exp(dot * scale - max_score) / sum_exp
            val += weight * Float32(V.load[2](IndexList[2](j, k))[0])

        output.store[2](IndexList[2](i, k), Float16(val))

    gpu_ctx.enqueue_function_experimental[_kernel](
        Int(seq_len), Int(d_k),
        grid_dim  = Int(seq_len),
        block_dim = BLOCK_SIZE,
    )