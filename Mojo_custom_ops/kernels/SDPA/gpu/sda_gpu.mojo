from gpu import thread_idx, block_idx, block_dim
from math import exp, sqrt, inf
from tensor import InputTensor, OutputTensor
from std.utils.index import IndexList
from std.runtime.asyncrt import DeviceContextPtr


# ─────────────────────────────────────────────────────────────────────────────
# Kernel: one block per query row i, one thread per d_k column k
#
# Thread (block=i, thread=k) computes output[i, k]:
#   1) Pass 1 — compute all Q[i]·K[j] scores, track running max
#   2) Pass 2 — sum exp(score - max) over j  →  softmax denominator
#   3) Pass 3 — weighted sum of V[j, k]      →  output[i, k]
#
# Deliberately redundant (recomputes dot products 3×) — this is the naive GPU
# direct translation of the CPU naive kernel.
# ─────────────────────────────────────────────────────────────────────────────
fn _sda_gpu_naive_kernel(
    output: OutputTensor[dtype = DType.float16, rank = 2, ...],
    Q: InputTensor[dtype = output.dtype, rank = 2, ...],
    K: InputTensor[dtype = output.dtype, rank = 2, ...],
    V: InputTensor[dtype = output.dtype, rank = 2, ...],
):
    var i = Int(block_idx.x)   # query row  (grid  axis)
    var k = Int(thread_idx.x)  # d_k column (block axis)

    var seq_len = Int(Q.dim_size(0))
    var d_k     = Int(Q.dim_size(1))

    if i >= seq_len or k >= d_k:
        return

    var scale = 1.0 / sqrt(Float32(d_k))

    # ── Pass 1: find max score for numerical stability ────────────────────
    var max_score = Float32(-3.4028235e+38)   # approx -inf for Float32

    for j in range(seq_len):
        var dot = Float32(0.0)
        for kk in range(d_k):
            dot += (
                Float32(Q.load[2](IndexList[2](i,  kk))[0]) *
                Float32(K.load[2](IndexList[2](j,  kk))[0])
            )
        dot *= scale
        if dot > max_score:
            max_score = dot

    # ── Pass 2: compute softmax denominator ───────────────────────────────
    var sum_exp = Float32(0.0)

    for j in range(seq_len):
        var dot = Float32(0.0)
        for kk in range(d_k):
            dot += (
                Float32(Q.load[2](IndexList[2](i,  kk))[0]) *
                Float32(K.load[2](IndexList[2](j,  kk))[0])
            )
        sum_exp += exp(dot * scale - max_score)

    # ── Pass 3: accumulate weighted V values → output[i, k] ──────────────
    var val = Float32(0.0)

    for j in range(seq_len):
        var dot = Float32(0.0)
        for kk in range(d_k):
            dot += (
                Float32(Q.load[2](IndexList[2](i,  kk))[0]) *
                Float32(K.load[2](IndexList[2](j,  kk))[0])
            )
        var weight = exp(dot * scale - max_score) / sum_exp
        val += weight * Float32(V.load[2](IndexList[2](j, k))[0])

    output.store[2](IndexList[2](i, k), Float16(val))


# ─────────────────────────────────────────────────────────────────────────────
# Launcher — called from sda.mojo when target == "gpu"
# ─────────────────────────────────────────────────────────────────────────────
def _sda_gpu_naive(
    output: OutputTensor[dtype = DType.float16, rank = 2, ...],
    Q: InputTensor[dtype = output.dtype, rank = 2, ...],
    K: InputTensor[dtype = output.dtype, rank = 2, ...],
    V: InputTensor[dtype = output.dtype, rank = 2, ...],
    ctx: DeviceContextPtr,
) raises:
    var seq_len = Int(Q.dim_size(0))
    var d_k     = Int(Q.dim_size(1))

    # grid  = (seq_len,)  →  one block  per query row
    # block = (d_k,)      →  one thread per output column
    ctx.enqueue_function[_sda_gpu_naive_kernel](
        output, Q, K, V,
        grid_dim  = (seq_len, 1, 1),
        block_dim = (d_k,     1, 1),
    )