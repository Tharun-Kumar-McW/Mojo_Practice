import compiler

from std.runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from std.utils.index import IndexList

from std.math import sqrt, exp
from std.collections import List


def _sda_cpu_naive(
    output: OutputTensor[dtype=DType.float16, rank=2, ...],
    Q: InputTensor[dtype=output.dtype, rank=2, ...],
    K: InputTensor[dtype=output.dtype, rank=2, ...],
    V: InputTensor[dtype=output.dtype, rank=2, ...],
) raises:
    var seq_len = Q.dim_size(0)
    var d_k = Q.dim_size(1)

    var scale: Float32 = 1.0 / sqrt(Float32(d_k))

    for i in range(seq_len):
        var scores = List[Float32]()   # compute in FP32

        var q_row = List[Float32]()
        for k in range(d_k):
            var qv = Q.load[2](IndexList[2](i, k))
            q_row.append(Float32(qv[0]))

        for j in range(seq_len):
            var dot: Float32 = 0.0

            for k in range(d_k):
                var kv = K.load[2](IndexList[2](j, k))
                dot = dot + (q_row[k] * Float32(kv[0]))

            scores.append(dot * scale)

        var max_val: Float32 = scores[0]
        for j in range(1, seq_len):
            if scores[j] > max_val:
                max_val = scores[j]

        var sum_exp: Float32 = 0.0
        for j in range(seq_len):
            var e = exp(scores[j] - max_val)
            scores[j] = e
            sum_exp = sum_exp + e

        for j in range(seq_len):
            scores[j] = scores[j] / sum_exp

        for k in range(d_k):
            var val: Float32 = 0.0

            for j in range(seq_len):
                var vv = V.load[2](IndexList[2](j, k))
                val = val + (scores[j] * Float32(vv[0]))

            output.store[2](IndexList[2](i, k), Float16(val))


def _sda_cpu_loop_reordered(
    output: OutputTensor[dtype=DType.float16, rank=2, ...],
    Q: InputTensor[dtype=output.dtype, rank=2, ...],
    K: InputTensor[dtype=output.dtype, rank=2, ...],
    V: InputTensor[dtype=output.dtype, rank=2, ...],
) raises:
    var seq_len = Q.dim_size(0)
    var d_k = Q.dim_size(1)

    var scale: Float32 = 1.0 / sqrt(Float32(d_k))

    for i in range(seq_len):
        var scores = List[Float32]()

        var q_row = List[Float32]()
        for k in range(d_k):
            var qv = Q.load[2](IndexList[2](i, k))
            q_row.append(Float32(qv[0]))
        for j in range(seq_len):
            var dot: Float32 = 0.0

            for k in range(d_k):
                var kv = K.load[2](IndexList[2](j, k))
                dot = dot + (q_row[k] * Float32(kv[0]))

            scores.append(dot * scale)

        var max_val: Float32 = scores[0]
        for j in range(1, seq_len):
            if scores[j] > max_val:
                max_val = scores[j]

        var sum_exp: Float32 = 0.0
        for j in range(seq_len):
            var e = exp(scores[j] - max_val)
            scores[j] = e
            sum_exp = sum_exp + e

        for j in range(seq_len):
            scores[j] = scores[j] / sum_exp

        var out_row = List[Float32]()
        for _ in range(d_k):
            out_row.append(0.0)

        for j in range(seq_len):
            var score = scores[j]
            for k in range(d_k):
                var vv = V.load[2](IndexList[2](j, k))
                out_row[k] = out_row[k] + (score * Float32(vv[0]))

        for k in range(d_k):
            output.store[2](IndexList[2](i, k), Float16(out_row[k]))

def _sda_cpu_loop_tiling(
    output: OutputTensor[dtype=DType.float16, rank=2, ...],
    Q: InputTensor[dtype=output.dtype, rank=2, ...],
    K: InputTensor[dtype=output.dtype, rank=2, ...],
    V: InputTensor[dtype=output.dtype, rank=2, ...],
) raises:
    var seq_len = Q.dim_size(0)
    var d_k = Q.dim_size(1)

    var scale: Float32 = 1.0 / sqrt(Float32(d_k))
    var J_TILE = 16
    var K_TILE = 16

    for i in range(seq_len):
        var scores = List[Float32]()
        scores.resize(seq_len, 0.0)

        var q_row = List[Float32]()
        for k in range(d_k):
            var qv = Q.load[2](IndexList[2](i, k))
            q_row.append(Float32(qv[0]))
        
        for jj in range(0, seq_len, J_TILE):
            for kk in range(0, d_k, K_TILE):
                for j in range(jj,min(jj+J_TILE, seq_len)):
                    var dot: Float32 = 0.0

                    for k in range(kk, min(kk+K_TILE, d_k)):
                        var kv = K.load[2](IndexList[2](j, k))
                        dot = dot + (q_row[k] * Float32(kv[0]))

                    # scores.append(dot * scale)
                    scores[j] = scores[j] + (dot * scale)

        var max_val: Float32 = scores[0]
        for j in range(1, seq_len):
            if scores[j] > max_val:
                max_val = scores[j]

        var sum_exp: Float32 = 0.0
        for j in range(seq_len):
            var e = exp(scores[j] - max_val)
            scores[j] = e
            sum_exp = sum_exp + e

        for j in range(seq_len):
            scores[j] = scores[j] / sum_exp

        var out_row = List[Float32]()
        for _ in range(d_k):
            out_row.append(0.0)


        for jj in range(0, seq_len, J_TILE):
            for kk in range(0, d_k, K_TILE):
                for j in range(jj,min(jj+J_TILE, seq_len)):
                    var score = scores[j]
                    for k in range(kk, min(kk+K_TILE, d_k)):
                        var vv = V.load[2](IndexList[2](j, k))
                        out_row[k] = out_row[k] + (score * Float32(vv[0]))

        for k in range(d_k):
            output.store[2](IndexList[2](i, k), Float16(out_row[k]))