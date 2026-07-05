import compiler

from std.runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from std.utils.index import IndexList
# ADD these imports at the very top of sda_cpu.mojo
from algorithm import sync_parallelize
from sys import num_physical_cores

# from std.sys.simd import SIMD
from std.sys.info import simd_width_of

from std.math import sqrt, exp
from std.collections import List
from std.algorithm import parallelize


def _sda_cpu_naive(
    output: OutputTensor[dtype=DType.float32, rank=2, ...],
    Q: InputTensor[dtype=output.dtype, rank=2, ...],
    K: InputTensor[dtype=output.dtype, rank=2, ...],
    V: InputTensor[dtype=output.dtype, rank=2, ...],
) raises:
    var seq_len = Q.dim_size(0)
    var d_k = Q.dim_size(1)

    var scale: Float32 = 1.0 / sqrt(Float32(d_k))

    for i in range(seq_len):
        var scores = List[Float32]()
        for j in range(seq_len):
            var dot: Float32 = 0.0
            for k in range(d_k):
                var qv = Q.load[2](IndexList[2](i, k))
                var kv = K.load[2](IndexList[2](j, k))
                dot = dot + (Float32(qv[0]) * Float32(kv[0]))
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
            output.store[2](IndexList[2](i, k), Float32(val))


def _sda_cpu_loop_reordered(
    output: OutputTensor[dtype=DType.float32, rank=2, ...],
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
            output.store[2](IndexList[2](i, k), Float32(out_row[k]))

def _sda_cpu_loop_tiling(
    output: OutputTensor[dtype=DType.float32, rank=2, ...],
    Q: InputTensor[dtype=output.dtype, rank=2, ...],
    K: InputTensor[dtype=output.dtype, rank=2, ...],
    V: InputTensor[dtype=output.dtype, rank=2, ...],
) raises:
    var seq_len = Q.dim_size(0)
    var d_k = Q.dim_size(1)

    var scale: Float32 = 1.0 / sqrt(Float32(d_k))
    var J_TILE = 32
    var K_TILE = 32

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
            output.store[2](IndexList[2](i, k), Float32(out_row[k]))


def _sda_cpu_loop_tiling_SIMD(
    output: OutputTensor[dtype=DType.float32, rank=2, ...],
    Q: InputTensor[dtype=output.dtype, rank=2, ...],
    K: InputTensor[dtype=output.dtype, rank=2, ...],
    V: InputTensor[dtype=output.dtype, rank=2, ...],
) raises:
    var seq_len = Q.dim_size(0)
    var d_k = Q.dim_size(1)

    var scale: Float32 = 1.0 / sqrt(Float32(d_k))
    comptime SIMD_WIDTH = simd_width_of[DType.float32]()
    var J_TILE = 32

    for i in range(seq_len):
        var scores = List[Float32]()
        scores.resize(seq_len, 0.0)

        var q_row = List[Float32]()
        for k in range(d_k):
            var qv = Q.load[2](IndexList[2](i, k))
            q_row.append(Float32(qv[0]))


        for jj in range(0, seq_len, J_TILE):
            for j in range(jj, min(jj + J_TILE, seq_len)):
                var dot_vec = SIMD[DType.float32, SIMD_WIDTH](0.0)
                var kk = 0

                while kk + SIMD_WIDTH <= d_k:
                    var q_vec = SIMD[DType.float32, SIMD_WIDTH](0.0)
                    var k_vec = SIMD[DType.float32, SIMD_WIDTH](0.0)

                    @parameter
                    for s in range(SIMD_WIDTH):
                        q_vec[s] = q_row[kk + s]
                        var kv = K.load[2](IndexList[2](j, kk + s))
                        k_vec[s] = Float32(kv[0])

                    dot_vec = dot_vec + q_vec * k_vec
                    kk += SIMD_WIDTH

                var dot: Float32 = dot_vec.reduce_add()


                for k in range(kk, d_k):
                    var kv = K.load[2](IndexList[2](j, k))
                    dot = dot + q_row[k] * Float32(kv[0])

                scores[j] = scores[j] + dot * scale

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
            for j in range(jj, min(jj + J_TILE, seq_len)):
                var score = scores[j]
                var score_vec = SIMD[DType.float32, SIMD_WIDTH](score)
                var kk = 0

                while kk + SIMD_WIDTH <= d_k:
                    var out_vec = SIMD[DType.float32, SIMD_WIDTH](0.0)
                    var v_vec = SIMD[DType.float32, SIMD_WIDTH](0.0)

                    @parameter
                    for s in range(SIMD_WIDTH):
                        out_vec[s] = out_row[kk + s]
                        var vv = V.load[2](IndexList[2](j, kk + s))
                        v_vec[s] = Float32(vv[0])

                    out_vec = out_vec + score_vec * v_vec

                    @parameter
                    for s in range(SIMD_WIDTH):
                        out_row[kk + s] = out_vec[s]

                    kk += SIMD_WIDTH


                for k in range(kk, d_k):
                    var vv = V.load[2](IndexList[2](j, k))
                    out_row[k] = out_row[k] + score * Float32(vv[0])

        for k in range(d_k):
            output.store[2](IndexList[2](i, k), Float32(out_row[k]))


def _sda_cpu_loop_tiling_SIMD_online_softmax_parallel(
    output: OutputTensor[dtype=DType.float32, rank=2, ...],
    Q: InputTensor[dtype=output.dtype, rank=2, ...],
    K: InputTensor[dtype=output.dtype, rank=2, ...],
    V: InputTensor[dtype=output.dtype, rank=2, ...],
) raises:
    var seq_len = Q.dim_size(0)
    var d_k = Q.dim_size(1)

    var scale: Float32 = 1.0 / sqrt(Float32(d_k))
    comptime SIMD_WIDTH = simd_width_of[DType.float32]()
    var J_TILE = 32

    # ── DEBUG: force single thread to isolate parallelism vs math bug ────────
    # var num_threads = min(seq_len, num_physical_cores())
    var num_threads = 1
    # ──────────────────────────────────────────────────────────────────────────

    @parameter
    def task_func(task_id: Int) raises:
        var rows_per_thread = (seq_len + num_threads - 1) // num_threads
        var start = task_id * rows_per_thread
        var end = min(start + rows_per_thread, seq_len)
        # print("task", task_id, "rows", start, "to", end - 1, "num_threads:", num_threads)
        if start >= seq_len:
            return
        for i in range(start, end):

            var running_max: Float32 = -1e9
            var running_sum: Float32 = 0.0
            var block_score = List[Float32]()
            block_score.resize(J_TILE, 0.0)

            # ── Q row load: pointer-based SIMD load ──────────────────────────
            var q_row = List[Float32]()
            for _ in range(d_k):
                q_row.append(0.0)

            var q_ptr = Q.unsafe_ptr() + i * d_k
            var kk_q = 0
            while kk_q + SIMD_WIDTH <= d_k:
                var q_vec = (q_ptr + kk_q).load[width=SIMD_WIDTH]()
                @parameter
                for s in range(SIMD_WIDTH):
                    q_row[kk_q + s] = q_vec[s]
                kk_q += SIMD_WIDTH
            for k in range(kk_q, d_k):
                q_row[k] = q_ptr[k]
            # ──────────────────────────────────────────────────────────────────

            var out_row = List[Float32]()
            for _ in range(d_k):
                out_row.append(0.0)

            for jj in range(0, seq_len, J_TILE):
                var tile_end = min(J_TILE, seq_len - jj)

                for j in range(jj, min(jj + J_TILE, seq_len)):
                    var dot_vec = SIMD[DType.float32, SIMD_WIDTH](0.0)
                    var kk = 0

                    var k_ptr = K.unsafe_ptr() + j * d_k
                    while kk + SIMD_WIDTH <= d_k:
                        var q_vec = SIMD[DType.float32, SIMD_WIDTH](0.0)
                        @parameter
                        for s in range(SIMD_WIDTH):
                            q_vec[s] = q_row[kk + s]
                        var k_vec = (k_ptr + kk).load[width=SIMD_WIDTH]()
                        dot_vec = dot_vec + q_vec * k_vec
                        kk += SIMD_WIDTH

                    var dot: Float32 = dot_vec.reduce_add()

                    for k in range(kk, d_k):
                        dot = dot + q_row[k] * k_ptr[k]

                    block_score[j - jj] = dot * scale

                var tile_max: Float32 = block_score[0]
                for j in range(1, tile_end):
                    if tile_max < block_score[j]:
                        tile_max = block_score[j]

                var new_max = max(running_max, tile_max)
                var fix_val = exp(running_max - new_max)

                for k in range(d_k):
                    out_row[k] *= fix_val

                var tile_sum: Float32 = 0.0
                for j in range(tile_end):
                    var val = exp(block_score[j] - new_max)
                    block_score[j] = val
                    tile_sum += val

                running_sum = running_sum * fix_val + tile_sum
                running_max = new_max

                for j in range(tile_end):
                    var score = block_score[j]
                    var score_vec = SIMD[DType.float32, SIMD_WIDTH](score)
                    var kk = 0

                    var v_ptr = V.unsafe_ptr() + (jj + j) * d_k
                    while kk + SIMD_WIDTH <= d_k:
                        var out_vec = SIMD[DType.float32, SIMD_WIDTH](0.0)
                        @parameter
                        for s in range(SIMD_WIDTH):
                            out_vec[s] = out_row[kk + s]
                        var v_vec = (v_ptr + kk).load[width=SIMD_WIDTH]()
                        out_vec = out_vec + score_vec * v_vec
                        @parameter
                        for s in range(SIMD_WIDTH):
                            out_row[kk + s] = out_vec[s]
                        kk += SIMD_WIDTH

                    for k in range(kk, d_k):
                        out_row[k] = out_row[k] + score * v_ptr[k]

            for k in range(d_k):
                out_row[k] = out_row[k] / running_sum

            for k in range(d_k):
                output.store[2](IndexList[2](i, k), out_row[k])

    sync_parallelize[task_func](num_threads)