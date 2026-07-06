# =============================================================================
# sda_gpu.mojo  —  SDPA GPU Kernel Progression
# =============================================================================
#
# Kernel 1: Naive          — 3× HBM re-reads, uncoalesced output/V store
# Kernel 2: Coalesced      — swap tx↔ty, coalesced output/V read
# Kernel 3: Shared Memory  — LayoutTensor smem tiles, 3-pass softmax
# Kernel 4: Fused Flash    — online softmax, single pass (reference pattern)
#
# All imports verified against:
#   modular_backup/max/examples/custom_ops/kernels/fused_attention.mojo
#
# Key rules:
#   - MutAnyOrigin is a BUILTIN — never import it
#   - std.memory.stack_allocation: flat [N, dtype, address_space] params, NO positional arg
#   - LayoutTensor[...].stack_allocation() is the method for LayoutTensor smem
#   - Layout.row_major(M, N) needs COMPILE-TIME M, N — use to_layout_tensor() for runtime shapes
#   - matmul["gpu", transpose_b=True]() and layout_max/layout_sum from layout.math
# =============================================================================

from std.math import exp, sqrt, ceildiv
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.compute.mma import mma
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.gpu.primitives.warp import shuffle_down
from std.gpu.sync import barrier
from std.runtime.asyncrt import DeviceContextPtr
from std.memory import stack_allocation
from tensor import InputTensor, OutputTensor
from std.utils.index import IndexList
from layout import (
    Layout,
    LayoutTensor,
)  # MutAnyOrigin is a BUILTIN — no import needed
from layout.math import max as layout_max, sum as layout_sum
from layout.tensor_core import TensorCore


# =============================================================================
# KERNEL 1: Naive SDPA
# =============================================================================
# Each thread computes one output[i, k].
# tx → row i  (UNCOALESCED: adjacent threads hit different rows)
# ty → col k
# Three full HBM passes: max, sum_exp, weighted-V.
# =============================================================================


def _sda_gpu_naive(
    output: OutputTensor[dtype=DType.float32, rank=2, ...],
    Q: InputTensor[dtype=output.dtype, rank=2, ...],
    K: InputTensor[dtype=output.dtype, rank=2, ...],
    V: InputTensor[dtype=output.dtype, rank=2, ...],
    ctx: DeviceContextPtr,
) raises:
    comptime BLOCK_X = 32
    comptime BLOCK_Y = 8

    var seq_len = output.dim_size(0)
    var d_k = output.dim_size(1)
    var gpu_ctx = ctx.get_device_context()

    @parameter
    def _kernel(seq_len_: Int, d_k_: Int) capturing -> None:
        var i = Int(
            block_dim.x * block_idx.x + thread_idx.x
        )  # row (UNCOALESCED)
        var k = Int(block_dim.y * block_idx.y + thread_idx.y)  # col

        if i >= seq_len_ or k >= d_k_:
            return

        var scale = Float32(1.0) / sqrt(Float32(d_k_))

        var max_score = Float32(-3.4028235e38)
        for j in range(seq_len_):
            var dot = Float32(0.0)
            for kk in range(d_k_):
                dot += Q[i, kk] * K[j, kk]
            dot *= scale
            if dot > max_score:
                max_score = dot

        var sum_exp = Float32(0.0)
        for j in range(seq_len_):
            var dot = Float32(0.0)
            for kk in range(d_k_):
                dot += Q[i, kk] * K[j, kk]
            sum_exp += exp(dot * scale - max_score)

        var val = Float32(0.0)
        for j in range(seq_len_):
            var dot = Float32(0.0)
            for kk in range(d_k_):
                dot += Q[i, kk] * K[j, kk]
            val += (exp(dot * scale - max_score) / sum_exp) * V[j, k]

        output[i, k] = val  # UNCOALESCED store

    var grid_x = ceildiv(seq_len, BLOCK_X)
    var grid_y = ceildiv(d_k, BLOCK_Y)

    gpu_ctx.enqueue_function_experimental[_kernel](
        Int(seq_len),
        Int(d_k),
        grid_dim=(Int(grid_x), Int(grid_y), 1),
        block_dim=(BLOCK_X, BLOCK_Y, 1),
    )


# =============================================================================
# KERNEL 2: Coalesced SDPA
# =============================================================================
# ONE change from Naive: tx → col k (adjacent threads = adjacent cols → coalesced)
# =============================================================================


def _sda_gpu_coalesced(
    output: OutputTensor[dtype=DType.float32, rank=2, ...],
    Q: InputTensor[dtype=output.dtype, rank=2, ...],
    K: InputTensor[dtype=output.dtype, rank=2, ...],
    V: InputTensor[dtype=output.dtype, rank=2, ...],
    ctx: DeviceContextPtr,
) raises:
    comptime BLOCK_X = 32
    comptime BLOCK_Y = 8

    var seq_len = output.dim_size(0)
    var d_k = output.dim_size(1)
    var gpu_ctx = ctx.get_device_context()

    @parameter
    def _kernel(seq_len_: Int, d_k_: Int) capturing -> None:
        var i = Int(block_dim.y * block_idx.y + thread_idx.y)  # row
        var k = Int(block_dim.x * block_idx.x + thread_idx.x)  # col (COALESCED)

        if i >= seq_len_ or k >= d_k_:
            return

        var scale = Float32(1.0) / sqrt(Float32(d_k_))

        var max_score = Float32(-3.4028235e38)
        for j in range(seq_len_):
            var dot = Float32(0.0)
            for kk in range(d_k_):
                dot += Q[i, kk] * K[j, kk]
            dot *= scale
            if dot > max_score:
                max_score = dot

        var sum_exp = Float32(0.0)
        for j in range(seq_len_):
            var dot = Float32(0.0)
            for kk in range(d_k_):
                dot += Q[i, kk] * K[j, kk]
            sum_exp += exp(dot * scale - max_score)

        var val = Float32(0.0)
        for j in range(seq_len_):
            var dot = Float32(0.0)
            for kk in range(d_k_):
                dot += Q[i, kk] * K[j, kk]
            val += (exp(dot * scale - max_score) / sum_exp) * V[
                j, k
            ]  # coalesced V ✓

        output[i, k] = val  # coalesced store ✓

    var grid_x = ceildiv(d_k, BLOCK_X)
    var grid_y = ceildiv(seq_len, BLOCK_Y)

    gpu_ctx.enqueue_function_experimental[_kernel](
        Int(seq_len),
        Int(d_k),
        grid_dim=(Int(grid_x), Int(grid_y), 1),
        block_dim=(BLOCK_X, BLOCK_Y, 1),
    )


def _sda_gpu_coalesced_shared_tiled(
    output: OutputTensor[dtype=DType.float32, rank=2, ...],
    Q: InputTensor[dtype=output.dtype, rank=2, ...],
    K: InputTensor[dtype=output.dtype, rank=2, ...],
    V: InputTensor[dtype=output.dtype, rank=2, ...],
    ctx: DeviceContextPtr,
) raises:
    comptime BLOCK_X = 32
    comptime BLOCK_Y = 8
    comptime MAX_D_K = 64

    var seq_len = output.dim_size(0)
    var d_k = output.dim_size(1)
    var gpu_ctx = ctx.get_device_context()

    @parameter
    def _kernel(seq_len_: Int, d_k_: Int) capturing -> None:
        var i = Int(block_dim.y * block_idx.y + thread_idx.y)
        var k = Int(block_dim.x * block_idx.x + thread_idx.x)

        if i >= seq_len_ or k >= d_k_:
            return

        var scale = Float32(1.0) / sqrt(Float32(d_k_))

        var Q_smem = stack_allocation[
            BLOCK_Y * MAX_D_K, DType.float32, address_space=AddressSpace.SHARED
        ]()
        var K_smem = stack_allocation[
            BLOCK_X * MAX_D_K, DType.float32, address_space=AddressSpace.SHARED
        ]()

        for col_base in range(0, d_k_, BLOCK_X):
            var col = col_base + Int(thread_idx.x)
            if col < d_k_:
                Q_smem[Int(thread_idx.y) * d_k_ + col] = Q[i, col]
        barrier()

        var max_score = Float32(-3.4028235e38)
        var j = 0
        while j < seq_len_:
            var k_row = j + Int(thread_idx.x)
            if k_row < seq_len_:
                for col_base in range(0, d_k_, BLOCK_Y):
                    var col = col_base + Int(thread_idx.y)
                    if col < d_k_:
                        K_smem[Int(thread_idx.x) * d_k_ + col] = K[k_row, col]
            barrier()
            for jl in range(min(BLOCK_X, seq_len_ - j)):
                var dot = Float32(0.0)
                for kk in range(d_k_):
                    dot += Q_smem[Int(thread_idx.y) * d_k_ + kk] * K_smem[jl * d_k_ + kk]
                dot *= scale
                if dot > max_score:
                    max_score = dot
            barrier()
            j += BLOCK_X

        var sum_exp = Float32(0.0)
        j = 0
        while j < seq_len_:
            var k_row = j + Int(thread_idx.x)
            if k_row < seq_len_:
                for col_base in range(0, d_k_, BLOCK_Y):
                    var col = col_base + Int(thread_idx.y)
                    if col < d_k_:
                        K_smem[Int(thread_idx.x) * d_k_ + col] = K[k_row, col]
            barrier()
            for jl in range(min(BLOCK_X, seq_len_ - j)):
                var dot = Float32(0.0)
                for kk in range(d_k_):
                    dot += Q_smem[Int(thread_idx.y) * d_k_ + kk] * K_smem[jl * d_k_ + kk]
                sum_exp += exp(dot * scale - max_score)
            barrier()
            j += BLOCK_X

        var val = Float32(0.0)
        j = 0
        while j < seq_len_:
            var k_row = j + Int(thread_idx.x)
            if k_row < seq_len_:
                for col_base in range(0, d_k_, BLOCK_Y):
                    var col = col_base + Int(thread_idx.y)
                    if col < d_k_:
                        K_smem[Int(thread_idx.x) * d_k_ + col] = K[k_row, col]
            barrier()
            for jl in range(min(BLOCK_X, seq_len_ - j)):
                var dot = Float32(0.0)
                for kk in range(d_k_):
                    dot += Q_smem[Int(thread_idx.y) * d_k_ + kk] * K_smem[jl * d_k_ + kk]
                val += (exp(dot * scale - max_score) / sum_exp) * V[j + jl, k]  # coalesced V ✓
            barrier()
            j += BLOCK_X

        output[i, k] = val  # coalesced store ✓

    var grid_x = ceildiv(d_k, BLOCK_X)
    var grid_y = ceildiv(seq_len, BLOCK_Y)

    gpu_ctx.enqueue_function_experimental[_kernel](
        Int(seq_len), Int(d_k),
        grid_dim=(Int(grid_x), Int(grid_y), 1),
        block_dim=(BLOCK_X, BLOCK_Y, 1),
    )


def _sda_gpu_coalesced_shared_register_tiled(
    output: OutputTensor[dtype=DType.float32, rank=2, ...],
    Q: InputTensor[dtype=output.dtype, rank=2, ...],
    K: InputTensor[dtype=output.dtype, rank=2, ...],
    V: InputTensor[dtype=output.dtype, rank=2, ...],
    ctx: DeviceContextPtr,
) raises:
    comptime BLOCK_X = 32
    comptime BLOCK_Y = 8
    comptime TM = 2
    comptime TN = 2
    comptime MAX_D_K = 64

    var seq_len = output.dim_size(0)
    var d_k     = output.dim_size(1)
    var gpu_ctx = ctx.get_device_context()

    @parameter
    def _kernel(seq_len_: Int, d_k_: Int) capturing -> None:

        var i_base = Int((block_dim.y * block_idx.y + thread_idx.y) * TM)
        var k_base = Int((block_dim.x * block_idx.x + thread_idx.x) * TN)

        var scale = Float32(1.0) / sqrt(Float32(d_k_))

        var Q_smem = stack_allocation[
            BLOCK_Y * TM * MAX_D_K, DType.float32,
            address_space=AddressSpace.SHARED
        ]()
        var K_smem = stack_allocation[
            BLOCK_X * MAX_D_K, DType.float32,
            address_space=AddressSpace.SHARED
        ]()

        var neg_inf  = Float32.MIN_FINITE
        var acc      = InlineArray[Float32, TM * TN](fill=0.0)
        var max_vals = InlineArray[Float32, TM](fill=neg_inf)
        var sum_vals = InlineArray[Float32, TM](fill=0.0)

        # Load Q once into shared memory
        for col_base in range(0, d_k_, BLOCK_X):
            var col = col_base + Int(thread_idx.x)
            if col < d_k_:
                comptime for row in range(TM):
                    var global_row = i_base + row
                    if global_row < seq_len_:
                        Q_smem[(Int(thread_idx.y) * TM + row) * d_k_ + col] = Q[global_row, col]
        barrier()

        # Pass 1 — find max score
        var j = 0
        while j < seq_len_:
            var k_row = j + Int(thread_idx.x)
            if k_row < seq_len_:
                for col_base in range(0, d_k_, BLOCK_Y):
                    var col = col_base + Int(thread_idx.y)
                    if col < d_k_:
                        K_smem[Int(thread_idx.x) * d_k_ + col] = K[k_row, col]
            barrier()
            for jl in range(min(BLOCK_X, seq_len_ - j)):
                comptime for row in range(TM):
                    var dot = Float32(0.0)
                    for kk in range(d_k_):
                        dot += (
                            Q_smem[(Int(thread_idx.y) * TM + row) * d_k_ + kk]
                            * K_smem[jl * d_k_ + kk]
                        )
                    dot *= scale
                    if dot > max_vals[row]:
                        max_vals[row] = dot
            barrier()
            j += BLOCK_X

        # Pass 2 — softmax denominator
        j = 0
        while j < seq_len_:
            var k_row = j + Int(thread_idx.x)
            if k_row < seq_len_:
                for col_base in range(0, d_k_, BLOCK_Y):
                    var col = col_base + Int(thread_idx.y)
                    if col < d_k_:
                        K_smem[Int(thread_idx.x) * d_k_ + col] = K[k_row, col]
            barrier()
            for jl in range(min(BLOCK_X, seq_len_ - j)):
                comptime for row in range(TM):
                    var dot = Float32(0.0)
                    for kk in range(d_k_):
                        dot += (
                            Q_smem[(Int(thread_idx.y) * TM + row) * d_k_ + kk]
                            * K_smem[jl * d_k_ + kk]
                        )
                    sum_vals[row] += exp(dot * scale - max_vals[row])
            barrier()
            j += BLOCK_X

        # Pass 3 — weighted V accumulation
        j = 0
        while j < seq_len_:
            var k_row = j + Int(thread_idx.x)
            if k_row < seq_len_:
                for col_base in range(0, d_k_, BLOCK_Y):
                    var col = col_base + Int(thread_idx.y)
                    if col < d_k_:
                        K_smem[Int(thread_idx.x) * d_k_ + col] = K[k_row, col]
            barrier()
            for jl in range(min(BLOCK_X, seq_len_ - j)):
                comptime for row in range(TM):
                    var dot = Float32(0.0)
                    for kk in range(d_k_):
                        dot += (
                            Q_smem[(Int(thread_idx.y) * TM + row) * d_k_ + kk]
                            * K_smem[jl * d_k_ + kk]
                        )
                    var weight = exp(dot * scale - max_vals[row]) / sum_vals[row]
                    comptime for col in range(TN):
                        var k_out = k_base + col
                        if k_out < d_k_:
                            acc[row * TN + col] += weight * Float32(V[j + jl, k_out])
            barrier()
            j += BLOCK_X

        # Write all TM*TN results to output — the only HBM write
        comptime for row in range(TM):
            comptime for col in range(TN):
                var i_out = i_base + row
                var k_out = k_base + col
                if i_out < seq_len_ and k_out < d_k_:
                    output[i_out, k_out] = acc[row * TN + col]

    var grid_x = ceildiv(d_k,     BLOCK_X * TN)
    var grid_y = ceildiv(seq_len, BLOCK_Y * TM)

    gpu_ctx.enqueue_function_experimental[_kernel](
        Int(seq_len), Int(d_k),
        grid_dim  = (Int(grid_x), Int(grid_y), 1),
        block_dim = (BLOCK_X, BLOCK_Y, 1),
    )


def _sda_gpu_tensor_core(
    output : OutputTensor[dtype = DType.float32, rank = 2, ...],
    Q      : InputTensor [dtype = output.dtype,  rank = 2, ...],
    K      : InputTensor [dtype = output.dtype,  rank = 2, ...],
    V      : InputTensor [dtype = output.dtype,  rank = 2, ...],
    ctx    : DeviceContextPtr,
) raises:

    comptime WARP_M  = 16
    comptime WARP_N  = 8
    comptime WARP_K  = 16
    comptime BLOCK_X = 64
    comptime BLOCK_Y = 2
    comptime WARPS_X = BLOCK_X // 32
    comptime WARPS_Y = BLOCK_Y
    comptime THREADS = BLOCK_X * BLOCK_Y
    comptime BM      = WARPS_Y * WARP_M
    comptime BN      = WARPS_X * WARP_N
    comptime BK      = WARP_K
    comptime MAX_D_K = 64

    var seq_len = output.dim_size(0)
    var d_k     = output.dim_size(1)
    var gpu_ctx = ctx.get_device_context()

    @parameter
    def _kernel(seq_len_: Int, d_k_: Int) capturing -> None:

        var warp_id  = Int(thread_idx.y) * WARPS_X + Int(thread_idx.x) // 32
        var lane_id  = Int(thread_idx.x) % 32
        var warp_row = warp_id // WARPS_X
        var warp_col = warp_id  % WARPS_X

        var bm_start = Int(block_idx.y) * BM
        var bn_start = Int(block_idx.x) * BN
        var wm_start = bm_start + warp_row * WARP_M
        var wn_start = bn_start + warp_col * WARP_N

        var scale = Float32(1.0) / sqrt(Float32(d_k_))

        var Q_smem = stack_allocation[
            BM * MAX_D_K, DType.float16, address_space=AddressSpace.SHARED
        ]()
        var K_smem = stack_allocation[
            BK * MAX_D_K, DType.float16, address_space=AddressSpace.SHARED
        ]()
        var V_smem = stack_allocation[
            BK * MAX_D_K, DType.float16, address_space=AddressSpace.SHARED
        ]()
        var P_smem = stack_allocation[
            BM * BK, DType.float16, address_space=AddressSpace.SHARED
        ]()

        var neg_inf  = Float32.MIN_FINITE
        var max_vals = InlineArray[Float32, WARP_M](fill=neg_inf)
        var sum_vals = InlineArray[Float32, WARP_M](fill=Float32(0.0))
        var acc_pv   = InlineArray[Float32, 4](fill=Float32(0.0))

        # Load Q once
        for flat in range(
            Int(thread_idx.y) * BLOCK_X + Int(thread_idx.x),
            BM * d_k_,
            THREADS,
        ):
            var row = flat // d_k_
            var col = flat  % d_k_
            var global_row = bm_start + row
            if global_row < seq_len_:
                Q_smem[flat] = Q[global_row, col].cast[DType.float16]()
        barrier()

        # ── Pass 1: find per-row max score ────────────────────────────────────
        var j = 0
        while j < seq_len_:
            for flat in range(
                Int(thread_idx.y) * BLOCK_X + Int(thread_idx.x),
                BK * d_k_,
                THREADS,
            ):
                var row = flat // d_k_
                var col = flat  % d_k_
                var global_row = j + row
                if global_row < seq_len_:
                    K_smem[flat] = K[global_row, col].cast[DType.float16]()
            barrier()

            var c_qk = InlineArray[Float32, 4](fill=Float32(0.0))
            for kk in range(0, d_k_, WARP_K):
                # [MMA INTRINSIC: c_qk += Q_smem[warp_rows, kk:kk+WARP_K]
                #                       × K_smem[kk:kk+WARP_K, warp_cols]^T]
                pass

            for fi in range(4):
                c_qk[fi] *= scale
                var frag_row = (lane_id // 4) + (fi // 2) * 8
                var val = SIMD[DType.float32, 1](c_qk[fi])
                val = max(val, shuffle_down(val, UInt32(1)))
                val = max(val, shuffle_down(val, UInt32(2)))
                if lane_id % 4 == 0:
                    if val[0] > max_vals[frag_row]:
                        max_vals[frag_row] = val[0]

            barrier()
            j += BK

        # ── Pass 2: softmax denominator ───────────────────────────────────────
        j = 0
        while j < seq_len_:
            for flat in range(
                Int(thread_idx.y) * BLOCK_X + Int(thread_idx.x),
                BK * d_k_,
                THREADS,
            ):
                var row = flat // d_k_
                var col = flat  % d_k_
                var global_row = j + row
                if global_row < seq_len_:
                    K_smem[flat] = K[global_row, col].cast[DType.float16]()
            barrier()

            var c_qk = InlineArray[Float32, 4](fill=Float32(0.0))
            for kk in range(0, d_k_, WARP_K):
                # [MMA INTRINSIC: c_qk += Q_smem[warp_rows, kk:kk+WARP_K]
                #                       × K_smem[kk:kk+WARP_K, warp_cols]^T]
                pass

            for fi in range(4):
                var frag_row = (lane_id // 4) + (fi // 2) * 8
                sum_vals[frag_row] += exp(c_qk[fi] * scale - max_vals[frag_row])

            barrier()
            j += BK

        # ── Pass 3: output = softmax(QK^T) × V ───────────────────────────────
        j = 0
        while j < seq_len_:
            for flat in range(
                Int(thread_idx.y) * BLOCK_X + Int(thread_idx.x),
                BK * d_k_,
                THREADS,
            ):
                var row = flat // d_k_
                var col = flat  % d_k_
                var global_row = j + row
                if global_row < seq_len_:
                    K_smem[flat] = K[global_row, col].cast[DType.float16]()
                    V_smem[flat] = V[global_row, col].cast[DType.float16]()
            barrier()

            # MMA-1: Q × K
            var c_qk = InlineArray[Float32, 4](fill=Float32(0.0))
            for kk in range(0, d_k_, WARP_K):
                # [MMA INTRINSIC: c_qk += Q_smem[warp_rows, kk:kk+WARP_K]
                #                       × K_smem[kk:kk+WARP_K, warp_cols]^T]
                pass

            # Softmax → P_smem
            for fi in range(4):
                var frag_row = (lane_id // 4) + (fi // 2) * 8
                var frag_col = (lane_id  % 4) * 2 + (fi  % 2)
                var score    = c_qk[fi] * scale
                var prob     = exp(score - max_vals[frag_row]) / sum_vals[frag_row]
                var p_row    = wm_start - bm_start + frag_row
                var p_col    = wn_start - bn_start + frag_col
                if p_row < BM and p_col < BK:
                    P_smem[p_row * BK + p_col] = prob.cast[DType.float16]()
            barrier()

            # MMA-2: P × V → acc_pv
            for kk in range(0, d_k_, WARP_K):
                # [MMA INTRINSIC: acc_pv += P_smem[warp_rows, 0:BK]
                #                        × V_smem[0:BK, kk:kk+WARP_K]]
                pass

            barrier()
            j += BK

        # ── Write output ──────────────────────────────────────────────────────
        for fi in range(4):
            var out_row = wm_start + (lane_id // 4) + (fi // 2) * 8
            var out_col = wn_start + (lane_id  % 4) * 2 + (fi  % 2)
            if out_row < seq_len_ and out_col < d_k_:
                output[out_row, out_col] = acc_pv[fi]

    var grid_x = ceildiv(d_k,     BN)
    var grid_y = ceildiv(seq_len, BM)

    gpu_ctx.enqueue_function_experimental[_kernel](
        Int(seq_len), Int(d_k),
        grid_dim  = (Int(grid_x), Int(grid_y), 1),
        block_dim = (BLOCK_X, BLOCK_Y, 1),
    )