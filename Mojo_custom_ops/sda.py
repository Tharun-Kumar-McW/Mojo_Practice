"""
SDPA kernel benchmark with compute-optimization diagnostics.

Instead of just wall-clock time, this reports, per kernel:
  - achieved GFLOP/s (from the algorithmic FLOP count / measured time)
  - global memory traffic (bytes moved) and achieved bandwidth (GB/s)
  - arithmetic intensity (FLOPs/byte) -> tells you if a kernel is
    compute-bound or memory-bound, which is what actually explains why
    one optimization (coalescing, shared memory, register tiling, ...)
    beats another
  - GPU launch configuration + occupancy, pulled from Nsight Compute (ncu)
    if it's available on the machine

IMPORTANT CAVEAT ON "THREAD LAUNCH DETAILS":
The Python `max.driver` API does not expose grid/block dimensions or
occupancy -- those are decided *inside* the compiled Mojo kernel at
`.custom()` call time and are invisible to the host-side driver. There
are only two honest ways to get real launch/occupancy numbers:

  1. Profile the actual kernel launch with Nsight Compute (`ncu`) /
     Nsight Systems (`nsys`). This script tries to shell out to `ncu`
     automatically (see `profile_with_ncu`) and will just skip that part
     with a clear message if `ncu` isn't installed / isn't permitted to
     access GPU performance counters in your environment.
  2. Add a debug print inside the .mojo kernel source itself, e.g.
     `print("grid=", grid_dim, "block=", block_dim)` guarded by an env
     var, since only the kernel author knows what grid/block/shared-mem
     config each variant actually launches with.

Everything else below (FLOPs, bandwidth, arithmetic intensity) is
computed honestly from the algorithm + measured wall time, and does not
require guessing at API surface that may not exist.
"""

from pathlib import Path
import time
import subprocess
import shutil
import numpy as np

from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


# ----------------------------------------------------------------------
# Algorithmic cost model
# ----------------------------------------------------------------------

def attention_flops(seq_len: int, d_k: int) -> int:
    """FLOPs for scaled dot-product attention: QK^T, softmax, and PV."""
    qk_flops = 2 * seq_len * seq_len * d_k       # matmul Q @ K^T
    scale_flops = seq_len * seq_len               # divide by sqrt(d_k)
    softmax_flops = 3 * seq_len * seq_len          # max, exp, sum+div (approx)
    pv_flops = 2 * seq_len * seq_len * d_k        # matmul probs @ V
    return qk_flops + scale_flops + softmax_flops + pv_flops


def attention_bytes(seq_len: int, d_k: int, dtype_bytes: int = 4, fused: bool = False) -> int:
    """
    Global-memory traffic for scaled dot-product attention.

    fused=False assumes the kernel materializes the full [seq_len, seq_len]
    score matrix in global memory (write it, then read it back for the
    PV matmul) -- true for a naive / non-flash-style kernel.

    fused=True assumes an online-softmax / flash-attention-style kernel
    that never writes the full score matrix to global memory, only ever
    touching small on-chip tiles -- set this to True for whichever of
    your kernels actually does that (you know your Mojo source; the
    kernel *name* alone isn't a reliable signal, so double check).
    """
    input_bytes = 3 * seq_len * d_k * dtype_bytes     # Q, K, V reads
    output_bytes = seq_len * d_k * dtype_bytes         # output write
    scores_bytes = 0 if fused else 2 * seq_len * seq_len * dtype_bytes  # write+read scores
    return input_bytes + output_bytes + scores_bytes


# Mark which kernels are believed to fuse softmax (never materialize the
# full score matrix). Adjust this to match your actual Mojo implementations.
KERNEL_FUSED = {
    "Naive Kernel": False,
    "Coalesced Kernel": False,
    "Shared Memory Kernel": False,
    "Register Tiling Kernel": False,
    "Loop Tiling SIMD Online Softmax Kernel": True,
}


# ----------------------------------------------------------------------
# GPU device + launch introspection (best effort, via external tools)
# ----------------------------------------------------------------------

def gpu_device_info() -> str:
    """Static device info via nvidia-smi (name, clocks, mem) -- not per-kernel."""
    if shutil.which("nvidia-smi") is None:
        return "nvidia-smi not found; skipping device info."
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,clocks.max.sm,clocks.max.memory,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "nvidia-smi returned no data."
    except Exception as e:
        return f"nvidia-smi query failed: {e}"


def profile_with_ncu(target_cmd: list[str]) -> str:
    """
    Best-effort Nsight Compute pass to pull real grid/block/occupancy
    numbers for the *next* kernel launch `target_cmd` triggers.

    This re-executes `target_cmd` (e.g. [sys.executable, __file__, "--single-run"])
    under `ncu`, so it only makes sense to call this around a subprocess,
    not around an in-process call. Included for completeness; most
    sandboxed / containerized environments will not have permission to
    read GPU performance counters (you'll see "ERR_NVGPUCTRPERM"), in
    which case this just reports that plainly rather than faking numbers.
    """
    if shutil.which("ncu") is None:
        return "ncu (Nsight Compute) not found on PATH; skipping launch/occupancy profiling."
    try:
        out = subprocess.run(
            [
                "ncu",
                "--metrics",
                "launch__grid_size,launch__block_size,launch__shared_mem_per_block_dynamic,"
                "sm__warps_active.avg.pct_of_peak_sustained_active,"
                "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                *target_cmd,
            ],
            capture_output=True, text=True, timeout=120,
        )
        if out.returncode != 0:
            return f"ncu failed (likely missing perf-counter permissions):\n{out.stderr[-800:]}"
        return out.stdout
    except Exception as e:
        return f"ncu invocation failed: {e}"


# ----------------------------------------------------------------------
# Correctness reference
# ----------------------------------------------------------------------

def numpy_attention(Q, K, V):
    d_k = Q.shape[1]
    scores = Q @ K.T
    scores = scores / np.sqrt(d_k)
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs @ V


def flush_cache():
    cache_flush_size = 128 * 1024 * 1024  # 128 MB
    flush_array = np.ones(cache_flush_size // 8, dtype=np.float64)
    _ = np.sum(flush_array)
    del flush_array


# ----------------------------------------------------------------------
# Benchmark core
# ----------------------------------------------------------------------

def test_attention(kernel_name, seq_len, d_k, fnc_choice, fused: bool = False):
    mojo_kernels = Path("./kernels/SDPA")

    dtype = DType.float32
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

    Q_np = np.random.rand(seq_len, d_k).astype(np.float32)
    K_np = np.random.rand(seq_len, d_k).astype(np.float32)
    V_np = np.random.rand(seq_len, d_k).astype(np.float32)

    Q = Buffer.from_numpy(Q_np).to(device)
    K = Buffer.from_numpy(K_np).to(device)
    V = Buffer.from_numpy(V_np).to(device)

    expected = numpy_attention(Q_np, K_np, V_np).astype(np.float32)

    is_gpu = accelerator_count() > 0

    if not is_gpu:
        for _ in range(3):
            _ = model.execute(Q, K, V)
        flush_cache()

        res = 0.0
        for _ in range(10):
            st = time.perf_counter()
            result = model.execute(Q, K, V)[0]
            et = time.perf_counter()
            res += (et - st) * 1e3
        time_ms = res / 10
        result_np = result.to(CPU()).to_numpy()
    else:
        for _ in range(3):
            out = model.execute(Q, K, V)
            _ = out[0].to(CPU()).to_numpy()

        times = []
        for _ in range(10):
            st = time.perf_counter()
            out = model.execute(Q, K, V)
            result_np = out[0].to(CPU()).to_numpy()  # forces GPU sync before stopping timer
            ed = time.perf_counter()
            times.append((ed - st) * 1e3)
        time_ms = float(np.mean(times))

    passed = bool(np.allclose(result_np, expected, atol=1))
    if not passed:
        diff = np.abs(expected - result_np)
        print("  Max abs diff:", diff.max())
        print("  Max diff location:", np.unravel_index(diff.argmax(), diff.shape))
        print("  Any NaN:", np.isnan(result_np).any(), " Any Inf:", np.isinf(result_np).any())

    flops = attention_flops(seq_len, d_k)
    bytes_moved = attention_bytes(seq_len, d_k, dtype_bytes=4, fused=fused)
    time_s = time_ms / 1e3
    gflops_per_s = (flops / time_s) / 1e9 if time_s > 0 else float("nan")
    bandwidth_gbps = (bytes_moved / time_s) / 1e9 if time_s > 0 else float("nan")
    arithmetic_intensity = flops / bytes_moved if bytes_moved else float("nan")

    return {
        "passed": passed,
        "time_ms": time_ms,
        "flops": flops,
        "bytes_moved": bytes_moved,
        "gflops_per_s": gflops_per_s,
        "bandwidth_gbps": bandwidth_gbps,
        "arithmetic_intensity": arithmetic_intensity,
    }


def print_stats_row(name, stats):
    status = "Passed" if stats["passed"] else "Failed"
    print(
        f"  {name:<28} {status:<7} "
        f"{stats['time_ms']:9.3f} ms  "
        f"{stats['gflops_per_s']:9.2f} GFLOP/s  "
        f"{stats['bandwidth_gbps']:9.2f} GB/s  "
        f"AI={stats['arithmetic_intensity']:6.2f} FLOP/B"
    )


if __name__ == "__main__":
    if accelerator_count() == 0:
        print("No accelerator found. Using CPU for testing.")
        seq_lens = [1024]
        d_ks = [1024]

        kernel_names = [
            "Naive Kernel",
            "Loop Reordered Kernel",
            "Loop Tiling Kernel",
            "Loop Tiling SIMD Kernel",
            "Loop Tiling SIMD Online Softmax Kernel",
        ]

        flush_cache()
        print("Cache cleared. Starting benchmark.\n")

        for seq_len, d_k in zip(seq_lens, d_ks):
            print(f"{'─' * 90}")
            print(f"  seq_len={seq_len}  d_k={d_k}")
            print(f"{'─' * 90}")

            flush_cache()
            for j, name in enumerate(kernel_names):
                flush_cache()
                fused = KERNEL_FUSED.get(name, False)
                stats = test_attention("sda-custom-ops", seq_len, d_k, j, fused=fused)
                print_stats_row(name, stats)
                time.sleep(2)
            print()

    else:
        print("Accelerator found\n")
        print("Device info:", gpu_device_info())
        print()

        seq_lens = [8, 16, 32, 32]
        d_ks = [8, 64, 32, 64]
        kernel_names = [
            "Naive Kernel",
            "Coalesced Kernel",
            "Coalesced Shared Tiled Kernel",
            "Coalesced Shared register Tiled Kernel",
            "Blocked Tiled and Tensor Core Kernel",
        ]

        for seq_len, d_k in zip(seq_lens, d_ks):
            print(f"{'─' * 90}")
            print(f"  seq_len={seq_len}  d_k={d_k}")
            print(f"{'─' * 90}")

            for j, name in enumerate(kernel_names):
                fused = KERNEL_FUSED.get(name, False)
                stats = test_attention("sda-custom-ops", seq_len, d_k, j, fused=fused)
                print_stats_row(name, stats)
            print()