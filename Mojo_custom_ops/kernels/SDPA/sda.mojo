import compiler

from std.runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from std.utils.index import IndexList

from .cpu.sda_cpu import (
    _sda_cpu_naive,
    _sda_cpu_loop_reordered,
    _sda_cpu_loop_tiling,
    _sda_cpu_loop_tiling_SIMD,
    _sda_cpu_loop_tiling_SIMD_online_softmax_parallel,
)
from .gpu.sda_gpu import (
    _sda_gpu_naive,
    _sda_gpu_coalesced,
    _sda_gpu_coalesced_shared_tiled,
    _sda_gpu_coalesced_shared_register_tiled,
    _sda_gpu_tensor_core,
)


@compiler.register("sda-custom-ops")
struct SdaCustomOps:
    @staticmethod
    def execute[
        target: StaticString, flag: Int
    ](
        output: OutputTensor[dtype=DType.float32, rank=2, ...],
        Q: InputTensor[dtype=output.dtype, rank=2, ...],
        K: InputTensor[dtype=output.dtype, rank=2, ...],
        V: InputTensor[dtype=output.dtype, rank=2, ...],
        ctx: DeviceContextPtr,
    ) raises:
        comptime if target == "cpu":
            comptime if flag == 0:
                _sda_cpu_naive(output, Q, K, V)
            elif flag == 1:
                _sda_cpu_loop_reordered(output, Q, K, V)
            elif flag == 2:
                _sda_cpu_loop_tiling(output, Q, K, V)
            elif flag == 3:
                _sda_cpu_loop_tiling_SIMD(output, Q, K, V)
            elif flag == 4:
                _sda_cpu_loop_tiling_SIMD_online_softmax_parallel(
                    output, Q, K, V
                )
            else:
                raise Error("Unknown Specified flag ", flag)
        elif target == "gpu":
            comptime if flag == 0:
                _sda_gpu_naive(output, Q, K, V, ctx)
            elif flag == 1:
                _sda_gpu_coalesced(output, Q, K, V, ctx)
            elif flag == 2:
                _sda_gpu_coalesced_shared_tiled(output, Q, K, V, ctx)
            elif flag == 3:
                _sda_gpu_coalesced_shared_register_tiled(output, Q, K, V, ctx)
            elif flag == 4:
                _sda_gpu_tensor_core(output, Q, K, V, ctx)
        else:
            raise Error("Unknown Specified target ", target)
