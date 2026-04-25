import compiler

from std.runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from std.utils.index import IndexList

from .cpu.sda_cpu import _sda_cpu_naive, _sda_cpu_loop_reordered, _sda_cpu_loop_tiling, _sda_cpu_loop_tiling_SIMD
from .gpu.sda_gpu import _sda_gpu_naive


@compiler.register("sda-custom-ops")
struct SdaCustomOps:
    @staticmethod
    def execute[target: StaticString, flag: Int](
        output: OutputTensor[dtype=DType.float16, rank=2, ...],
        Q: InputTensor[dtype=output.dtype, rank=2, ...],
        K: InputTensor[dtype=output.dtype, rank=2, ...],
        V: InputTensor[dtype=output.dtype, rank=2, ...],
        ctx: DeviceContextPtr,
    ) raises:
        comptime if target == "cpu":
            @parameter
            if flag == 0:
                _sda_cpu_naive(output, Q, K, V)
            elif flag == 1:
                _sda_cpu_loop_reordered(output, Q, K, V)
            elif flag == 2:
                _sda_cpu_loop_tiling(output, Q, K, V)
            elif flag == 3:
                _sda_cpu_loop_tiling_SIMD(output, Q, K, V)
        elif target == "gpu":
            _sda_gpu_naive(output, Q, K, V, ctx)
        else:
            raise Error("Unknown Specified target ", target)