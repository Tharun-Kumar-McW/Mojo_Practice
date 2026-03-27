import compiler
from std.runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from std.utils.index import IndexList
from std.algorithm import parallelize

@compiler.register("matmultiply")
struct MatMul:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor,
        a: InputTensor[dtype=output.dtype, rank=2, ...],
        b: InputTensor[dtype=output.dtype, rank=2, ...],
        ctx: DeviceContextPtr,
    ) raises:
        var M = output.dim_size(0)
        var N = output.dim_size(1)
        var K = a.dim_size(1)

        fn matmul_cal() capturing: #capturing is used to access variables from the outer scope
            for m in range(M):
                for col in range(N):
                    var acc = SIMD[output.dtype, 1](0.0)
                    
                    for k in range(K):
                        var av = a.load[1](IndexList[2](m, k))
                        var bv = b.load[1](IndexList[2](k, col))
                        acc += av * bv
                        
                    output.store[1](IndexList[2](m, col), acc)
        matmul_cal()