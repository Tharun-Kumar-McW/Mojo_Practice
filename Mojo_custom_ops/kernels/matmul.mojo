import compiler
from std.runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from std.utils.index import IndexList
from std.algorithm import parallelize

@compiler.register("matmultiply-native")
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
                    var acc = SIMD[output.dtype, 1](0)
                    for k in range(K):
                        var av = a.load[1](IndexList[2](m, k))
                        var bv = b.load[1](IndexList[2](k, col))
                        acc += av * bv
                        
                    output.store[1](IndexList[2](m, col), acc)
        matmul_cal()

@compiler.register("matmultiply-vectorized")
struct MatMulVectorized:
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
        comptime M_TILE = 16
        comptime N_TILE = 16
        comptime K_TILE = 16
        comptime TILE = 16

        fn matmul_tiled() capturing:
            var out_ptr = output.unsafe_ptr()
            for m_tile in range(0, M, TILE):           
                for n_tile in range(0, N, TILE):       
                    var m_end = min(m_tile + TILE, M)   
                    var n_end = min(n_tile + TILE, N)   
                    for m in range(m_tile, m_end):
                        for col in range(n_tile, n_end):
                            out_ptr.store(m * N + col, SIMD[output.dtype, 1](0))
                    for k_tile in range(0, K, TILE):   
                        var k_end = min(k_tile + TILE, K)  
                        for m in range(m_tile, m_end):
                            for col in range(n_tile, n_end):
                                var acc = SIMD[output.dtype, 1](0)
                                for k in range(k_tile, k_end):
                                    var av = a.load[1](IndexList[2](m, k))
                                    var bv = b.load[1](IndexList[2](k, col))
                                    acc += av * bv
                                var flat_idx = m * N + col
                                var existing_val = out_ptr.load(flat_idx)
                                out_ptr.store(flat_idx, existing_val + acc)

        matmul_tiled()