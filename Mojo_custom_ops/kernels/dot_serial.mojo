import compiler
from std.runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from std.utils.index import IndexList
from std.algorithm import parallelize
from std.os import Atomic
from std.sys.info import simd_width_of

@compiler.register("dot-serial")
struct DotSerial:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=DType.int32, rank=1, ...],
        a: InputTensor[dtype=output.dtype, rank=1, ...],
        b: InputTensor[dtype=output.dtype, rank=1, ...],
        ctx: DeviceContextPtr,
    ) raises:
        var size = a.dim_size(0)
        
        var acc: Int32 = 0
        for i in range(size):
            acc += a.load[1](IndexList[1](i)) * b.load[1](IndexList[1](i))
        output.store[1](IndexList[1](0), acc)

@compiler.register("dot-parallel")
struct DotParallel:
    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor[dtype=DType.int32, rank=1, ...],
        a: InputTensor[dtype=output.dtype, rank=1, ...],
        b: InputTensor[dtype=output.dtype, rank=1, ...],
        ctx: DeviceContextPtr,
    ) raises:
        var n = a.dim_size(0)
        var total = Atomic[DType.int32](0)
        comptime chunk_size = 512 
        var num_chunks = (n + chunk_size - 1) // chunk_size

        @parameter
        fn worker(chunk_idx: Int):
            var start = chunk_idx * chunk_size
            var end = min(start + chunk_size, n)
            var local_sum: Int32 = 0
            for i in range(start, end):
                local_sum += a.load[1](IndexList[1](i)) * b.load[1](IndexList[1](i))
            total.fetch_add(local_sum)

        parallelize[worker](num_chunks)
        output.store[1](IndexList[1](0), total.load())

@compiler.register("dot-parallel-simd")
struct DotParallelSIMD:
    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor[dtype=DType.int32, rank=1, ...],
        a: InputTensor[dtype=output.dtype, rank=1, ...],
        b: InputTensor[dtype=output.dtype, rank=1, ...],
        ctx: DeviceContextPtr,
    ) raises:
        var n = a.dim_size(0)
        var total = Atomic[DType.int32](0)
        comptime chunk_size = 512
        var num_chunks = (n + chunk_size - 1) // chunk_size
        comptime simd_width = simd_width_of[DType.int32]()
        @parameter
        fn worker(chunk_idx: Int):
            var start = chunk_idx * chunk_size
            var end = min(start + chunk_size, n)
            var local_sum: Int32 = 0
            var acc = SIMD[DType.int32, simd_width](0)  
            var i = start
            while i + simd_width <= end:
                var av = SIMD[DType.int32, simd_width](
                    a.load[1](IndexList[1](i)),
                    a.load[1](IndexList[1](i + 1)),
                    a.load[1](IndexList[1](i + 2)),
                    a.load[1](IndexList[1](i + 3)),
                    a.load[1](IndexList[1](i + 4)),
                    a.load[1](IndexList[1](i + 5)),
                    a.load[1](IndexList[1](i + 6)),
                    a.load[1](IndexList[1](i + 7))
                )
                var bv = SIMD[DType.int32, simd_width](
                    b.load[1](IndexList[1](i)),
                    b.load[1](IndexList[1](i + 1)),
                    b.load[1](IndexList[1](i + 2)),
                    b.load[1](IndexList[1](i + 3)),
                    b.load[1](IndexList[1](i + 4)),
                    b.load[1](IndexList[1](i + 5)),
                    b.load[1](IndexList[1](i + 6)),
                    b.load[1](IndexList[1](i + 7))
                )
                acc += av * bv
                i += simd_width
            for j in range(simd_width):
                local_sum += acc[j]
            for i in range(i, end):
                local_sum += a.load[1](IndexList[1](i)) * b.load[1](IndexList[1](i))
            total.fetch_add(local_sum)
        parallelize[worker](num_chunks)
        output.store[1](IndexList[1](0), total.load())
