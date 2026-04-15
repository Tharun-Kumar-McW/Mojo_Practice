import compiler

from std.runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from std.utils.index import IndexList

from std.math import sqrt, exp
from std.collections import List


@compiler.register("attention-serial")
struct AttentionSerial:
    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor[dtype=DType.float32, rank=2, ...],
        Q: InputTensor[dtype=output.dtype, rank=2, ...],
        K: InputTensor[dtype=output.dtype, rank=2, ...],
        V: InputTensor[dtype=output.dtype, rank=2, ...],
        ctx: DeviceContextPtr,
    ) raises:

        var seq_len = Q.dim_size(0)
        var d_k = Q.dim_size(1)

        var scale: Float32 = 1.0 / sqrt(Float32(d_k))

        for i in range(seq_len):
            var scores = List[Float32]()

            var q_row = List[Float32]()
            for k in range(d_k):
                var qv = Q.load[2](IndexList[2](i, k))
                q_row.append(qv[0])

            for j in range(seq_len):
                var dot: Float32 = 0.0

                var k_row = List[Float32]()
                for k in range(d_k):
                    var kv = K.load[2](IndexList[2](j, k))
                    k_row.append(kv[0])

                for k in range(d_k):
                    dot = dot + (q_row[k] * k_row[k])

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

                    # Extract scalar again
                    val = val + (scores[j] * vv[0])

                output.store[2](IndexList[2](i, k), val)

@compiler.register("attention-Loop-reordered")
struct AttentionLoopReordered:
    @staticmethod
    def execute[target: StaticString](
        output: OutputTensor[dtype=DType.float32, rank=2, ...],
        Q: InputTensor[dtype=output.dtype, rank=2, ...],
        K: InputTensor[dtype=output.dtype, rank=2, ...],
        V: InputTensor[dtype=output.dtype, rank=2, ...],
        ctx: DeviceContextPtr,
    ) raises:

        var seq_len = Q.dim_size(0)
        var d_k = Q.dim_size(1)

        var scale: Float32 = 1.0 / sqrt(Float32(d_k))

        for i in range(seq_len):
            var scores = List[Float32]()

            var q_row = List[Float32]()
            for k in range(d_k):
                var qv = Q.load[2](IndexList[2](i, k))
                q_row.append(qv[0])

            for j in range(seq_len):
                var dot: Float32 = 0.0

                var k_row = List[Float32]()
                for k in range(d_k):
                    var kv = K.load[2](IndexList[2](j, k))
                    k_row.append(kv[0])

                for k in range(d_k):
                    dot = dot + (q_row[k] * k_row[k])

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

            # Outer loop j, Inner loop k: ensures contiguous memory access on V(j, k)
            for j in range(seq_len):
                var score = scores[j] # Load score once per row
                for k in range(d_k):
                    var vv = V.load[2](IndexList[2](j, k))
                    out_row[k] = out_row[k] + (score * vv[0])

            # Write the accumulated row out to the output tensor
            for k in range(d_k):
                output.store[2](IndexList[2](i, k), out_row[k])