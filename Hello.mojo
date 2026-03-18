from std.time import perf_counter_ns   # Fix 1: 'now' is gone, use this

fn Scalar_add(a: List[Int], b: List[Int]) -> Int:
    var sum = 0
    for i in range(8):
        sum += a[i] + b[i]
    return sum

fn Vector_add(a: List[Int], b: List[Int]) -> Int:
    var sum: Int = 0
    for i in range(0, 8, 4):
        # Fix 3: explicit Int64(...) wrapping — no more implicit conversion
        var vec_a = SIMD[DType.int64, 4](
            Int64(a[i]), Int64(a[i+1]), Int64(a[i+2]), Int64(a[i+3])
        )
        var vec_b = SIMD[DType.int64, 4](
            Int64(b[i]), Int64(b[i+1]), Int64(b[i+2]), Int64(b[i+3])
        )
        var vec_sum = vec_a + vec_b
        sum += Int(vec_sum[0]) + Int(vec_sum[1]) + Int(vec_sum[2]) + Int(vec_sum[3])
    return sum

fn main():
    # Fix 2: list literal syntax with explicit type annotation
    var a: List[Int] = [1, 2, 3, 4, 5, 6, 7, 8]
    var b: List[Int] = [10, 20, 30, 40, 50, 60, 70, 80]

    var st = perf_counter_ns()
    print("Result of scalar addition:", Scalar_add(a, b))
    print("Scalar time:", perf_counter_ns() - st, "ns")

    var st2 = perf_counter_ns()
    print("Result of SIMD addition:", Vector_add(a, b))
    print("SIMD time:", perf_counter_ns() - st2, "ns")