from std.time import monotonic, perf_counter_ns
from std.algorithm import parallelize
from std.algorithm.functional import vectorize
from std.sys import simd_width_of


comptime type = DType.int64

fn serial_sum(data: List[Int], size: Int) -> Int:
    var total: Int = 0
    var st = monotonic()
    for i in range(size):
        total += data[i]
    var end = monotonic()
    print("Serial Time: ", end - st, " ns seconds")
    return total

fn parallel_cpu_sum(data: List[Int], size: Int) -> Int:
    comptime num_cores = 6
    # comptime simd_width = 8
    comptime simd_width = 8
    var partial = List[Int]()
    for _ in range(num_cores):
        partial.append(0)
    var st = perf_counter_ns()
    @parameter
    fn worker(core_id: Int):
        var chunk  = size // num_cores
        var start  = core_id * chunk
        var end    = start + chunk if core_id < num_cores - 1 else size
        var local  = 0
        var acc1 = SIMD[DType.int64, simd_width](0, 0, 0, 0, 0, 0, 0, 0)
        var vector : SIMD[type, simd_width]
        var i = start
        while i+simd_width <= end:
            vector = SIMD[type, simd_width](Int64(data[i]), Int64(data[i+1]), Int64(data[i+2]), Int64(data[i+3]), Int64(data[i+4]), Int64(data[i+5]), Int64(data[i+6]), Int64(data[i+7]))
            # print(vector)
            acc1 += vector
            i += simd_width
        for x in range(simd_width):
            local += Int(acc1[x])
        for j in range(i, end):
            local += data[j]
        partial[core_id] = local

        # for i in range(start, end):
        #     local += data[i]
        # partial[core_id] = local

    parallelize[worker](num_cores)

    var total = 0
    for v in partial:
        total += v
    var end = perf_counter_ns()
    print("Parallel Time: ", end - st, " ns seconds")
    return total

fn main():
    var num_elements: Int = 10000000

    var data_list = List[Int]()
    data_list.reserve(num_elements)

    for i in range(num_elements):
        data_list.append(i)

    # var st = monotonic()
    # var s_res = serial_sum(data_list, num_elements)
    # var end = monotonic()
    # print("Serial Time: ", end - st, " ns seconds")


    # var stt = monotonic()
    var p_res = parallel_cpu_sum(data_list, num_elements)
    # var endt = monotonic()
    # print("Parallel Time: ", endt - stt, " ns seconds")

    # print("Serial Sum: ", s_res)
    print("Parallel Sum: ", p_res)