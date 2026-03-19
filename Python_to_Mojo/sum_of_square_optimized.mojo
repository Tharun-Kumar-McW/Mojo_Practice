def optimized_sum_of_squares(data : Int) -> Float64:
    size = data
    var sum : Float64
    sum = (size*(size+1)*(2*size+1))/(6)
    return sum

def optimized_sum_of_squares_SIMD(data : List[Float32]) -> Float64:
    """Compute sum of squares element by element."""
    total = 0.0
    var acc : SIMD[DType.float32, 8] = SIMD[DType.float32, 8](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    var acc2 : SIMD[DType.float32, 8] = SIMD[DType.float32, 8](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    var acc3 : SIMD[DType.float32, 8] = SIMD[DType.float32, 8](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    var acc4 : SIMD[DType.float32, 8] = SIMD[DType.float32, 8](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    var acc5 : SIMD[DType.float32, 8] = SIMD[DType.float32, 8](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    var acc6 : SIMD[DType.float32, 8] = SIMD[DType.float32, 8](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    var acc7 : SIMD[DType.float32, 8] = SIMD[DType.float32, 8](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    var acc8 : SIMD[DType.float32, 8] = SIMD[DType.float32, 8](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
 
    var vector : SIMD[DType.float32, 8]
    var vector2 : SIMD[DType.float32, 8]
    var vector3 : SIMD[DType.float32, 8]
    var vector4 : SIMD[DType.float32, 8]
    var vector5 : SIMD[DType.float32, 8]
    var vector6 : SIMD[DType.float32, 8]
    var vector7 : SIMD[DType.float32, 8]
    var vector8 : SIMD[DType.float32, 8]
    var a : Int = 0
    for i in range(0, len(data)-63, 64):
        vector = SIMD[DType.float32, 8](data[i], data[i+1], data[i+2], data[i+3], data[i+4], data[i+5], data[i+6], data[i+7])
        vector2 = SIMD[DType.float32, 8](data[i+8], data[i+9], data[i+10], data[i+11], data[i+12], data[i+13], data[i+14], data[i+15])
        vector3 = SIMD[DType.float32, 8](data[i+16], data[i+17], data[i+18], data[i+19], data[i+20], data[i+21], data[i+22], data[i+23])
        vector4 = SIMD[DType.float32, 8](data[i+24], data[i+25], data[i+26], data[i+27], data[i+28], data[i+29], data[i+30], data[i+31])
        vector5 = SIMD[DType.float32, 8](data[i+32], data[i+33], data[i+34], data[i+35], data[i+36], data[i+37], data[i+38], data[i+39])
        vector6 = SIMD[DType.float32, 8](data[i+40], data[i+41], data[i+42], data[i+43], data[i+44], data[i+45], data[i+46], data[i+47])
        vector7 = SIMD[DType.float32, 8](data[i+48], data[i+49], data[i+50], data[i+51], data[i+52], data[i+53], data[i+54], data[i+55])
        vector8 = SIMD[DType.float32, 8](data[i+56], data[i+57], data[i+58], data[i+59], data[i+60], data[i+61], data[i+62], data[i+63])
        acc += vector * vector
        acc2 += vector2 * vector2
        acc3 += vector3 * vector3
        acc4 += vector4 * vector4
        acc5 += vector5 * vector5
        acc6 += vector6 * vector6
        acc7 += vector7 * vector7
        acc8 += vector8 * vector8
        a = i + 64
    for i in range(8):
        total += Float64(acc[i]) + Float64(acc2[i]) + Float64(acc3[i]) + Float64(acc4[i]) + Float64(acc5[i]) + Float64(acc6[i]) + Float64(acc7[i]) + Float64(acc8[i])
    for j in range(a, len(data)):
        total += Float64(data[j]) * Float64(data[j])
    return total

def main():
    N = 100
    data = [Float32(i) for i in range(N)]
    result = optimized_sum_of_squares(N)
    print("Optimized sum of squares = {}".format(result))
    print("Optimized sum of squares with SIMD = {}".format(optimized_sum_of_squares_SIMD(data)))