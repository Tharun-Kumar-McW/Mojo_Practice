fn matrix_multiply(A: List[List[Int]], B: List[List[Int]], N: Int) raises -> List[List[Int]]:
    var C: List[List[Int]] = [[0] * N for _ in range(N)]
    for i in range(N):
        for k in range(N):
            for j in range(N):
                C[i][j] += A[i][k] * B[k][j]
    return C.copy()

fn main() raises:
    var N: Int = 250

    var A: List[List[Int]] = [[0] * N for _ in range(N)]
    var B: List[List[Int]] = [[0] * N for _ in range(N)]

    for i in range(N):
        for j in range(N):
            A[i][j] = (i * N + j)
            B[i][j] = (j * N + i)

    var C = matrix_multiply(A, B, N)

    print("Matrix size: {}x{}".format(N, N))
    print("C[0][0] = {}".format(C[0][0]))
    print("C[N-1][N-1] = {}".format(C[N-1][N-1]))