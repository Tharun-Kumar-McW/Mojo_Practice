def matrix_multiply(A, B, N):
    """Multiply two NxN matrices using pure Python loops."""
    C = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]
    return C

def main():
    N = 250

    # Initialize matrices with simple values
    A = [[i * N + j for j in range(N)] for i in range(N)]
    B = [[j * N + i for j in range(N)] for i in range(N)]

    C = matrix_multiply(A, B, N)

    print(f"Matrix size: {N}x{N}")
    print(f"C[0][0] = {C[0][0]}")
    print(f"C[N-1][N-1] = {C[N-1][N-1]}")

if __name__ == "__main__":
    main()