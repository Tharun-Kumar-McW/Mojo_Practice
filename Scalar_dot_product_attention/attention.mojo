# =====================================================================
# Scaled Dot Product Attention Implementation in Mojo
#
# This implements the core attention mechanism used in transformers:
# Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
# =====================================================================

from std.math import sqrt, exp
from std.collections import List
struct Matrix(Movable):
    var rows: Int
    var cols: Int
    var data: List[Float32]
    fn __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.data = List[Float32]()
        for _ in range(rows * cols):
            self.data.append(0.0)

    fn __init__(out self, *, deinit take: Self):
        self.rows = take.rows
        self.cols = take.cols
        self.data = take.data^
            
    fn __getitem__(self, row: Int, col: Int) -> Float32:
        return self.data[row * self.cols + col]
        
    fn __setitem__(mut self, row: Int, col: Int, val: Float32):
        self.data[row * self.cols + col] = val


fn softmax(mut x: Matrix, dim: Int):
    var rows = x.rows
    var cols = x.cols
    for i in range(rows):
        var max_val: Float32 = x[i, 0]
        for j in range(1, cols):
            if x[i, j] > max_val:
                max_val = x[i, j]

        var sum_exp: Float32 = 0.0
        for j in range(cols):
            var val = exp(x[i, j] - max_val)
            x[i, j] = val
            sum_exp += val
        
        for j in range(cols):
            x[i, j] = x[i, j] / sum_exp


fn matmul(a: Matrix, b: Matrix) -> Matrix:
    var m = a.rows
    var k = a.cols
    var n = b.cols
    
    var c = Matrix(m, n)

    for i in range(m):
        for j in range(n):
            var sum: Float32 = 0.0
            for k_idx in range(k):
                sum += a[i, k_idx] * b[k_idx, j]
            c[i, j] = sum
    
    return c^


fn transpose(x: Matrix) -> Matrix:
    var rows = x.rows
    var cols = x.cols
    
    var result = Matrix(cols, rows)
    
    for i in range(rows):
        for j in range(cols):
            result[j, i] = x[i, j]
    
    return result^


fn scaled_dot_product_attention(
    Q: Matrix,
    K: Matrix,
    V: Matrix
) -> Matrix:
    
    var seq_len = Q.rows
    var d_k = Q.cols
    
    # Q x K^T
    var K_transpose = transpose(K)
    var scores = matmul(Q, K_transpose)
    
    print("Step 1 - Q · K^T shape:", scores.rows, "x", scores.cols)
    
    # Step 2: Scale by √d_k
    var scale_factor: Float32 = 1.0 / sqrt(Float32(d_k))
    
    for i in range(seq_len):
        for j in range(seq_len):
            scores[i, j] = scores[i, j] * scale_factor
    
    print("Step 2 - After scaling by 1/√d_k")
    
    # Step 3: Apply softmax (row-wise)
    softmax(scores, dim=1)
    
    print("Step 3 - After softmax (attention weights)")
    print("Each row sums to 1.0 (probability distribution)")
    print("Printing scores after softmax:")
    for i in range(seq_len):
        print("  Token", i, ":", scores[i, 0], scores[i, 1], scores[i, 2])
    print()
    
    # Step 4: Weighted sum of values
    var output = matmul(scores, V)
    
    print("Step 4 - Final output shape:", output.rows, "x", output.cols)
    
    return output^


fn main():
    """Example usage with concrete numbers."""
    
    print("=" * 60)
    print("SCALED DOT PRODUCT ATTENTION EXAMPLE")
    print("=" * 60)
    print()
    
    var seq_len = 3
    var d_model = 4
    
    print("Sequence length:", seq_len)
    print("Embedding dimension:", d_model)
    print()
    
    var Q = Matrix(seq_len, d_model)
    var K = Matrix(seq_len, d_model)
    var V = Matrix(seq_len, d_model)
    
    Q[0, 0] = 1.0; Q[0, 1] = 0.5; Q[0, 2] = 0.2; Q[0, 3] = 0.8
    Q[1, 0] = 0.3; Q[1, 1] = 0.9; Q[1, 2] = 0.4; Q[1, 3] = 0.1
    Q[2, 0] = 0.7; Q[2, 1] = 0.2; Q[2, 2] = 0.6; Q[2, 3] = 0.5
    
    K[0, 0] = 0.8; K[0, 1] = 0.3; K[0, 2] = 0.5; K[0, 3] = 0.9
    K[1, 0] = 0.2; K[1, 1] = 0.7; K[1, 2] = 0.4; K[1, 3] = 0.3
    K[2, 0] = 0.6; K[2, 1] = 0.1; K[2, 2] = 0.8; K[2, 3] = 0.2
    
    V[0, 0] = 0.5; V[0, 1] = 0.8; V[0, 2] = 0.3; V[0, 3] = 0.1
    V[1, 0] = 0.9; V[1, 1] = 0.2; V[1, 2] = 0.6; V[1, 3] = 0.4
    V[2, 0] = 0.1; V[2, 1] = 0.7; V[2, 2] = 0.5; V[2, 3] = 0.9
    
    print("Query (Q) matrix:")
    for i in range(seq_len):
        print("  Token", i, ":", Q[i, 0], Q[i, 1], Q[i, 2], Q[i, 3])
    print()
    
    print("Key (K) matrix:")
    for i in range(seq_len):
        print("  Token", i, ":", K[i, 0], K[i, 1], K[i, 2], K[i, 3])
    print()
    
    print("Value (V) matrix:")
    for i in range(seq_len):
        print("  Token", i, ":", V[i, 0], V[i, 1], V[i, 2], V[i, 3])
    print()
    
    print("-" * 60)
    print("Computing Attention...")
    print("-" * 60)
    print()
    
    var output = scaled_dot_product_attention(Q, K, V)
    
    print()
    print("=" * 60)
    print("ATTENTION OUTPUT")
    print("=" * 60)
    print()
    print("Output (context-aware embeddings):")
    for i in range(seq_len):
        print("  Token", i, ":", output[i, 0], output[i, 1], output[i, 2], output[i, 3])
    print()
    
    print("Interpretation:")
    print("  Each output row is a weighted combination of all value vectors")
    print("  Weights determined by attention scores (query-key similarity)")
    print("  This captures context: each token now 'knows' about others")