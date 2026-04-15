# Scaled Dot Product Attention – Mojo Optimization

## Overview

This project implements **Scaled Dot Product Attention** using Mojo custom kernels and applies an optimization to improve performance.

Two versions:
- `attention-serial` → Baseline
- `attention-Loop-reordered` → Optimized

Both are verified against a NumPy implementation.

---

## Formula

Attention is computed as:

Attention(Q, K, V) = softmax((Q × Kᵀ) / √d_k) × V

---

## Baseline (Serial Version)

### How it works
1. Compute dot product (Q × Kᵀ)
2. Scale by √d_k
3. Apply softmax
4. Multiply with V

### Problem
- Accesses memory inefficiently
- Inner loop repeatedly loads:
  - `scores[j]`
  - `V[j][k]`
- Poor cache usage → slower execution

---

## Optimization (Loop Reordering)

### Change

Before:

    for k:
        for j:
            val += scores[j] * V[j][k]


After:

    for j:
        score = scores[j]
        for k:
            out_row[k] += score * V[j][k]


---

## Why This Works

### 1. Better Memory Access
- Accesses `V[j][k]` row-wise (contiguous)
- Improves cache locality

### 2. Reduces Redundant Work
- `scores[j]` is loaded once instead of multiple times

### 3. Faster Execution
- CPU cache is used efficiently
- Inner loop becomes faster

---

## Performance Results

Test:
- seq_len = 128
- d_k = 128

| Run | Serial (ms) | Optimized (ms) | Speedup |
|-----|------------|----------------|---------|
| 1 | 7.912 | 6.217 | 1.27x |
| 2 | 7.468 | 6.326 | 1.18x |
| 3 | 8.069 | 6.189 | 1.30x |
| 4 | 7.595 | 6.349 | 1.20x |

---

## Average Speedup

Average ≈ **1.24x faster**

---

## Correctness Check

np.allclose(result, expected, atol=1e-4)

Ensures both implementations produce correct output.

---

## Key Takeaways

- Loop order affects performance significantly
- Cache-friendly access improves speed
- Small change → noticeable gain (~24%)

---

## Future Work

- SIMD optimization
- Parallel execution
- GPU implementation

---

## Run

- pixi run python3 sda.py

---

## Conclusion

Loop reordering improves:
- Cache usage
- Data reuse
- Execution time

Result: **~24% performance improvement**