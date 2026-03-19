# fibonacci.py
def fib_iterative(n):
    """Compute the N-th Fibonacci number iteratively."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
 
 
def compute_fib_series(count):
    """Compute a series of Fibonacci numbers and store in a dict."""
    results = {}
    for i in range(count):
        results[i] = fib_iterative(i)
    return results
 
 
def main():
    N = 5000
    results = compute_fib_series(N)
    print(f"Computed {N:,} Fibonacci numbers")
    print(f"fib(10) = {results[10]}")
    print(f"fib(30) = {results[30]}")
 
 
if __name__ == "__main__":
    main()