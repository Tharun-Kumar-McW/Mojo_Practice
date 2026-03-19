def fib_iterative(n : Int) -> Int:
    """Compute the N-th Fibonacci number iteratively."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
 
 
def compute_fib_series(count : Int , out results: Dict[Int, Int]):
    """Compute a series of Fibonacci numbers and store in a dict."""
    results = Dict[Int, Int]()
 
    for i in range(count):
        results[i] = fib_iterative(i)
 
def main():
    N = 50000
 
    var results : Dict[Int, Int]
    results = compute_fib_series(N)
 
    print("Computed {} Fibonacci numbers".format(N))
    print("fib(10) = {}".format(results[10]))
    print("fib(30) = {}".format(results[30]))