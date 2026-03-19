def fib_optimized(n : Int) -> Int:
    """Compute the N-th Fibonacci number using golden ratio."""
    if n <= 1:
        return n
    phi = (1 + 5 ** 0.5) / 2
    return Int((phi ** n - (1 - phi) ** n) / 5 ** 0.5)
 
 
def compute_fib_series(count : Int , out results: Dict[Int, Int]):
    """Compute a series of Fibonacci numbers and store in a dict."""
    results = Dict[Int, Int]()
 
    for i in range(count):
        results[i] = results[i - 1] + results[i - 2] if i > 1 else fib_optimized(i)
 
def main():
    N = 50000
 
    var results : Dict[Int, Int]
    results = compute_fib_series(N)
 
    print("Computed {} Fibonacci numbers".format(N))
    print("fib(10) = {}".format(results[10]))
    print("fib(30) = {}".format(results[30]))