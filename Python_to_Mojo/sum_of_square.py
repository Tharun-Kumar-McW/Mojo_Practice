# sum_of_squares.py
import time
 
def sum_of_squares(data):
    """Compute sum of squares element by element."""
    total = 0.0
    for x in data:
        total += x * x
    return total
 
def main():
    N = 10_000_000
    data = [float(i) / N for i in range(N)]
 
    start = time.time()
    result = sum_of_squares(data)
    elapsed = time.time() - start
 
    print(f"N = {N:,}")
    print(f"Sum of squares = {result:.10f}")
    print(f"Python elapsed: {elapsed:.4f} seconds")
 
if __name__ == "__main__":
    main()