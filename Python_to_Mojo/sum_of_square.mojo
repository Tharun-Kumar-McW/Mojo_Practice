def sum_of_squares(data : List[Float32]) -> Float64:
    """Compute sum of squares element by element."""
    total = 0.0
    for x in data:
        total += Float64(x) * Float64(x)
    return total
 
def main():
    N = 100
    data = [Float32(i) for i in range(N)]
 
    result = sum_of_squares(data)
 
    print("sum of squares = {}".format(result))