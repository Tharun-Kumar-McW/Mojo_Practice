# point_distance.py

import time
import math

class Point3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def euclidean_distance(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def total_pairwise_distance(points):
    """Sum of all pairwise distances (upper triangle only)."""
    total = 0.0
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            total += euclidean_distance(points[i], points[j])
    return total

def main():
    N = 2000

    points = [Point3D(float(i), float(i * 2), float(i * 3)) for i in range(N)]

    start = time.time()
    result = total_pairwise_distance(points)
    elapsed = time.time() - start

    print(f"Points: {N}")
    print(f"Total pairwise distance: {result:.4f}")
    print(f"Python elapsed: {elapsed:.4f} seconds")

if __name__ == "__main__":
    main()