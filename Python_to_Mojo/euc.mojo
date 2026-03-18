@fieldwise_init
struct Point3D(Copyable):
    var x: Float32
    var y: Float32
    var z: Float32

fn euclidean_distance(p1: Point3D, p2: Point3D) -> Float32:
    var dx = p1.x - p2.x
    var dy = p1.y - p2.y
    var dz = p1.z - p2.z
    return (dx * dx + dy * dy + dz * dz)**0.5

fn pair_wise_distances(points: List[Point3D]) -> Float32:
    var n = len(points)
    var total_distance: Float32 = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            var dist = euclidean_distance(points[i], points[j])
            total_distance += dist
    return total_distance

fn main():
    var N: Int = 2000
    var points: List[Point3D] = [Point3D(Float32(i), Float32(i * 2), Float32(i * 3)) for i in range(N)]
    var total_dist = pair_wise_distances(points)
    print("Total pairwise distance:", total_dist)