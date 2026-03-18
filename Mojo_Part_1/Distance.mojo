fn Calculate() -> None:
    var velocity = SIMD[DType.float32,4](3.0, 7.5, 2.1, 9.8)
    var time = SIMD[DType.float32,4](2.0)
    var distance = velocity * time
    print("Distance :",end=" ")
    for i in range(4):
        print(Float64(distance[i]), end=" ")
    print()

fn main(): 
    Calculate()