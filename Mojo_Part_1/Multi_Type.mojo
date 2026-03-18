fn aggregate_readings[*args: Intable](*values:*args) -> Int:
    var total: Int = 0
    comptime for value in range(values.__len__()):
        total += Int(values[value])
    return total

fn main():
    var sum = aggregate_readings(10, 20.5, 30, 40)
    print("Total of readings:", sum)