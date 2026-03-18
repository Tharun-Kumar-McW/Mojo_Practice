fn print_average(name: String, scores: List[Int]) -> None:
    var total: Int = 0
    for s in scores:
        total += s
    var avg = Float64(total) / Float64(len(scores))
    print(name, "- Average:", avg)

fn main():
    var students = Dict[String, List[Int]]()
    var alice = List[Int]()
    alice.append(80); alice.append(90); alice.append(85)
    students["Alice"] = alice.copy()

    var bob = List[Int]()
    bob.append(95); bob.append(88); bob.append(90)
    students["Bob"] = bob.copy()

    var carol = List[Int]()
    carol.append(70); carol.append(75); carol.append(75)
    students["Carol"] = carol.copy()

    for item in students.items():
        print_average(item.key, item.value)

    print("--- Lookup ---")
    var lookup = students.get("Dave")
    if lookup:
        print("Dave :", lookup.value())
    else:
        print("Dave not found in records")