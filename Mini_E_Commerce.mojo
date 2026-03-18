@fieldwise_init
struct OrderItem(Copyable , ImplicitlyCopyable):
    var product_name: String
    var quantity: Float64
    var price_per_unit: Float64

    fn subtotal(self) -> Float64:
        return self.quantity * self.price_per_unit

fn Create_Order(items : List[String]) raises -> None:
    print("Creating order")
    var count : Int = 1
    for item in items:
        print("Added : {}".format(item))
        count += 1
    print("Total items in order:", count - 1)

fn Order_Summary(order_items : List[OrderItem]) raises -> None:
    var sum : Float64 = 0.0
    print("\nOrder Summary:")
    for item in order_items:
        var item_total = item.subtotal()
        sum += item_total
        print("{} - Quantity: {}, Subtotal: ${}".format(item.product_name, item.quantity, item_total))
    print("Subtotal: ${}".format(sum))
    var discount : Optional[Float64] = 10.0
    print("Total after discount: ${}".format(apply_discount(sum, discount)))

fn apply_discount(total : Float64, discount : Optional[Float64]) raises -> Float64:
    var discounted_amount : Float64 = 0.0
    if discount is not None:
        var discounted_amount : Float64 = (total * discount.value()) / 100.0
        return total - discounted_amount
    return total-discounted_amount

fn User_Input() raises -> None:
    var order_track = List[OrderItem]()
    var ordered_items = List[String]()
    while True:
        print("Enter product name (or 'done' to finish):")
        var product_name = input()
        if product_name == "done":
            break
        print("Enter quantity:")
        var quantity = Float64(input())
        print("Enter price per unit:")
        var price_per_unit = Float64(input())
        
        var item = OrderItem(product_name, quantity, price_per_unit)
        order_track.append(item)
        ordered_items.append(product_name)
    Order_Summary(order_track)
    Create_Order(ordered_items)

fn main() raises:
    User_Input()
