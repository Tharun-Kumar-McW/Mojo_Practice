fn Image_pixel() -> None:
    var original_pixel = SIMD[DType.uint8,4](100, 150, 200, 50)
    var second_pixel = SIMD[DType.uint8,4](2, 1, 1, 3)
    var boost_pixel = SIMD[DType.uint8,4](20, 20, 20, 20)
    var boosted_pixel = original_pixel + boost_pixel
    var blended_pixel = boosted_pixel * second_pixel
    print("Original Pixel:", original_pixel)
    print("Boosted Pixel:", boosted_pixel)
    print("Blended Pixel:", blended_pixel)
fn main():
    Image_pixel()