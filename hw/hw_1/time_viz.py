import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from resizeable_image import ResizeableImage


def create_random_image(size: int) -> ResizeableImage:
    pixels = np.random.randint(0, 256, size=(size, size, 3)).astype(np.uint8)
    image = ResizeableImage(Image.fromarray(pixels))
    return image


def time_seams(
    images: list[ResizeableImage], dp: bool
) -> tuple[list[int], list[float]]:
    sizes = []
    times = []
    print(f"{dp=}")
    for image in images:
        t0 = time.time()
        image.best_seam(dp=dp)
        elapsed_time = time.time() - t0
        sizes.append(image.height)
        times.append(elapsed_time)
        print(f"{str(image.height):7s} {elapsed_time:.6f} sec")
    return sizes, times


if __name__ == "__main__":
    small_images = [create_random_image(s) for s in range(1, 14)]
    large_images = [create_random_image(100 * s) for s in range(1, 13)]
    given_images = list(
        map(
            ResizeableImage,
            ("sunset_small.png", "sunset_full.png", "cat_fortress_small.jpg"),
        )
    )

    plt.plot(*time_seams(given_images, dp=True), "r.-", label="DP Given")
    plt.plot(*time_seams(small_images + large_images, dp=True), "g.-", label="DP")
    plt.plot(*time_seams(small_images, dp=False), "b.-", label="Recur")

    plt.title("Times for Computing Best Vertical Seam")
    plt.xlabel("Image Height (pixels)")
    plt.ylabel("Time (sec)")
    plt.legend(loc="best")
    plt.savefig("seam_times.png")
