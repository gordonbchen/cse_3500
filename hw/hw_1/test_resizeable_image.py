import unittest
import sys
import numpy as np
from PIL import Image
from resizeable_image import ResizeableImage

class TestImage(unittest.TestCase):
    def test_small(self):
        self.image_test('sunset_small.png', 23147)

    def test_large(self):
        self.image_test('sunset_full.png', 26010)

    def image_test(self, filename, expected_cost):
        image = ResizeableImage(filename)
        seam = image.best_seam()

        # Make sure the seam is of the appropriate length.
        self.assertEqual(image.height, len(seam), 'Seam wrong size.')

        # Make sure the pixels in the seam are properly connected.
        seam = sorted(seam,key=lambda x: x[1]) # sort by height
        for i in range(1, len(seam)):
            self.assertTrue(abs(seam[i][0]-seam[i-1][0]) <= 1, 'Not a proper seam.')
            self.assertEqual(i, seam[i][1], 'Not a proper seam.')

        # Make sure the energy of the seam matches what we expect.
        total = sum([image.energy(coord[0], coord[1]) for coord in seam])
        self.assertEqual(total, expected_cost)
    
    def test_dp_recur_same(self):
        for size in range(5, 12):
            pixels = np.random.randint(0, 256, size=(size, size, 3)).astype(np.uint8)
            image = ResizeableImage(Image.fromarray(pixels))
            dp_seam = image.best_seam(dp=True)
            recur_seam = image.best_seam(dp=False)
            self.assertEqual(dp_seam, recur_seam, "DP and recur seams do not match")

if __name__ == '__main__':
    unittest.main(argv = sys.argv + ['--verbose'])
