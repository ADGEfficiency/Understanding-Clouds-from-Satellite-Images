import unittest

from com.understandingclouds.rle import *


class RLETest(unittest.TestCase):
    def test_mask_to_rle(self):
        mask = np.array([[1, 0], [1, 1], [1, 0], [0, 1]])
        rle = mask_to_rle(mask)
        self.assertEqual(rle, '1 3 6 1 8 1')

    def test_mask_to_rle_only_zeros(self):
        mask = np.array([[0, 0, 0], [0, 0, 0]])
        rle = mask_to_rle(mask)
        self.assertEqual(rle, '')

    def test_mask_to_rle_only_ones(self):
        mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        rle = mask_to_rle(mask)
        self.assertEqual(rle, '1 9')

    def test_mask_to_rle_empty_mask(self):
        mask = np.array([])
        rle = mask_to_rle(mask)
        self.assertEqual(rle, '')


if __name__ == '__main__':
    unittest.main()
