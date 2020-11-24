# encoding: utf-8

# author: cole.zhang
# contact: longzonejazz@gmail.com

import unittest
import numpy as np
from coco_nlp.utils import unison_shuffled_copies
from coco_nlp.utils import get_list_subset


class TestUtils(unittest.TestCase):

    def test_unison_shuffled_copies(self):
        x: np.ndarray = np.random.randint(0, 10, size=(100, 5))
        y: np.ndarray = np.random.randint(0, 10, size=(100, ))

        new_x, new_y = unison_shuffled_copies(x, y)
        assert new_x.shape == x.shape
        assert new_y.shape == y.shape

    def test_get_list_subset(self):
        x = list(range(0, 100))
        subset = get_list_subset(x, list(range(10, 20)))
        assert subset == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


if __name__ == "__main__":
    pass
