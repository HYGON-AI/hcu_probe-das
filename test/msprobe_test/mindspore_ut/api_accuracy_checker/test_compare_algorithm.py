import unittest
import sys
import logging
import os
import mindspore
import torch
from unittest.mock import MagicMock

from msprobe.mindspore.api_accuracy_checker.base_compare_algorithm import compare_algorithms
from msprobe.core.common.const import CompareConst

logging.basicConfig(stream = sys.stdout, level = logging.INFO, format = '[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

file_path = os.path.abspath(__file__)
directory = os.path.dirname(file_path)


class TestCompareAlgorithms(unittest.TestCase):

    def setUp(self):
        self.mock_torchtensor_compute_element = MagicMock()
        self.mock_torchtensor_compute_element.get_parameter.return_value = torch.Tensor([1., 2., 3.])
        self.mock_torchtensor_compute_element.get_shape.return_value = (3,)
        self.mock_mstensor_compute_element = MagicMock()
        self.mock_mstensor_compute_element.get_parameter.return_value = mindspore.Tensor([1., 1.9, 3.])
        self.mock_mstensor_compute_element.get_shape.return_value = (3,)

    def test_cosine_similarity(self):
        compare_result = compare_algorithms[CompareConst.COSINE](self.mock_torchtensor_compute_element, self.mock_mstensor_compute_element)
        self.assertAlmostEqual(compare_result.compare_value, 0.9997375534689601, places=5)
        self.assertEqual(compare_result.pass_status, CompareConst.PASS)

    def test_max_absolute_difference(self):
        compare_result = compare_algorithms[CompareConst.MAX_ABS_ERR](self.mock_torchtensor_compute_element, self.mock_mstensor_compute_element)
        self.assertAlmostEqual(compare_result.compare_value, 0.1, places=5)
        self.assertEqual(compare_result.pass_status, CompareConst.ERROR)

    def test_max_relative_difference(self):
        compare_result = compare_algorithms[CompareConst.MAX_RELATIVE_ERR](self.mock_torchtensor_compute_element, self.mock_mstensor_compute_element)
        self.assertAlmostEqual(compare_result.compare_value, 0.05, places=5)
        self.assertEqual(compare_result.pass_status, CompareConst.ERROR)

if __name__ == '__main__':
    unittest.main()
