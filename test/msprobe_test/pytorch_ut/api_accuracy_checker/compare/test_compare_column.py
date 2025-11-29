import unittest

from msprobe.pytorch.api_accuracy_checker.compare.compare_column import ApiPrecisionOutputColumn


class TestCompareColumns(unittest.TestCase):

    def test_api_precision_output_column(self):
        col = ApiPrecisionOutputColumn()
        self.assertIsInstance(col.to_column_value(), list)
