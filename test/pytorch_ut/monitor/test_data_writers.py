import unittest
from unittest import TestCase
from unittest.mock import patch

from msprobe.core.monitor.anomaly_processor import AnomalyTurbulence
from msprobe.pytorch.monitor.data_writers import BaseWriterWithAD, WriterInput


class TestBaseWriterWithAD(TestCase):

    def setUp(self) -> None:
        self.BaseWriter = BaseWriterWithAD(WriterInput('', None, None))

    def test_get_anomalies(self):
        expected = []

        self.assertEqual(self.BaseWriter.get_anomalies(), expected)

    def test_clear_anomalies(self):
        self.BaseWriter.anomalies = ['anomaly1', 'anomaly2']
        self.BaseWriter.clear_anomalies()

        self.assertEqual(self.BaseWriter.anomalies, [])

    @patch("msprobe.pytorch.monitor.data_writers.logger")
    def test_add_scalar(self, mock_logger):
        AnomalyTurbulence_obj = AnomalyTurbulence(0.2)
        self.BaseWriter.ad_rules = [AnomalyTurbulence_obj]
        tag = ('0:1.post_attention_norm.weight/rank0/pre_grad', 'mean')
        self.BaseWriter.tag2scalars = {tag: {'avg': 1.0, 'count': 1}}
        self.BaseWriter.add_scalar(tag, 2.0)

        mock_logger.info.assert_called_once()

    def test_ad(self):
        AnomalyTurbulence_obj = AnomalyTurbulence(0.2)
        self.BaseWriter.ad_rules = [AnomalyTurbulence_obj]
        expected = True, "AnomalyTurbulence"

        self.assertEqual(self.BaseWriter._ad(2.0, 1.0), expected)

    def test_update_tag2scalars(self):
        self.BaseWriter._update_tag2scalars('tag1', 1.0)
        self.assertEqual(self.BaseWriter.tag2scalars['tag1']['avg'], 1.0)
        self.assertEqual(self.BaseWriter.tag2scalars['tag1']['count'], 1)
        self.BaseWriter._update_tag2scalars('tag1', 2.0)
        self.assertEqual(self.BaseWriter.tag2scalars['tag1']['avg'], 1.01)
        self.assertEqual(self.BaseWriter.tag2scalars['tag1']['count'], 2)


if __name__ == '__main__':
    unittest.main()
