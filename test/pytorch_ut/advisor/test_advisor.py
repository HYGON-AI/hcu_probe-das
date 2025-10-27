import difflib
import os
import shutil
import unittest
import logging
from unittest.mock import patch

import pandas

from python.msprobe.core.advisor.advisor import Advisor
from python.msprobe.core.advisor.advisor_const import AdvisorConst


class TestAdvisor(unittest.TestCase):

    def setUp(self):
        self.base_test_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        self.input_dir = os.path.join(self.base_test_dir, 'resources')
        self.output_path = os.path.abspath(os.path.join(self.base_test_dir, 'test_output'))

        os.makedirs(self.output_path, mode=0o700, exist_ok=True)
        self.has_error = False

        self.input_data = pandas.read_csv(os.path.join(self.input_dir, 'compare_result_20230703104808.csv'))
        self.advisor = Advisor(self.input_data, self.output_path)

    def tearDown(self) -> None:
        shutil.rmtree(self.output_path, ignore_errors=True)

    @patch("os.path.realpath")
    def test_init(self, mock_realpath):
        mock_realpath.return_value = 'real_output_path'
        adv = Advisor(self.input_data, self.output_path)
        self.assertEqual(adv.out_path, 'real_output_path')

    def test_deterministic_advisor_when_api_in_need_determ_api(self):
        msg = self.advisor.deterministic_advisor('', 'Functional.layer_norm.0.forward_input.0')
        self.assertEqual(msg, AdvisorConst.DETERMINISTIC_SUGGEST)

    def test_deterministic_advisor_when_api_not_in_need_determ_api(self):
        mock_message = 'mock message'
        msg = self.advisor.deterministic_advisor(mock_message, 'Functional.linear.0.forward_input.0')
        self.assertEqual(msg, mock_message)

    def test_batch_norm_advisor(self):
        mock_message = 'mocked batch norm advisor message'
        msg1 = self.advisor.batch_norm_advisor(mock_message, AdvisorConst.FUNC_BATCH_NORM + '' +
                                               AdvisorConst.FORWARD_INPUT_1)
        msg2 = self.advisor.batch_norm_advisor(mock_message, 'Functional.linear.0.forward_output.1')
        self.assertEqual(msg1, AdvisorConst.BATCH_NORM_SUGGEST)
        self.assertEqual(msg2, mock_message)

    def test_gen_advisor_message(self):
        self.assertIn(AdvisorConst.FORWARD_OUTPUT_SUGGEST, self.advisor.gen_advisor_message(
            'Functional.linear.0.forward_output.1'))
        self.assertIn(AdvisorConst.BACKWARD_INPUT_SUGGEST, self.advisor.gen_advisor_message(
            'Functional.linear.0.backward_input.1'))

    def test_advisor_summary_file(self):
        self.advisor.analysis()
        filenames = os.listdir(self.output_path)
        for filename in filenames:
            filename = os.path.join(self.output_path, filename)
            self.result_check(os.path.join(self.input_dir, 'advisor.txt'), filename)
        self.assertFalse(self.has_error)

    def result_check(self, standard_file, output_file):
        with open(standard_file, 'r', encoding='utf-8') as st_file:
            standard_content = st_file.read().splitlines()
        with open(output_file, 'r', encoding='utf-8') as out_file:
            output_content = out_file.read().splitlines()
        result = list(difflib.unified_diff(standard_content, output_content, n=0))
        if result:
            logging.basicConfig(level=logging.INFO)
            logging.info('\n\n-------------------------------------------------------------------------')
            logging.error(f'[ERROR] {output_file.replace(self.output_path, "")} advisor summary are inconsistent.')
            logging.error('\n'.join(result))
            logging.info('\n\n-------------------------------------------------------------------------')
            self.has_error = True


if __name__ == '__main__':
    unittest.main()
