# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

import unittest

from msprobe.infer.utils.check.string_checker import StringChecker


class TestStringChecker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pass_msg = "pass"

    def setUp(self):
        self.sc = StringChecker()

    def test_not_str(self):
        err_msg = "is not a string"

        invalid_str = [1, 2.5, -3j, b'abc', (1,), [1, 2], {1, 2, 3}]
        for path in invalid_str:
            with self.subTest(path=path):
                res_msg = str(StringChecker().is_str().check(path))
                self.assertRegex(res_msg, err_msg)
    
    def test_str(self):
        path = "abc"
        res_msg = str(StringChecker().is_str().check(path))
        self.assertRegex(res_msg, self.pass_msg)
    
    def test_name_too_long(self):
        err_msg = "File name too long"

        invalid_str = ["s/" * 2048, 's' * 256]
        for path in invalid_str:
            with self.subTest(path=path):
                res_msg = str(StringChecker().is_file_name_too_long().check(path))
                self.assertEqual(res_msg, err_msg)
    
    def test_name_not_too_long(self):
        valid_str = ["a", "b", "c", "ab", "ac", "bc", "abc"]
        for path in valid_str:
            with self.subTest(path=path):
                res_msg = str(StringChecker().is_file_name_too_long().check(path))
                self.assertEqual(res_msg, self.pass_msg)

    def test_str_not_safe(self):
        err_msg = "String parameter contains invalid characters"

        invalid_str = ['&', '+', '@', '#', '$', "b=d", "echo xxx > /dev/null", "{", "}", "<", ">", "~", "'", '"', 
                       "[", "]", "(", ")"]
        for path in invalid_str:
            with self.subTest(path=path):
                res_msg = str(StringChecker().is_str_safe().check(path))
                self.assertEqual(res_msg, err_msg)

    def test_str_safe(self):
        valid_str = ['a', 'b', 'a_b', 'c-d', 'bd', 'rm']
        for path in valid_str:
            with self.subTest(path=path):
                res_msg = str(StringChecker().is_str_safe().check(path))
                self.assertEqual(res_msg, self.pass_msg)

    def test_str_not_valid_bool(self):
        err_msg = "Boolean value expected 'yes', 'y', 'Y', 'YES', 'true', 't', 'TRUE', 'True', '1' for true"

        invalid_str = ['n', 'no', '\n', '\t', 'k']
        for path in invalid_str:
            with self.subTest(path=path):
                res_msg = str(StringChecker().is_str_valid_bool().check(path))
                self.assertEqual(res_msg, err_msg)

    def test_str_valid_bool(self):
        valid_str = ['y', 'yes', 't', 'true', 'true', '1']
        for path in valid_str:
            with self.subTest(path=path):
                res_msg = str(StringChecker().is_str_valid_bool().check(path))
                self.assertEqual(res_msg, self.pass_msg)

    def test_str_not_valid_str(self):
        err_msg = "Input path contains invalid characters"

        invalid_str = ['>', '>>', ' xxx@xxx', '1+1=2', 'rm -rf /']
        for path in invalid_str:
            with self.subTest(path=path):
                res_msg = str(StringChecker().is_str_valid_path().check(path))
                self.assertEqual(res_msg, err_msg)
    
    def test_str_valid_str(self):
        valid_str = ['a', 'b', '1_b', 'c-d']
        for path in valid_str:
            with self.subTest(path=path):
                res_msg = str(StringChecker().is_str_valid_path().check(path))
                self.assertEqual(res_msg, self.pass_msg)
    
    def test_str_not_valid_ids(self):
        err_msg = "dym range string"

        invalid_str = ['>', '>>', ' xxx@xxx', '1+1=2', 'rm -rf /', '1_2 ', '1_2, 123']
        for path in invalid_str:
            with self.subTest(path=path):
                res_msg = str(StringChecker().is_str_valid_ids().check(path))
                self.assertRegex(res_msg, err_msg)

    def test_str_valid_ids(self):
        valid_str = ['1_2', '2_3,4_5', '4_5,6_7,796_12321', '123']
        for path in valid_str:
            with self.subTest(path=path):
                res_msg = str(StringChecker().is_str_valid_ids().check(path))
                self.assertRegex(res_msg, self.pass_msg)
    
    def test_str_has_invalid_char(self):
        err_msg = "Input string contains invalid chars"

        invalid_str = ['&', '\n', '\f', '\u007F', '@', ';', '#']
        for path in invalid_str:
            with self.subTest(path=path):
                res_msg = str(StringChecker().str_has_no_invalid_char().check(path))
                self.assertRegex(res_msg, err_msg)

    def test_str_has_no_invalid_char(self):
        valid_str = ["a", "b", "c", "ab", "ac", "bc", "abc"]
        for path in valid_str:
            with self.subTest(path=path):
                res_msg = str(StringChecker().is_file_name_too_long().check(path))
                self.assertEqual(res_msg, self.pass_msg)
