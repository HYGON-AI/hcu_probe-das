# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import os
import tempfile
from unittest import mock

from msprobe.infer.utils.check import path_checker


class TestPathChecker(unittest.TestCase):

    def setUp(self) -> None:
        self.fp = tempfile.NamedTemporaryFile()
        self.dp = tempfile.TemporaryDirectory()

        self.sl = "softlink"
        os.symlink(self.fp.name, self.sl)

        self.pc = path_checker.PathChecker()

    def test_exists(self):
        self.assertEqual(str(self.pc.exists().check(self.fp.name)), "pass")

    def test_not_exists(self):
        temp_name = "lsiwmcv"
        self.assertRegex(str(self.pc.exists().check(temp_name)), "No such file or directory")

    def test_is_file(self):
        self.assertEqual(str(self.pc.is_file().check(self.fp.name)), "pass")

    def test_not_file(self):
        self.assertRegex(str(self.pc.is_file().check(self.dp.name)), "Not a file")

    def test_is_dir(self):
        self.assertEqual(str(self.pc.is_dir().check(self.dp.name)), "pass")

    def test_not_dir(self):
        self.assertRegex(str(self.pc.is_dir().check(self.fp.name)), "Not a directory")

    def test_not_softlink(self):
        self.assertRegex(str(self.pc.is_softlink().check(self.fp.name)), "Not a soft link")

    def test_is_uid_matched(self):
        self.assertEqual(str(self.pc.is_uid_matched(os.getuid()).check(self.fp.name)), "pass")

    def test_is_owner(self):
        self.assertEqual(str(self.pc.is_owner(os.getuid()).check(self.fp.name)), "pass")

    @unittest.skipIf(os.getuid() == 0, "root can be skipped")
    def test_uid_not_matched(self):
        self.assertRegex(str(self.pc.is_uid_matched(0).check(self.fp.name)), "User ID not matched")

    def test_is_gid_matched(self):
        self.assertEqual(str(self.pc.is_gid_matched(os.getgid()).check(self.fp.name)), "pass")

    @unittest.skipIf(os.getuid() == 0, "any file is readable to root")
    def test_is_readable(self):
        with tempfile.NamedTemporaryFile() as fp:
            os.chmod(fp.name, 0o400)
            self.assertEqual(str(self.pc.is_readable().check(fp.name)), "pass")

    @unittest.skipIf(os.getuid() == 0, "any file is readable to root")
    def test_is_writeable(self):
        with tempfile.NamedTemporaryFile() as fp:
            os.chmod(fp.name, 0o600)
            self.assertEqual(str(self.pc.is_writeable().check(fp.name)), "pass")

    @unittest.skipIf(os.getuid() == 0, "any file is readable to root")
    def test_is_executable(self):
        with tempfile.NamedTemporaryFile() as fp:
            os.chmod(fp.name, 0o500)
            self.assertEqual(str(self.pc.is_executable().check(fp.name)), "pass")

    def test_readable_to_others(self):
        with tempfile.NamedTemporaryFile() as fp:
            os.chmod(fp.name, 0o777)
            self.assertIn("is readable to others", str(self.pc.is_not_readable_to_others().check(fp.name)))

    def test_not_readable_to_others(self):
        self.assertEqual(str(self.pc.is_not_readable_to_others().check(self.fp.name)), "pass")

    def test_writable_to_others(self):
        with tempfile.NamedTemporaryFile() as fp:
            os.chmod(fp.name, 0o777)
            self.assertIn("is writable to others", str(self.pc.is_not_writable_to_others().check(fp.name)))

    def test_not_writable_to_others(self):
        self.assertEqual(str(self.pc.is_not_writable_to_others().check(self.fp.name)), "pass")

    def test_executable_to_others(self):
        with tempfile.NamedTemporaryFile() as fp:
            os.chmod(fp.name, 0o777)
            self.assertIn("is executable to others", str(self.pc.is_not_executable_to_others().check(fp.name)))

    def test_not_executable_to_others(self):
        self.assertEqual(str(self.pc.is_not_executable_to_others().check(self.fp.name)), "pass")

    def test_higher_perm(self):
        self.assertEqual(str(self.pc.max_perm(0o777).check(self.fp.name)), "pass")

    def test_lower_perm(self):
        self.assertRegex(str(self.pc.max_perm(0o000).check(self.fp.name)), "should not have")

    def test_invalid_perm(self):
        self.assertRaises(ValueError, self.pc.max_perm(-1).check, self.fp.name)

    def test_smaller_size(self):
        self.assertEqual(str(self.pc.max_size(2048).check(self.fp.name)), "pass")

    def test_larger_size(self):
        self.assertRegex(str(self.pc.max_size(0).check(self.dp.name)), "File size larger than expected")

    def test_right_extension(self):
        self.assertEqual(str(self.pc.check_extensions("").check(self.fp.name)), "pass")

    def test_right_extension(self):
        self.assertRegex(str(self.pc.check_extensions(".bin").check(self.fp.name)), "Wrong file suffix")

    def test_file_name_too_long(self):
        self.assertRegex(str(self.pc.exists().check("s" * 256)), "File name too long")

    def test_file_name_too_long_raise(self):
        with self.assertRaises(ValueError) as cm:
            self.pc.exists().check("s" * 256, True)

        self.assertRegex(str(cm.exception), "File name too long")

    def test_file_not_found(self):
        self.assertRegex(str(self.pc.exists().check("s" * 240)), "No such file or directory")

    def test_file_not_found_raise(self):
        with self.assertRaises(ValueError) as cm:
            self.pc.exists().check("s" * 240, True)

        self.assertRegex(str(cm.exception), "No such file or directory")

    @unittest.skipIf(os.getuid() == 0, "root can be skipped")
    def test_is_safe_parent_dir_when_other_has_w_then_failed(self):
        with tempfile.TemporaryDirectory() as dp:
            os.chmod(dp, 0o702)
            fp = os.path.join(dp, "test_file")
            self.assertFalse(bool(path_checker.PathChecker().is_safe_parent_dir().check(fp)))

    @unittest.skipIf(os.getuid() == 0, "root can be skipped")
    def test_is_safe_parent_dir_when_group_has_w_then_failed(self):
        with tempfile.TemporaryDirectory() as dp:
            os.chmod(dp, 0o720)
            fp = os.path.join(dp, "test_file")
            self.assertFalse(bool(path_checker.PathChecker().is_safe_parent_dir().check(fp)))

    @unittest.skipIf(os.getuid() == 0, "root can be skipped")
    def test_is_safe_parent_dir_when_all_good_then_pass(self):
        with tempfile.TemporaryDirectory() as dp:
            os.chmod(dp, 0o750)
            fp = os.path.join(dp, "test_file")
            self.assertTrue(bool(path_checker.PathChecker().is_safe_parent_dir().check(fp)))

    @unittest.skipIf(os.getuid() == 0, "root can be skipped")
    def test_is_safe_parent_dir_when_user_is_root_then_pass(self):
        ret_root = mock.Mock(return_value=0)
        with mock.patch('os.getuid', ret_root):
            with tempfile.TemporaryDirectory() as dp:
                os.chmod(dp, 0o702)
                fp = os.path.join(dp, "test_file")
                self.assertTrue(bool(path_checker.PathChecker().is_safe_parent_dir().check(fp)))

    @unittest.skipIf(os.getuid() == 0, "root can be skipped")
    def test_no_perm(self):
        self.assertRegex(str(self.pc.exists().check("/root/a")), "Permission denied")

    @unittest.skipIf(os.getuid() == 0, "root can be skipped")
    def test_no_perm_raise(self):
        with self.assertRaises(ValueError) as cm:
            self.pc.exists().check("/root/a", True)

        self.assertRegex(str(cm.exception), "Permission denied")

    def test_error_type_raise(self):
        with self.assertRaises(TypeError):
            self.pc.exists().check(2, True)

    def tearDown(self) -> None:
        os.unlink(self.sl)
        self.fp.close()
        self.dp.cleanup()
