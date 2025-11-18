import unittest
from unittest.mock import patch

import numpy as np

from msprobe.core.common.file_utils import FileCheckConst
from msprobe.mindspore.compare.utils import read_npy_data


class TestReadNpyData(unittest.TestCase):

    @patch('msprobe.mindspore.compare.utils.load_npy')
    @patch('msprobe.mindspore.compare.utils.FileChecker')
    @patch('os.path.join', return_value='/fake/path/to/file.npy')
    def test_read_real_data_ms(self, mock_os, mock_file_checker, mock_load_npy):
        mock_file_checker.return_value.common_check.return_value = '/fake/path/to/file.npy'

        mock_load_npy.return_value = np.array([1.0, 2.0, 3.0])

        result = read_npy_data('/fake/dir', 'file_name.npy')

        mock_file_checker.assert_called_once_with('/fake/path/to/file.npy', FileCheckConst.FILE, FileCheckConst.READ_ABLE, FileCheckConst.NUMPY_SUFFIX, False)
        mock_load_npy.assert_called_once_with('/fake/path/to/file.npy')
        self.assertTrue(np.array_equal(result, np.array([1.0, 2.0, 3.0])))
