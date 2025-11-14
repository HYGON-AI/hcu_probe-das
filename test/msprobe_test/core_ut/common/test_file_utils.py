# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tempfile
import unittest
from unittest.mock import patch, mock_open, MagicMock
from zipfile import ZipFile, ZipInfo

import pytest
from msprobe.core.common.file_utils import *


class TestFileChecks:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.test_file = str(tmp_path / "test_file.txt")
        self.test_dir = tmp_path / "test_dir"

        # Common mocks
        self.mock_stat = MagicMock()
        self.mock_stat.st_mode = 0o755
        self.mock_stat.st_uid = 1000

    def test_check_link(self):
        with patch('os.path.islink', return_value=False):
            check_link(self.test_file)

    def test_check_path_length(self):
        # Test normal path
        check_path_length(str(self.test_file))

        # Test too long path
        long_path = self.test_dir / ('a' * FileCheckConst.DIRECTORY_LENGTH)
        with pytest.raises(FileCheckException) as exc_info:
            check_path_length(str(long_path))
        assert exc_info.value.code == FileCheckException.ILLEGAL_PATH_ERROR

    def test_check_path_exists(self):
        with patch('os.path.exists', return_value=False), \
                pytest.raises(FileCheckException) as exc_info:
            check_path_exists(self.test_file)
        assert exc_info.value.code == FileCheckException.ILLEGAL_PATH_ERROR

    def test_check_path_readability(self):
        with patch('os.access', return_value=False), \
                pytest.raises(FileCheckException) as exc_info:
            check_path_readability(self.test_file)
        assert exc_info.value.code == FileCheckException.FILE_PERMISSION_ERROR

        with patch('os.access', return_value=True):
            check_path_readability(self.test_file)

    def test_check_path_writability(self):
        with patch('os.access', return_value=False), \
                pytest.raises(FileCheckException) as exc_info:
            check_path_writability(self.test_file)
        assert exc_info.value.code == FileCheckException.FILE_PERMISSION_ERROR

        with patch('os.access', return_value=True):
            check_path_writability(self.test_file)

    def test_check_path_executable(self):
        with patch('os.access', return_value=False), \
                pytest.raises(FileCheckException) as exc_info:
            check_path_executable(self.test_file)
        assert exc_info.value.code == FileCheckException.FILE_PERMISSION_ERROR

        with patch('os.access', return_value=True):
            check_path_executable(self.test_file)

    def test_check_other_user_writable(self):
        self.mock_stat.st_mode = 0o777  # Others writable
        with patch('os.stat', return_value=self.mock_stat), \
                pytest.raises(FileCheckException) as exc_info:
            check_other_user_writable(self.test_file)
        assert exc_info.value.code == FileCheckException.FILE_PERMISSION_ERROR

        self.mock_stat.st_mode = 0o755  # Others not writable
        with patch('os.stat', return_value=self.mock_stat):
            check_other_user_writable(self.test_file)

    def test_check_path_owner_consistent(self):
        with patch('os.stat', return_value=self.mock_stat), \
                patch('os.getuid', return_value=1001), \
                pytest.raises(FileCheckException) as exc_info:
            check_path_owner_consistent(self.test_file)
        assert exc_info.value.code == FileCheckException.FILE_PERMISSION_ERROR

        # Test root user case
        with patch('os.stat', return_value=self.mock_stat), \
                patch('os.getuid', return_value=0):
            check_path_owner_consistent(self.test_file)

    def test_check_path_pattern_valid(self):
        valid_paths = [
            self.test_dir / "file.txt",
            self.test_dir / "file-1.txt",
            self.test_dir / "file_1.txt",
            self.test_dir / "file.1.txt",
        ]

        invalid_paths = [
            self.test_dir / "file*.txt",
            self.test_dir / "file?.txt",
            self.test_dir / "file;.txt",
            self.test_dir / "file|.txt",
        ]

        for path in valid_paths:
            path = str(path)
            check_path_pattern_valid(path)

        for path in invalid_paths:
            path = str(path)
            with pytest.raises(FileCheckException) as exc_info:
                check_path_pattern_valid(path)
            assert exc_info.value.code == FileCheckException.ILLEGAL_PATH_ERROR

    @pytest.mark.parametrize("file_size,max_size,should_raise", [
        (100, 200, False),
        (200, 100, True),
        (1024 * 1024, 1024 * 1024 - 1, True),
    ])
    def test_check_file_size(self, file_size, max_size, should_raise):
        with patch('os.path.getsize', return_value=file_size):
            if should_raise:
                with pytest.raises(FileCheckException) as exc_info:
                    check_file_size(self.test_file, max_size)
                assert exc_info.value.code == FileCheckException.FILE_TOO_LARGE_ERROR
            else:
                check_file_size(self.test_file, max_size)


class TestFileOperations:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.test_file = tmp_path / "test_file"
        self.test_dir = tmp_path / "test_dir"
        self.yaml_file = tmp_path / "test.yaml"
        self.json_file = tmp_path / "test.json"
        self.npy_file = tmp_path / "test.npy"
        self.excel_file = tmp_path / "test.xlsx"

    def test_check_common_file_size(self):
        with patch('os.path.isfile', return_value=True), \
                patch('os.path.getsize', return_value=100):
            check_common_file_size(str(self.test_file))
            check_common_file_size(str(self.test_file.with_suffix('.csv')))

        with patch('os.path.isfile', return_value=True), \
                patch('os.path.getsize', return_value=FileCheckConst.COMMOM_FILE_SIZE + 1), \
                pytest.raises(FileCheckException) as exc_info:
            check_common_file_size(str(self.test_file))
        assert exc_info.value.code == FileCheckException.FILE_TOO_LARGE_ERROR

    def test_check_file_suffix(self):
        check_file_suffix(str(self.test_file.with_suffix('.txt')), '.txt')

        with pytest.raises(FileCheckException) as exc_info:
            check_file_suffix(str(self.test_file.with_suffix('.txt')), '.csv')
        assert exc_info.value.code == FileCheckException.INVALID_FILE_ERROR

        check_file_suffix((self.test_file.with_suffix('.txt')), None)

    def test_make_dir(self):
        with patch('os.path.isdir', return_value=False), \
                patch('os.makedirs') as mock_makedirs, \
                patch('msprobe.core.common.file_utils.FileChecker') as mock_checker:
            mock_checker.return_value.common_check.return_value = None
            make_dir(self.test_dir)
            mock_makedirs.assert_called_once_with(
                str(self.test_dir),
                mode=FileCheckConst.DATA_DIR_AUTHORITY,
                exist_ok=True
            )

    def test_load_yaml(self):
        yaml_content = """
        key: value
        list:
          - item1
          - item2
        """
        with patch('builtins.open', mock_open(read_data=yaml_content)), \
                patch('msprobe.core.common.file_utils.FileChecker') as mock_checker, \
                patch('msprobe.core.common.file_utils.FileOpen.check_file_path', return_value=None):
            mock_checker.return_value.common_check.return_value = str(self.yaml_file)
            result = load_yaml(str(self.yaml_file))
            assert result == {'key': 'value', 'list': ['item1', 'item2']}

        # Test load error
        with patch('builtins.open', mock_open(read_data="invalid: yaml: content")), \
                patch('msprobe.core.common.file_utils.FileChecker') as mock_checker, \
                pytest.raises(RuntimeError):
            mock_checker.return_value.common_check.return_value = str(self.yaml_file)
            load_yaml(str(self.yaml_file))

    def test_load_npy(self):
        mock_array = np.array([1, 2, 3])
        with patch('numpy.load', return_value=mock_array), \
                patch('msprobe.core.common.file_utils.check_file_or_directory_path', return_value=None):
            result = load_npy(str(self.npy_file))
            np.testing.assert_array_equal(result, mock_array)

        with patch('numpy.load', side_effect=Exception), \
                patch('msprobe.core.common.file_utils.check_file_or_directory_path', return_value=None), \
                pytest.raises(RuntimeError):
            load_npy(str(self.npy_file))

    def test_save_npy(self):
        mock_data = np.array([1, 2, 3])
        with patch('numpy.save') as mock_save, \
                patch('os.chmod') as mock_chmod:
            save_npy(mock_data, str(self.npy_file))
            mock_save.assert_called_once()

    def test_save_json(self):
        test_data = {'key': 'value'}
        mock_file = mock_open()

        with patch('builtins.open', mock_file), \
                patch('fcntl.flock') as mock_flock, \
                patch('json.dump') as mock_dump, \
                patch('os.chmod') as mock_chmod:
            save_json(self.json_file, test_data)
            mock_file.assert_called_once_with(str(self.json_file), 'w', encoding='utf-8')
            mock_dump.assert_called_once_with(test_data, mock_file(), indent=None)

    def test_load_json(self):
        test_data = '{"key": "value"}'
        mock_file = mock_open(read_data=test_data)

        with patch('builtins.open', mock_file), \
                patch('fcntl.flock') as mock_flock, \
                patch('msprobe.core.common.file_utils.FileOpen.check_file_path', return_value=None):
            result = load_json(str(self.json_file))
            mock_file.assert_called_once_with(str(self.json_file), 'r', encoding='utf-8')
            assert mock_flock.call_count == 2
            assert result == {'key': 'value'}

    def test_save_yaml(self):
        test_data = {'key': 'value'}
        mock_file = mock_open()

        with patch('builtins.open', mock_file), \
                patch('fcntl.flock') as mock_flock, \
                patch('yaml.dump') as mock_dump, \
                patch('os.chmod') as mock_chmod:
            save_yaml(str(self.yaml_file), test_data)
            mock_file.assert_called_once_with(str(self.yaml_file), 'w', encoding='utf-8')
            assert mock_flock.call_count == 2
            mock_dump.assert_called_once_with(test_data, mock_file(), sort_keys=False)\

    def test_save_excel_tiny(self):
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        with patch('pandas.DataFrame.to_excel') as mock_to_excel, \
                patch('pandas.ExcelWriter') as mock_writer, \
                patch('os.chmod') as mock_chmod:
            save_excel(self.excel_file, df)
            mock_to_excel.assert_called_once_with(mock_writer().__enter__(), sheet_name='Sheet1', index=False)

    def test_save_excel_large(self):
        df = pd.DataFrame({'col1': list(range(1500000)), 'col2': list(range(1500000, 0, -1))})
        with patch('pandas.DataFrame.to_excel') as mock_to_excel, \
                patch('pandas.ExcelWriter') as mock_writer, \
                patch('os.chmod') as mock_chmod:
            save_excel(self.excel_file, df)
            mock_to_excel.assert_called_with(mock_writer().__enter__(), sheet_name='part_1', index=False)

    def test_move_file(self):
        dst_file = self.test_dir / "moved_file"
        with patch('shutil.move') as mock_move, \
                patch('os.chmod') as mock_chmod, \
                patch('msprobe.core.common.file_utils.check_file_or_directory_path', return_value=None), \
                patch('msprobe.core.common.file_utils.check_path_before_create', return_value=None):
            move_file(str(self.test_file), str(dst_file))
            mock_move.assert_called_once_with(str(self.test_file), str(dst_file))

        with patch('shutil.move', side_effect=Exception), \
                patch('msprobe.core.common.file_utils.check_file_or_directory_path', return_value=None), \
                patch('msprobe.core.common.file_utils.check_path_before_create', return_value=None), \
                pytest.raises(RuntimeError):
            move_file(self.test_file, dst_file)


class TestCSVOperations:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.csv_file = tmp_path / "test.csv"
        self.test_data = [['header1', 'header2'], ['value1', 'value2']]
        self.test_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

    def test_write_csv(self):
        mock_file = mock_open()
        with patch('builtins.open', mock_file), \
                patch('csv.writer') as mock_writer, \
                patch('os.chmod') as mock_chmod:
            write_csv(self.test_data, self.csv_file)
            mock_file.assert_called_once_with(
                str(self.csv_file), 'a+', encoding='utf-8-sig'
            )
            mock_writer.return_value.writerows.assert_called_once_with(self.test_data)

    def test_write_csv_malicious_check(self):
        test_data = [['normal', '=1+1']]  # Formula injection attempt
        with pytest.raises(RuntimeError):
            write_csv(test_data, self.csv_file, malicious_check=True)

    def test_read_csv(self):
        # Test pandas read
        with patch('pandas.read_csv', return_value=self.test_df), \
                patch('msprobe.core.common.file_utils.check_file_or_directory_path', return_value=None):
            result = read_csv(str(self.csv_file), as_pd=True)
            assert isinstance(result, pd.DataFrame)
            pd.testing.assert_frame_equal(result, self.test_df)

        # Test standard csv read
        mock_file = mock_open()
        with patch('builtins.open', mock_file), \
                patch('csv.reader', return_value=self.test_data), \
                patch('msprobe.core.common.file_utils.check_file_or_directory_path', return_value=None), \
                patch('msprobe.core.common.file_utils.FileOpen.check_file_path', return_value=None):
            result = read_csv(self.csv_file, as_pd=False)
            assert result == self.test_data

    def test_write_df_to_csv(self):
        with patch('pandas.DataFrame.to_csv') as mock_to_csv, \
                patch('os.chmod') as mock_chmod:
            write_df_to_csv(self.test_df, str(self.csv_file))
            mock_to_csv.assert_called_once_with(
                str(self.csv_file),
                mode='w',
                header=True,
                index=False
            )

        # Test invalid data type
        with pytest.raises(ValueError):
            write_df_to_csv([1, 2, 3], self.csv_file)

        # Test malicious check
        df_with_formula = pd.DataFrame({'col1': ['=1+1']})
        with pytest.raises(RuntimeError):
            write_df_to_csv(df_with_formula, self.csv_file, malicious_check=True)


class TestPathOperations:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.test_path = tmp_path / "test_path"
        self.test_dir = tmp_path / "test_dir"
        self.test_file = tmp_path / "test_file.txt"

    def test_check_path_type(self):
        # Test file type
        with patch('os.path.isfile', return_value=True):
            check_path_type(self.test_file, FileCheckConst.FILE)

        with patch('os.path.isfile', return_value=False), \
                pytest.raises(FileCheckException) as exc_info:
            check_path_type(self.test_file, FileCheckConst.FILE)
        assert exc_info.value.code == FileCheckException.INVALID_FILE_ERROR

        # Test directory type
        with patch('os.path.isdir', return_value=True):
            check_path_type(self.test_dir, FileCheckConst.DIR)

        with patch('os.path.isdir', return_value=False), \
                pytest.raises(FileCheckException) as exc_info:
            check_path_type(self.test_dir, FileCheckConst.DIR)
        assert exc_info.value.code == FileCheckException.INVALID_FILE_ERROR

    def test_check_others_writable(self):
        mock_stat = MagicMock()

        # Test others writable
        mock_stat.st_mode = stat.S_IWOTH
        with patch('os.stat', return_value=mock_stat):
            assert check_others_writable(self.test_path) is True

        # Test not writable by others
        mock_stat.st_mode = stat.S_IRUSR | stat.S_IWUSR
        with patch('os.stat', return_value=mock_stat):
            assert check_others_writable(self.test_path) is False

    def test_create_directory(self):
        with patch('os.path.isdir', return_value=True), \
                patch('os.makedirs') as mock_makedirs, \
                patch('msprobe.core.common.file_utils.FileChecker') as mock_checker:
            mock_checker.return_value.common_check.return_value = None
            create_directory(str(self.test_dir))

    def test_check_path_before_create(self):
        # Test valid path
        check_path_before_create(self.test_path)

        # Test path length exceeds limit
        long_path = self.test_dir / ('a' * FileCheckConst.DIRECTORY_LENGTH)
        with pytest.raises(FileCheckException) as exc_info:
            check_path_before_create(long_path)
        assert exc_info.value.code == FileCheckException.ILLEGAL_PATH_ERROR

        # Test invalid characters
        invalid_path = self.test_dir / "test*file"
        with pytest.raises(FileCheckException) as exc_info:
            check_path_before_create(invalid_path)
        assert exc_info.value.code == FileCheckException.ILLEGAL_PATH_ERROR


class TestUtilityOperations:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.test_file = tmp_path / "test_file"
        self.test_dir = tmp_path / "test_dir"
        self.npy_file = tmp_path / "test.npy"
        self.txt_file = tmp_path / "test.txt"
        self.workbook_file = tmp_path / "test.xlsx"

    def test_save_npy_to_txt(self):
        test_data = np.array([1, 2, 3, 4])

        with patch('os.path.exists', return_value=False), \
                patch('numpy.savetxt') as mock_savetxt:
            # Test without alignment
            save_npy_to_txt(test_data, self.txt_file)
            mock_savetxt.assert_called_once()

        # Test with alignment
        with patch('os.path.exists', return_value=False), \
                patch('numpy.savetxt') as mock_savetxt:
            save_npy_to_txt(test_data, self.txt_file, align=3)
            mock_savetxt.assert_called_once()

    def test_save_workbook(self):
        mock_workbook = MagicMock()
        with patch('os.chmod') as mock_chmod:
            save_workbook(mock_workbook, self.workbook_file)
            mock_workbook.save.assert_called_once_with(str(self.workbook_file))

        # Test save error
        mock_workbook = MagicMock()
        mock_workbook.save.side_effect = Exception
        with pytest.raises(RuntimeError):
            save_workbook(mock_workbook, self.workbook_file)

    def test_remove_path(self):
        # Test remove file
        with patch('os.path.exists', return_value=True), \
                patch('os.path.islink', return_value=False), \
                patch('os.path.isfile', return_value=True), \
                patch('os.remove') as mock_remove:
            remove_path("/test_remove_path/test/test.txt")
            mock_remove.assert_called_once_with("/test_remove_path/test/test.txt")

        # Test remove directory
        with patch('os.path.exists', return_value=True), \
                patch('os.path.islink', return_value=False), \
                patch('os.path.isfile', return_value=False), \
                patch('shutil.rmtree') as mock_rmtree:
            remove_path("/test_remove_path/test")
            mock_rmtree.assert_called_once_with("/test_remove_path/test")

    def test_get_json_contents(self):
        json_content = '{"key": "value"}'
        with patch('builtins.open', mock_open(read_data=json_content)), \
                patch('os.path.exists', return_value=True), \
                patch('msprobe.core.common.file_utils.FileOpen.check_file_path', return_value=None):
            result = get_json_contents(str(self.test_file))
            assert result == {'key': 'value'}

        # Test invalid JSON
        with patch('builtins.open', mock_open(read_data='invalid json')), \
                patch('os.path.exists', return_value=True), \
                pytest.raises(FileCheckException) as exc_info:
            get_json_contents(self.test_file)
        assert exc_info.value.code == FileCheckException.FILE_PERMISSION_ERROR

    def test_get_file_content_bytes(self):
        test_content = b'test content'
        with patch('builtins.open', mock_open(read_data=test_content)), \
                patch('os.path.exists', return_value=True), \
                patch('msprobe.core.common.file_utils.FileOpen.check_file_path', return_value=None):
            result = get_file_content_bytes(self.test_file)
            assert result == test_content

    def test_os_walk_for_files(self):
        mock_walk_data = [
            (str(self.test_dir), ['dir1'], ['file1.txt']),
            (str(self.test_dir / 'dir1'), [], ['file2.txt'])
        ]

        with patch('os.walk', return_value=mock_walk_data), \
                patch('msprobe.core.common.file_utils.check_file_or_directory_path'):
            # Test with depth 1
            result = os_walk_for_files(str(self.test_dir), 2)
            assert len(result) == 2
            assert result[0]['file'] == 'file1.txt'
            assert result[1]['file'] == 'file2.txt'

            # Test with depth 0
            result = os_walk_for_files(str(self.test_dir), 1)
            assert len(result) == 1
            assert result[0]['file'] == 'file1.txt'


class TestDirectoryChecks:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.test_dir = tmp_path / "test_dir"
        self.test_file = tmp_path / "test_file"

    def test_check_dirpath_permission(self):
        with patch('msprobe.core.common.file_utils.check_others_writable', return_value=True), \
                patch('msprobe.core.common.file_utils.check_path_owner_consistent',
                      side_effect=FileCheckException(0)), \
                patch('msprobe.core.common.file_utils.logger') as mock_logger:
            check_dirpath_permission(self.test_dir)
            assert mock_logger.warning.call_count == 2

    def test_check_file_or_directory_path(self):
        with patch('msprobe.core.common.file_utils.FileChecker') as mock_checker:
            mock_checker.return_value.common_check.return_value = None
            # Test file path
            check_file_or_directory_path(self.test_file, isdir=False)
            # Test directory path
            check_file_or_directory_path(self.test_dir, isdir=True)


cur_dir = os.path.dirname(os.path.realpath(__file__))
zip_dir = os.path.join(cur_dir, 'test_temp_zip_file')


class TestCheckZipFile(unittest.TestCase):
    def setUp(self):
        os.makedirs(zip_dir, mode=0o750, exist_ok=True)

    def tearDown(self):
        if os.path.exists(zip_dir):
            shutil.rmtree(zip_dir)

    @staticmethod
    def create_fake_zip_with_sizes(file_sizes):
        """创建临时 zip 文件，file_sizes 为每个文件的大小列表，伪造一个具有 file_size=size 的 ZIP 条目"""
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".zip", dir=zip_dir)
        os.close(tmp_fd)
        with ZipFile(tmp_path, 'w', allowZip64=True) as zipf:
            for i, size in enumerate(file_sizes):
                info = ZipInfo(f"file_{i}.bin")
                zipf.writestr(info, b'')  # 实际内容为空，但声明文件大小为 size
                info.file_size = size
        return tmp_path

    def test_valid_zip(self):
        file_sizes = [100, 200, 300]
        zip_path = self.create_fake_zip_with_sizes(file_sizes)
        try:
            check_zip_file(zip_path)
        finally:
            os.remove(zip_path)

    def test_single_file_too_large(self):
        file_sizes = [FileCheckConst.MAX_FILE_IN_ZIP_SIZE + 1]
        zip_path = self.create_fake_zip_with_sizes(file_sizes)
        try:
            with self.assertRaises(ValueError) as cm:
                check_zip_file(zip_path)
            self.assertIn("is too large to extract", str(cm.exception))
        finally:
            os.remove(zip_path)

    def test_total_size_too_large(self):
        count = 20
        size_each = (FileCheckConst.MAX_ZIP_SIZE // count) + 1
        file_sizes = [size_each] * count
        zip_path = self.create_fake_zip_with_sizes(file_sizes)
        try:
            with self.assertRaises(ValueError) as cm:
                check_zip_file(zip_path)
            self.assertIn("Total extracted size exceeds the limit", str(cm.exception))
        finally:
            os.remove(zip_path)
