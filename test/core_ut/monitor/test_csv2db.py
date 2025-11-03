import unittest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import pandas as pd

from msprobe.core.monitor.csv2db import (
    CSV2DBConfig,
    validate_process_num,
    validate_step_partition,
    validate_data_type_list,
    _pre_scan_single_rank,
    _pre_scan,
    process_single_rank,
    import_data,
    csv2db,
    all_data_type_list,
)
from msprobe.core.common.const import MonitorConst


class TestCSV2DBValidations(unittest.TestCase):
    def test_validate_process_num_valid(self):
        """测试有效的进程数"""
        validate_process_num(1)
        validate_process_num(MonitorConst.MAX_PROCESS_NUM)

    def test_validate_process_num_invalid(self):
        """测试无效的进程数"""
        with self.assertRaises(ValueError):
            validate_process_num(0)
        with self.assertRaises(ValueError):
            validate_process_num(-1)
        with self.assertRaises(ValueError):
            validate_process_num(MonitorConst.MAX_PROCESS_NUM + 1)

    def test_validate_step_partition_valid(self):
        """测试有效的step分区"""
        validate_step_partition(MonitorConst.MIN_PARTITION)
        validate_step_partition(MonitorConst.MAX_PARTITION)

    def test_validate_step_partition_invalid(self):
        """测试无效的step分区"""
        with self.assertRaises(ValueError):
            validate_step_partition(MonitorConst.MAX_PARTITION + 1)
        with self.assertRaises(ValueError):
            validate_step_partition(MonitorConst.MIN_PARTITION - 1)
        with self.assertRaises(TypeError):
            validate_step_partition(500.0)

    def test_validate_data_type_list_valid(self):
        """测试有效的数据类型列表"""
        validate_data_type_list(["actv", "grad_reduced"])
        validate_data_type_list(all_data_type_list[:2])

    def test_validate_data_type_list_invalid(self):
        """测试无效的数据类型列表"""
        with self.assertRaises(ValueError):
            validate_data_type_list(["invalid_type"])
        with self.assertRaises(ValueError):
            validate_data_type_list(["actv", "invalid_type"])


class TestPreScanFunctions(unittest.TestCase):
    def setUp(self):
        # 创建临时目录和测试CSV文件
        self.temp_dir = tempfile.mkdtemp()
        self.temp_dir_rank2 = tempfile.mkdtemp()
        self.test_csv_path_actv = os.path.join(self.temp_dir, "actv_0-100.csv")
        self.test_csv_path_rank2_grad = os.path.join(
            self.temp_dir_rank2, "grad_reduced_100-200.csv")
        self.test_csv_path_rank_inv = os.path.join(
            self.temp_dir_rank2, "invalid_metric_100-200.csv")

        # 创建测试CSV数据
        test_data_actv = {
            "name": ["layer1", "layer2"],
            "vpp_stage": [0, 0],
            "micro_step": [0, 1],
            "step": [10, 20],
            "min": [0.1, 0.2],
            "max": [1.0, 2.0]
        }
        test_data_grad = {
            "name": ["layer1_weight", "layer2_weight"],
            "vpp_stage": [0, 0],
            "micro_step": [0, 1],
            "step": [10, 20],
            "min": [0.1, 0.2],
            "max": [1.0, 2.0]
        }
        df = pd.DataFrame(test_data_actv)
        df.to_csv(self.test_csv_path_actv, index=False)
        df = pd.DataFrame(test_data_grad)
        df.to_csv(self.test_csv_path_rank2_grad, index=False)
        df = pd.DataFrame(test_data_grad)
        df.to_csv(self.test_csv_path_rank_inv, index=False)

    def tearDown(self):
        # 清理临时目录
        shutil.rmtree(self.temp_dir)

    def test_pre_scan_single_rank(self):
        """测试单个rank的预扫描"""
        rank = 0
        files = [self.test_csv_path_actv]
        result = _pre_scan_single_rank(rank, files)
        self.assertEqual(result["max_rank"], rank)
        self.assertEqual(result["metrics"], {"actv"})
        self.assertEqual(result["min_step"], 0)
        self.assertEqual(result["max_step"], 100)
        self.assertEqual(result["metric_stats"], {"actv": {"min", "max"}})
        self.assertEqual(len(result["targets"]), 2)

    def test_pre_scan(self):
        """测试完整预扫描流程"""
        # 模拟MonitorDB
        mock_db = MagicMock()

        # 测试数据
        data_dirs = {0: self.temp_dir, 2: self.temp_dir_rank2}
        data_type_list = ["actv", "grad_reduced"]

        result = _pre_scan(mock_db, data_dirs, data_type_list)

        self.assertEqual(sorted(list(result.keys())), [0, 2])

        mock_db.insert_dimensions.assert_called_once()
        mock_db.update_global_stats.assert_called_with(
            max_rank=2, min_step=0, max_step=200
        )


class TestProcessSingleRank(unittest.TestCase):
    @patch("msprobe.core.monitor.csv2db.MonitorDB")
    @patch("msprobe.core.monitor.csv2db.read_csv")
    def test_process_single_rank(self, mock_read_csv, mock_db_class):
        """测试处理单个rank的数据"""
        # 模拟数据库和映射
        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        mock_db.get_metric_table_name.return_value = (
            "metric_1_step_0_99", 0, 99)
        mock_db.insert_rows.return_value = 2

        # 模拟CSV数据
        mock_result = pd.DataFrame({
            "name": ["layer1", "layer2"],
            "vpp_stage": [0, 0],
            "micro_step": [0, 1],
            "step": [10, 20],
            "norm": [0.1, 0.2],
            "max": [1.0, 2.0]
        })
        mock_read_csv.return_value = mock_result

        # 测试数据
        task = (0, ["actv_10-20.csv"])
        metric_id_dict = {"actv": (1, ["norm", "max"])}
        target_dict = {("layer1", 0, 0): 1, ("layer2", 0, 1): 2}
        step_partition_size = 100
        db_path = "dummy.db"

        result = process_single_rank(
            task, metric_id_dict, target_dict, step_partition_size, db_path)

        self.assertEqual(result, 2)
        mock_db.insert_rows.assert_called_with(
            "metric_1_step_0_99", [(0, 10, 1, 0.1, 1.0), (0, 20, 2, 0.2, 2.0)]
        )


class TestImportData(unittest.TestCase):
    @patch("msprobe.core.monitor.csv2db._pre_scan")
    def test_import_data_success(self, mock_pre_scan):
        """测试数据导入成功场景"""
        # 模拟预扫描结果
        mock_pre_scan.return_value = {
            0: ["actv_10-20.csv"], 1: ["actv_10-20.csv"]}

        # 模拟数据库
        mock_db = MagicMock()
        mock_db.get_metric_mapping.return_value = {"actv": (1, ["min", "max"])}
        mock_db.get_target_mapping.return_value = {("layer1", 0, 0): 1}

        # 测试数据
        data_dirs = {0: "dir0", 1: "dir1"}
        data_type_list = ["actv"]
        workers = 2

        import_data(mock_db, data_dirs, data_type_list, workers)

        mock_db.init_schema.assert_called_once()
        mock_pre_scan.assert_called_once()

    @patch("msprobe.core.monitor.csv2db._pre_scan")
    def test_import_data_no_files(self, mock_pre_scan):
        """测试没有找到数据文件的情况"""
        mock_pre_scan.return_value = {}

        mock_db = MagicMock()
        data_dirs = {0: "dir0"}
        data_type_list = ["actv"]

        result = import_data(mock_db, data_dirs, data_type_list)

        self.assertFalse(result)
        mock_pre_scan.assert_called_once()


class TestCSV2DBMain(unittest.TestCase):
    @patch("msprobe.core.monitor.csv2db.import_data")
    @patch("msprobe.core.monitor.csv2db.get_target_output_dir")
    @patch("msprobe.core.monitor.csv2db.create_directory")
    def test_csv2db(self, mock_create_dir, mock_get_dirs, mock_import):
        """测试主函数csv2db"""
        # 模拟配置
        config = CSV2DBConfig(
            monitor_path="test_path",
            data_type_list=["actv"],
            process_num=4,
            step_partition=500
        )

        # 模拟依赖函数
        mock_get_dirs.return_value = {0: "dir0", 1: "dir1"}
        mock_import.return_value = True

        csv2db(config)

        mock_get_dirs.assert_called_once()
        mock_create_dir.assert_called_once()
        mock_import.assert_called_once()
