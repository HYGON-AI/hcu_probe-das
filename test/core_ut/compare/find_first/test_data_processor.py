import unittest
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from python.msprobe.core.compare.find_first.data_processor import DataProcessor
from python.msprobe.core.common.const import Const


class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        # 创建测试路径
        self.npu_path = "/path/to/npu"
        self.bench_path = "/path/to/bench"
        self.output_path = "/path/to/output"
    
    def test_init_pytorch(self):
        # 测试PyTorch框架初始化
        processor = DataProcessor(Const.PT_FRAMEWORK)
        from python.msprobe.pytorch.compare.distributed_compare import compare_distributed

        # 验证初始化
        self.assertEqual(processor.data_frame, Const.PT_FRAMEWORK)
        self.assertEqual(processor.process_func, compare_distributed)
    
    def test_init_mindspore(self):
        # 测试MindSpore框架初始化
        processor = DataProcessor(Const.MS_FRAMEWORK)
        from python.msprobe.mindspore.compare.distributed_compare import ms_compare_distributed

        # 验证初始化
        self.assertEqual(processor.data_frame, Const.MS_FRAMEWORK)
        self.assertEqual(processor.process_func, ms_compare_distributed)

    
    def test_init_unsupported(self):
        # 测试不支持的框架
        with self.assertRaises(ValueError):
            DataProcessor("unsupported_framework")

if __name__ == '__main__':
    unittest.main()