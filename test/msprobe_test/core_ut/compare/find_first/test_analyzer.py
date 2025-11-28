import unittest
import os
import sys
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock, mock_open

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from msprobe.core.compare.find_first.analyzer import DiffAnalyzer
from msprobe.core.compare.find_first.utils import RankPath, FileCache
from msprobe.core.compare.find_first.graph import DataNode, CommunicationNode


class TestDiffAnalyzer(unittest.TestCase):
    def setUp(self):
        # 创建临时目录用于测试
        self.temp_dir = tempfile.mkdtemp()
        self.npu_path = os.path.join(self.temp_dir, "npu")
        self.bench_path = os.path.join(self.temp_dir, "bench")
        self.output_path = os.path.join(self.temp_dir, "output")
        
        # 创建目录结构
        os.makedirs(self.npu_path, exist_ok=True)
        os.makedirs(self.bench_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)
        
        # 创建测试文件
        self.create_test_files()
        
        # 初始化分析器
        self.analyzer = DiffAnalyzer(self.npu_path, self.bench_path, self.output_path)
    
    def tearDown(self):
        # 清理临时目录
        shutil.rmtree(self.temp_dir)
        # 重置FileCache单例
        FileCache._instance = None
    
    def create_test_files(self):
        # 创建比较结果文件
        compare_result_rank0 = os.path.join(self.output_path, "compare_result_rank0_123456.json")
        compare_result_rank1 = os.path.join(self.output_path, "compare_result_rank1_123456.json")
        
        # 创建测试数据
        rank0_data = {
            "Torch.add.1": {
                "is_same": True,
                "op_items": [
                    {"NPU_Name": "input.0", "NPU_Max": 1.0, "NPU_Min": 0.0, "NPU_Mean": 0.5, "NPU_Norm": 0.7, "Stack": [["Torch.add.1", {"file": "test.py", "line": 10}]]}
                ]
            },
            "Distributed.all_reduce.2": {
                "is_same": False,
                "op_items": [
                    {"NPU_Name": "input.0.dst", "NPU_Max": 1, "Stack": [["Distributed.all_reduce.2", {"file": "test.py", "line": 20}]]},
                    {"NPU_Name": "output.0", "NPU_Max": 2.0, "Stack": "N/A"}
                ]
            },
            "Torch.mul.3": {
                "is_same": False,
                "op_items": [
                    {"NPU_Name": "input.0", "NPU_Max": 2.0, "Stack": [["Torch.mul.3", {"file": "test.py", "line": 30}]]},
                    {"NPU_Name": "output.0", "NPU_Max": 4.0, "Stack": "N/A"}
                ]
            }
        }
        
        rank1_data = {
            "Torch.add.1": {
                "is_same": True,
                "op_items": [
                    {"NPU_Name": "input.0", "NPU_Max": 1.0, "Stack": [["Torch.add.1", {"file": "test.py", "line": 10}]]}
                ]
            },
            "Distributed.all_reduce.2": {
                "is_same": True,
                "op_items": [
                    {"NPU_Name": "input.0.src", "NPU_Max": 0, "Stack": [["Distributed.all_reduce.2", {"file": "test.py", "line": 20}]]},
                    {"NPU_Name": "output.0", "NPU_Max": 2.0, "Stack": "N/A"}
                ]
            }
        }
        
        # 写入测试数据
        with open(compare_result_rank0, "w") as f:
            json.dump(rank0_data, f)
        
        with open(compare_result_rank1, "w") as f:
            json.dump(rank1_data, f)

    def test_pre_process(self):
        self.analyzer._pre_process()

        # 验证路径解析
        self.assertEqual(len(self.analyzer._paths), 2)  # 应该有两个rank路径
        self.assertIn(0, self.analyzer._paths)
        self.assertIn(1, self.analyzer._paths)
    
    def test_resolve_input_path(self):
        # 测试解析输入路径
        self.analyzer._resolve_input_path(self.output_path)
        
        # 验证路径解析
        self.assertEqual(len(self.analyzer._paths), 2)  # 应该有两个rank路径
        self.assertIn(0, self.analyzer._paths)
        self.assertIn(1, self.analyzer._paths)
        self.assertEqual(self.analyzer._paths[0].rank, 0)
        self.assertEqual(self.analyzer._paths[1].rank, 1)
    
    @patch.object(FileCache, 'load_json')
    def test_pre_analyze(self, mock_load_json):
        # 模拟加载JSON数据
        mock_load_json.side_effect = lambda path: {
            "Torch.add.1.forward": {"is_same": False, "op_items": []},
            "Distributed.all_reduce.2.forward": {"is_same": True, "op_items": []}
        } if "rank0" in path else {
            "Torch.add.1.forward": {"is_same": True, "op_items": []},
            "Distributed.all_reduce.2.forward": {"is_same": True, "op_items": []}
        }
        
        # 设置路径
        self.analyzer._paths = {
            0: RankPath(0, os.path.join(self.output_path, "compare_result_rank0_123456.json")),
            1: RankPath(1, os.path.join(self.output_path, "compare_result_rank1_123456.json"))
        }
        
        # 执行预分析
        self.analyzer._pre_analyze()
        
        # 验证结果
        self.assertEqual(len(self.analyzer._diff_nodes), 1)  # 应该找到一个异常节点
        self.assertEqual(self.analyzer._diff_nodes[0].op_name, "Torch.add.1.forward")
        self.assertEqual(self.analyzer._first_comm_nodes[1], "Distributed.all_reduce.2.forward")
    
    @patch.object(DiffAnalyzer, '_analyze_comm_nodes')
    @patch.object(DiffAnalyzer, '_connect_comm_nodes')
    @patch.object(DiffAnalyzer, '_pruning')
    @patch.object(DiffAnalyzer, '_search_first_diff')
    def test_analyze(self, mock_search, mock_pruning, mock_connect, mock_analyze_comm):
        # 模拟分析过程
        self.analyzer._paths = {
            0: RankPath(0, os.path.join(self.output_path, "compare_result_rank0_123456.json")),
            1: RankPath(1, os.path.join(self.output_path, "compare_result_rank1_123456.json"))
        }
        
        # 执行分析
        self.analyzer._analyze()
        
        # 验证调用
        mock_analyze_comm.assert_called()
        mock_connect.assert_called_once()
        mock_pruning.assert_called_once()
        mock_search.assert_called_once()
    
    @patch.object(FileCache, 'load_json')
    def test_analyze_comm_nodes(self, mock_load_json):
        # 模拟加载JSON数据
        mock_load_json.return_value = {
            "Distributed.all_reduce.1.forward": {"is_same": False, "op_items": []},
            "Torch.add.2": {"is_same": True, "op_items": []},
            "Distributed.all_reduce.3.forward": {"is_same": True, "op_items": []}
        }
        
        # 设置首个通信节点
        self.analyzer._first_comm_nodes = {0: "Distributed.all_reduce.1.forward"}
        
        # 设置路径
        self.analyzer._paths = {
            0: RankPath(0, os.path.join(self.output_path, "compare_result_rank0_123456.json"))
        }
        
        # 执行通信节点分析
        result = self.analyzer._analyze_comm_nodes(0)
        
        # 验证结果
        self.assertEqual(len(result), 2)  # 应该有两个通信节点
        self.assertIn("0.Distributed.all_reduce.1.forward", result)
        self.assertIn("0.Distributed.all_reduce.3.forward", result)
    
    def test_get_node_by_id(self):
        # 设置通信节点字典
        node = MagicMock(spec=CommunicationNode)
        self.analyzer._rank_comm_nodes_dict = {0: {"0.Distributed.all_reduce.1.forward": node}}
        
        # 测试获取节点
        result = self.analyzer._get_node_by_id("0.Distributed.all_reduce.1.forward")
        
        # 验证结果
        self.assertEqual(result, node)
        
        # 测试无效节点ID
        with self.assertRaises(RuntimeError):
            self.analyzer._get_node_by_id("invalid_id")
    
    @patch('msprobe.core.compare.find_first.analyzer.save_json')
    @patch('msprobe.core.compare.find_first.analyzer.create_directory')
    @patch('msprobe.core.compare.find_first.analyzer.time')
    def test_gen_analyze_info(self, mock_time, mock_create_directory, mock_save_json):
        # 模拟时间戳
        mock_time.time_ns.return_value = 123456789
        
        # 设置异常节点
        node = MagicMock(spec=DataNode)
        node.rank = 0
        node.gen_node_info.return_value = {"op_name": "test_op"}
        self.analyzer._diff_nodes = [node]
        
        # 设置路径
        self.analyzer._paths = {0: MagicMock(spec=RankPath)}
        
        # 生成分析信息
        self.analyzer._gen_analyze_info()
        
        # 验证调用
        mock_save_json.assert_called_once()


if __name__ == '__main__':
    unittest.main()