import unittest
from msprobe.visualization.graph.node_colors import NodeColors, SUMMARY_DESCRIPTION, REAL_DATA_DESCRIPTION, \
    NOT_MATCHED
from msprobe.visualization.utils import GraphConst


class TestNodeColors(unittest.TestCase):

    def test_get_info_by_mode(self):
        node_yellow = NodeColors.YELLOW_1
        summary_info = node_yellow.get_info_by_mode(GraphConst.SUMMARY_COMPARE)
        self.assertEqual(summary_info[GraphConst.VALUE], [0, 0.2])
        self.assertEqual(summary_info[GraphConst.DESCRIPTION], SUMMARY_DESCRIPTION)
        node_grey = NodeColors.GREY
        md5_info = node_grey.get_info_by_mode(GraphConst.MD5_COMPARE)
        self.assertEqual(md5_info[GraphConst.VALUE], [])
        self.assertEqual(md5_info[GraphConst.DESCRIPTION], NOT_MATCHED)
        node_red = NodeColors.RED
        real_info = node_red.get_info_by_mode(GraphConst.REAL_DATA_COMPARE)
        self.assertEqual(real_info[GraphConst.VALUE], [0.2, 1])
        self.assertEqual(real_info[GraphConst.DESCRIPTION], REAL_DATA_DESCRIPTION)
        none_info = node_yellow.get_info_by_mode("non_existent_mode")
        self.assertEqual(none_info, {})

    def test_get_node_colors(self):
        # 测试获取所有颜色信息的函数
        mode = GraphConst.SUMMARY_COMPARE
        colors_info = NodeColors.get_node_colors(mode)
        self.assertIn("#FFFCF3", colors_info)
        self.assertIn("#FFEDBE", colors_info)
        self.assertIn("#FFDC7F", colors_info)
        self.assertIn("#FFC62E", colors_info)
        self.assertIn("#FF704D", colors_info)
        self.assertIn("#C7C7C7", colors_info)

        # 确保返回的字典具有正确的描述和值范围
        expected_value_range = [0, 0.2]
        expected_description = "此节点所有输入输出的统计量相对误差, 值越大代表测量值与标杆值的偏差越大, 相对误差计算方式:|(测量值-标杆值)/标杆值|"
        self.assertEqual(colors_info["#FFFCF3"][GraphConst.VALUE], expected_value_range)
        self.assertEqual(colors_info["#FFFCF3"][GraphConst.DESCRIPTION], expected_description)

        mode = GraphConst.MD5_COMPARE
        colors_info = NodeColors.get_node_colors(mode)
        self.assertIn("#FFFCF3", colors_info)
        self.assertIn("#C7C7C7", colors_info)
        self.assertNotIn("#FFDC7F", colors_info)

        expected_value_range = [1, 1]
        expected_description = "与标杆相比, 此节点所有输入输出的md5值相同"
        self.assertEqual(colors_info["#FFFCF3"][GraphConst.VALUE], expected_value_range)
        self.assertEqual(colors_info["#FFFCF3"][GraphConst.DESCRIPTION], expected_description)

    def test_get_node_error_status(self):
        # 测试错误状态判断功能
        mode = GraphConst.SUMMARY_COMPARE
        value0 = 0
        value1 = 0.25
        value2 = 0.55
        value3 = 111
        self.assertFalse(NodeColors.get_node_error_status(mode, value0))
        self.assertFalse(NodeColors.get_node_error_status(mode, value1))
        self.assertTrue(NodeColors.get_node_error_status(mode, value2))
        self.assertTrue(NodeColors.get_node_error_status(mode, value3))


if __name__ == '__main__':
    unittest.main()
