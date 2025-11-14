import unittest

from msprobe.core.config_check.checkers.random_checker import stack_match

class TestStackMatch(unittest.TestCase):
    def test_identical_stacks(self):
        stack1 = [
            "File /project/utils/funcs.py, line 42, in calculate_sum, return a + b",
            "File /project/main.py, line 15, in main, result = calculate_sum(1, 2)"
        ]
        stack2 = [
            "File /project/utils/funcs.py, line 42, in calculate_sum, return a + b",
            "File /project/main.py, line 15, in main, result = calculate_sum(1, 2)"
        ]
        self.assertTrue(stack_match(stack1, stack2))
    
    def test_different_paths_same_file(self):
        stack1 = [
            "File /user1/project/utils/funcs.py, line 42, in calculate_sum, return a + b"
        ]
        stack2 = [
            "File /user2/another_project/utils/funcs.py, line 42, in calculate_sum, return a + b"
        ]
        # 文件名相同，函数名和代码行相同
        self.assertTrue(stack_match(stack1, stack2))
    
    def test_different_filenames(self):
        stack1 = [
            "File /project/utils/funcs.py, line 42, in calculate_sum, return a + b"
        ]
        stack2 = [
            "File /project/utils/other_funcs.py, line 42, in calculate_sum, return a + b"
        ]
        # 文件名不同
        self.assertFalse(stack_match(stack1, stack2))
    
    def test_different_line_numbers(self):
        stack1 = [
            "File /project/utils/funcs.py, line 42, in calculate_sum, return a + b"
        ]
        stack2 = [
            "File /project/utils/funcs.py, line 45, in calculate_sum, return a + b"
        ]
        self.assertTrue(stack_match(stack1, stack2))
    
    def test_different_functions(self):
        stack1 = [
            "File /project/utils/funcs.py, line 42, in calculate_sum, return a + b"
        ]
        stack2 = [
            "File /project/utils/funcs.py, line 42, in multiply, return a * b"
        ]
        self.assertFalse(stack_match(stack1, stack2))
    
    def test_similar_code_different_variables(self):
        stack1 = [
            "File /project/main.py, line 15, in main, result = calculate_sum(a, b)"
        ]
        stack2 = [
            "File /project/main.py, line 15, in main, result = calculate_sum(x, y)"
        ]
        # 代码行前缀和结构相似
        self.assertTrue(stack_match(stack1, stack2))
    
    def test_different_code_structure(self):
        stack1 = [
            "File /project/main.py, line 15, in main, result = calculate_sum(a, b)"
        ]
        stack2 = [
            "File /project/main.py, line 15, in main, print('Hello, world!')"
        ]
        self.assertFalse(stack_match(stack1, stack2))
