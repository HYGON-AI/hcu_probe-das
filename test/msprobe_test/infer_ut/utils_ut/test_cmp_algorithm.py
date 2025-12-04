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

import os
import unittest
from unittest import mock
import torch
import torch.nn.functional as F
import pytest

from msprobe.infer.utils.cmp_algorithm import cosine_similarity, max_relative_error, mean_relative_error, \
    max_absolute_error, kl_divergence, mean_absolute_error, relative_euclidean_distance, \
    CMP_ALG_MAP, register_custom_compare_algorithm


class TestMetrics(unittest.TestCase):

    def setUp(self):
        # Test data preparation
        self.golden_data_all_zeros = torch.tensor([0.0, 0.0, 0.0])
        self.my_data_all_zeros = torch.tensor([0.0, 0.0, 0.0])
        self.golden_data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        self.my_data = torch.tensor([1.1, 2.0, 3.1], dtype=torch.float64)
        self.golden_data_with_zero = torch.tensor([0.0, 2.0, 3.0], dtype=torch.float64)
        self.my_data_with_zero = torch.tensor([0.0, 2.0, 3.1], dtype=torch.float64)

    def test_cosine_similarity(self):
        similarity, _ = cosine_similarity(self.golden_data, self.my_data)
        expected_similarity = torch.cosine_similarity(
            self.golden_data.double(),
            self.my_data.double(),
            dim=0
        ).item()
        self.assertAlmostEqual(similarity, expected_similarity, places=10)
        similarity, _ = cosine_similarity(self.golden_data_all_zeros, self.my_data_all_zeros)
        self.assertEqual(similarity, 1.0)

    def test_max_relative_error(self):
        error, _ = max_relative_error(self.golden_data, self.my_data)
        expected_error = torch.max(torch.abs(self.my_data / self.golden_data - 1)).item()
        self.assertAlmostEqual(error, expected_error, places=6)

        error, _ = max_relative_error(self.golden_data_with_zero, self.my_data_with_zero)
        excepted = torch.max(torch.abs(self.my_data_with_zero[self.golden_data_with_zero != 0] /
                                       self.golden_data_with_zero[self.golden_data_with_zero != 0] - 1)).item()
        self.assertAlmostEqual(error, excepted, places=6)

    def test_mean_relative_error(self):
        error, _ = mean_relative_error(self.golden_data, self.my_data)
        expected_error = torch.mean(torch.abs(self.my_data / self.golden_data - 1)).item()
        self.assertAlmostEqual(error, expected_error, places=6)

    def test_max_absolute_error(self):
        error, _ = max_absolute_error(self.golden_data, self.my_data)
        self.assertAlmostEqual(error, 0.1, places=6)

    def test_mean_absolute_error(self):
        error, _ = mean_absolute_error(self.golden_data, self.my_data)
        expected_error = torch.abs(self.my_data - self.golden_data).mean().item()
        self.assertAlmostEqual(error, expected_error, places=6)

    def test_kl_divergence(self):
        divergence, _ = kl_divergence(self.golden_data, self.my_data)
        expected_divergence = F.kl_div(
            F.log_softmax(self.my_data, dim=-1),
            F.softmax(self.golden_data, dim=-1),
            reduction="sum"
        ).item()
        self.assertAlmostEqual(divergence, expected_divergence, places=6)

    def test_relative_euclidean_distance(self):
        distance, _ = relative_euclidean_distance(self.golden_data, self.my_data)
        ground_truth_square_num = (self.golden_data ** 2).sum()
        expected_distance = ((self.my_data - self.golden_data) ** 2).sum() / ground_truth_square_num
        expected_distance = torch.sqrt(expected_distance).item()
        self.assertAlmostEqual(distance, expected_distance, places=6)


# Test cases for cosine_similarity
def test_cosine_similarity_given_both_zero_when_called_then_one():
    golden_data = torch.zeros([1])
    my_data = torch.zeros([1])
    result, _ = CMP_ALG_MAP["cosine_similarity"](golden_data, my_data)
    assert result == 1.0


def test_cosine_similarity_given_nonzero_when_called_then_within_range():
    golden_data = torch.tensor([1.0, 2.0, 3.0])
    my_data = torch.tensor([1.0, 2.0, 3.0])
    result, _ = CMP_ALG_MAP["cosine_similarity"](golden_data, my_data)
    assert 0 <= result <= 1


# Test cases for max_relative_error
def test_max_relative_error_given_valid_tensors_when_called_then_calculate_correctly():
    golden_data = torch.tensor([1.0, 2.0, 3.0])
    my_data = torch.tensor([2.0, 4.0, 6.0])
    result, _ = CMP_ALG_MAP["max_relative_error"](golden_data, my_data)
    assert result == pytest.approx(1.0)


def test_max_relative_error_given_zeros_when_called_then_zero():
    golden_data = torch.zeros([1])
    my_data = torch.zeros([1])
    result, _ = CMP_ALG_MAP["max_relative_error"](golden_data, my_data)
    assert result == 0


# Test cases for mean_relative_error
def test_mean_relative_error_given_valid_tensors_when_called_then_calculate_correctly():
    golden_data = torch.tensor([1.0, 2.0, 3.0])
    my_data = torch.tensor([2.0, 4.0, 6.0])
    result, _ = CMP_ALG_MAP["mean_relative_error"](golden_data, my_data)
    assert result == pytest.approx(1.0)


def test_mean_relative_error_given_zeros_when_called_then_zero():
    golden_data = torch.zeros([1])
    my_data = torch.zeros([1])
    result, _ = CMP_ALG_MAP["mean_relative_error"](golden_data, my_data)
    assert result == 0


# Test cases for max_absolute_error
def test_max_absolute_error_given_valid_tensors_when_called_then_calculate_correctly():
    golden_data = torch.tensor([1.0, 2.0, 3.0])
    my_data = torch.tensor([2.0, 4.0, 6.0])
    result, _ = CMP_ALG_MAP["max_absolute_error"](golden_data, my_data)
    assert result == pytest.approx(3.0)


def test_max_absolute_error_given_zeros_when_called_then_zero():
    golden_data = torch.zeros([1])
    my_data = torch.zeros([1])
    result, _ = CMP_ALG_MAP["max_absolute_error"](golden_data, my_data)
    assert result == 0


def test_mean_absolute_error_given_zeros_when_called_then_zero():
    golden_data = torch.zeros([1])
    my_data = torch.zeros([1])
    result, _ = CMP_ALG_MAP["mean_absolute_error"](golden_data, my_data)
    assert result == 0


# Test cases for kl_divergence
def test_kl_divergence_given_valid_tensors_when_called_then_calculate_positive():
    golden_data = torch.tensor([1.0, 2.0, 3.0])
    my_data = torch.tensor([2.0, 4.0, 6.0])
    result, _ = CMP_ALG_MAP["kl_divergence"](golden_data, my_data)
    assert result >= 0


def test_kl_divergence_given_identical_tensors_when_called_then_zero():
    golden_data = torch.tensor([1.0, 2.0, 3.0])
    my_data = torch.tensor([1.0, 2.0, 3.0])
    result, _ = CMP_ALG_MAP["kl_divergence"](golden_data, my_data)
    assert result == pytest.approx(0.0)


# Test cases for relative_euclidean_distance
def test_relative_euclidean_distance_given_valid_tensors_when_called_then_calculate_correctly():
    golden_data = torch.tensor([1.0, 2.0, 3.0])
    my_data = torch.tensor([2.0, 4.0, 6.0])
    result, _ = CMP_ALG_MAP["relative_euclidean_distance"](golden_data, my_data)
    assert result > 0


def test_relative_euclidean_distance_given_zeros_when_called_then_zero():
    golden_data = torch.zeros([1])
    my_data = torch.zeros([1])
    result, _ = CMP_ALG_MAP["relative_euclidean_distance"](golden_data, my_data)
    assert result == 0


# Test cases for register_custom_compare_algorithm
def test_register_custom_compare_algorithm_given_invalid_format_when_called_then_raise_value_error():
    with pytest.raises(ValueError):
        register_custom_compare_algorithm("invalid:format")


def test_register_custom_compare_algorithm_given_nonexistent_file_when_called_then_raise_value_error():
    with pytest.raises(ValueError):
        register_custom_compare_algorithm("/path/to/nonexistent.py:function_name")


def test_register_custom_compare_algorithm_given_not_py_file_when_called_then_raise_value_error():
    with pytest.raises(ValueError):
        register_custom_compare_algorithm("/path/to/file.txt:function_name")


def test_register_custom_compare_algorithm_given_illegal_permission_when_called_then_raise_value_error():
    with pytest.raises(ValueError):
        register_custom_compare_algorithm("/path/to/unreadable.py:function_name")


def test_register_custom_compare_algorithm_given_import_error_when_called_then_raise_value_error():
    with pytest.raises(ValueError):
        register_custom_compare_algorithm("/path/to/bad_module.py:function_name")


def test_register_custom_compare_algorithm_given_function_not_found_when_called_then_raise_value_error():
    with pytest.raises(ValueError):
        register_custom_compare_algorithm("/path/to/module.py:nonexistent_function")


def test_register_custom_compare_algorithm_given_incorrect_signature_when_called_then_raise_value_error():
    with pytest.raises(ValueError):
        register_custom_compare_algorithm("/path/to/module.py:function_with_incorrect_signature")


def test_register_custom_compare_algorithm_given_return_type_mismatch_when_called_then_raise_value_error():
    with pytest.raises(ValueError):
        register_custom_compare_algorithm("/path/to/module.py:function_with_return_type_mismatch")


# Test case to cover the logger.info call
@mock.patch('msprobe.infer.utils.cmp_algorithm.logger')
def test_register_custom_compare_algorithm_given_all_correct_when_called_then_log_added(mock_logger):
    os.makedirs("./resource", exist_ok=True)
    with open("./resource/valid_module.py", 'w') as f:
        f.write("""def valid_function(tensor1, tensor2):\n\treturn (0.0, '')""")
    try:
        register_custom_compare_algorithm("./resource/valid_module.py:valid_function")
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")
    finally:
        os.remove("./resource/valid_module.py")
    mock_logger.info.assert_called_once_with("Added custom comparing algorithm: valid_function")
