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

from unittest.mock import MagicMock
import pytest

from msprobe.infer.utils.check.func_wrapper import FuncWrapper, validate_params


@pytest.fixture(scope="module", autouse=True)
def setup():
    # Setup any mock objects or state needed for tests
    pass


class TestFuncWrapper:

    @staticmethod
    def test_is_function_param_valid_given_no_rules_when_called_then_true():
        func_wrapper = FuncWrapper({})
        result = func_wrapper.is_function_param_valid(False)
        assert result is True

    @staticmethod
    def test_is_function_param_valid_given_invalid_kwarg_when_called_then_false_with_message():
        check_rule_mock = MagicMock()
        check_rule_mock.check.return_value = False
        func_wrapper = FuncWrapper({'test_param': check_rule_mock})
        result = func_wrapper.is_function_param_valid(False, test_param='invalid')
        assert isinstance(result, tuple)
        assert result[0] is False
        assert 'test_param is invalid.' in result[1]

    @staticmethod
    def test_is_function_param_valid_given_valid_kwarg_when_called_then_true():
        check_rule_mock = MagicMock()
        check_rule_mock.check.return_value = True
        func_wrapper = FuncWrapper({'test_param': check_rule_mock})
        result = func_wrapper.is_function_param_valid(False, test_param='valid')
        assert result is True

    @staticmethod
    def test_is_function_param_valid_given_invalid_var_keyword_when_called_then_false_with_message():
        check_rule_mock = MagicMock()
        check_rule_mock.check.return_value = False
        func_wrapper = FuncWrapper({})
        func_wrapper.var_keyword_rule = check_rule_mock
        result = func_wrapper.is_function_param_valid(False, unexpected_param='invalid')
        assert isinstance(result, tuple)
        assert result[0] is False
        assert 'unexpected_param is invalid.' in result[1]

    @staticmethod
    def test_is_function_param_valid_given_valid_var_keyword_when_called_then_true():
        check_rule_mock = MagicMock()
        check_rule_mock.check.return_value = True
        func_wrapper = FuncWrapper({})
        func_wrapper.var_keyword_rule = check_rule_mock
        result = func_wrapper.is_function_param_valid(False, unexpected_param='valid')
        assert result is True

    @staticmethod
    def test_is_function_param_valid_given_invalid_args_when_called_then_false_with_message():
        check_rule_mock = MagicMock()
        check_rule_mock.check.return_value = False
        func_wrapper = FuncWrapper({'param1': check_rule_mock})
        func_wrapper.args_names = ['param1']
        result = func_wrapper.is_function_param_valid(False, 'invalid')
        assert isinstance(result, tuple)
        assert result[0] is False
        assert 'param1 is invalid.' in result[1]

    @staticmethod
    def test_is_function_param_valid_given_valid_args_when_called_then_true():
        check_rule_mock = MagicMock()
        check_rule_mock.check.return_value = True
        func_wrapper = FuncWrapper({'param1': check_rule_mock})
        func_wrapper.args_names = ['param1']
        result = func_wrapper.is_function_param_valid(False, 'valid')
        assert result is True

    @staticmethod
    def test_parse_function_given_function_with_params_when_called_then_parses_correctly():
        def test_func(param1, param2=None, **kwargs): pass
        func_wrapper = FuncWrapper({})
        func_wrapper.parse_function(test_func)
        assert func_wrapper.args_names == ['param1', 'param2']
        assert 'param1' in func_wrapper.args_name_set
        assert 'param2' in func_wrapper.args_name_set

    @staticmethod
    def test_parse_function_given_function_with_var_keyword_when_called_then_sets_var_keyword_rule():
        def test_func(**kwargs): pass
        func_wrapper = FuncWrapper({'kwargs': MagicMock()})
        func_wrapper.parse_function(test_func)
        assert func_wrapper.var_keyword_rule is not None

    @staticmethod
    def test_create_wrapper_given_ret_value_and_to_raise_when_called_then_returns_decorator():
        func_wrapper = FuncWrapper({})
        decorator = func_wrapper.create_wrapper('ret_value', True)
        assert callable(decorator)

    @staticmethod
    def test_create_wrapper_given_function_when_wrapped_then_validates_params():
        check_rule_mock = MagicMock()
        check_rule_mock.check.return_value = True
        func_wrapper = FuncWrapper({'param1': check_rule_mock})

        @func_wrapper.create_wrapper('default_return', False)
        def test_func(param1): return 'original_return'

        result = test_func('valid')
        assert result == 'original_return'

    @staticmethod
    def test_create_wrapper_given_invalid_params_when_wrapped_then_returns_ret_value():
        check_rule_mock = MagicMock()
        check_rule_mock.check.return_value = False
        logger_mock = MagicMock()
        func_wrapper = FuncWrapper({'param1': check_rule_mock})

        @func_wrapper.create_wrapper('default_return', False, logger_mock)
        def test_func(param1): return 'original_return'

        result = test_func('invalid')
        logger_mock.error.assert_called_once()
        assert result == 'default_return'

    @staticmethod
    def test_to_return_given_ret_value_when_called_then_creates_wrapper():
        func_wrapper = FuncWrapper({})
        wrapper = func_wrapper.to_return('ret_value')
        assert callable(wrapper)

    @staticmethod
    def test_to_raise_given_called_when_called_then_creates_wrapper():
        func_wrapper = FuncWrapper({})
        wrapper = func_wrapper.to_raise()
        assert callable(wrapper)

    @staticmethod
    def test_validate_params_given_check_rules_when_called_then_returns_func_wrapper():
        check_rules = {'param1': MagicMock()}
        result = validate_params(**check_rules)
        assert isinstance(result, FuncWrapper)
        assert result.check_rules == check_rules