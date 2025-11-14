import os
import sys
import pytest
from unittest.mock import patch, MagicMock

from msprobe.pytorch.torchair_dump.torchair_dump import try_import_torchair, set_ge_dump_config, set_fx_dump_config


def test_try_import_torchair_given_missing_torch_when_called_then_error():
    with patch.dict(sys.modules, {'torch': None}):
        with pytest.raises(ModuleNotFoundError):
            try_import_torchair()


@patch('os.path.exists', return_value=True)
@patch('time.strftime', return_value='20250414_120000')
def test_set_ge_dump_config_given_valid_params_when_called_then_success(
    mock_time, mock_exists
):
    mock_torchair = MagicMock()
    mock_torchair.configs.compiler_config = MagicMock()
    mock_torch_npu = MagicMock()
    with patch.dict('sys.modules', {
             'torchair': mock_torchair,
             'torchair.configs.compiler_config': mock_torchair.configs.compiler_config,
             'torch_npu': mock_torch_npu,
         }):
        dump_path = os.path.dirname(os.path.realpath(__file__))
        config = set_ge_dump_config(
            dump_path=dump_path,
            fusion_switch_file="fusion_switch.cfg",
            dump_token=[1, 2],
            dump_layer=["conv"]
        )

        assert config.dump_config.enable_dump
        assert config.fusion_config.fusion_switch_file == "fusion_switch.cfg"
        assert config.dump_config.dump_step == "1|2"
        assert config.dump_config.dump_layer == "conv"


def test_set_fx_dump_config_given_default_when_called_then_npy_type():
    mock_torchair = MagicMock()
    mock_torchair.configs.compiler_config = MagicMock()
    mock_torch_npu = MagicMock()
    with patch.dict('sys.modules', {
             'torchair': mock_torchair,
             'torchair.configs.compiler_config': mock_torchair.configs.compiler_config,
             'torch_npu': mock_torch_npu,
         }):
        config = set_fx_dump_config()
        assert config.debug.data_dump.type == "npy"
