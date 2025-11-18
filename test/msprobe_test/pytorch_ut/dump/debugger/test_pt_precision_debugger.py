import unittest
from unittest.mock import MagicMock, patch

from msprobe.core.dump.common_config import CommonConfig
from msprobe.core.dump.debugger.precision_debugger import BasePrecisionDebugger
from msprobe.pytorch.dump.debugger.precision_debugger import PrecisionDebugger
from msprobe.pytorch.dump.pt_config import StatisticsConfig

from msprobe.core.common.const import Const
from msprobe.core.common.exceptions import MsprobeException, FileCheckException
from msprobe.core.common.utils import get_real_step_or_rank


class Args:
    def __init__(self, config_path=None, task=None, dump_path=None, level=None, model=None):
        self.config_path = config_path
        self.task = task
        self.dump_path = dump_path
        self.level = level
        self.model = model


class TestPrecisionDebugger(unittest.TestCase):
    json_config = {
        "task": "statistics",
        "dump_path": "/absolute_path",
        "rank": [],
        "step": [],
        "level": "L1",
        "async_dump": False
    }

    statistics_common_config = CommonConfig(json_config)
    statistics_task_config = StatisticsConfig(json_config)

    def test_init(self):
        step = get_real_step_or_rank([0, 1, "3-5"], Const.STEP)
        self.assertListEqual(step, [0, 1, 3, 4, 5])

    def test_check_input_params(self):
        args = Args(config_path=1)
        with self.assertRaises(MsprobeException) as context:
            PrecisionDebugger._check_input_params(args.config_path, args.task, args.dump_path, args.level)
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)

        args = Args(config_path="/")
        with self.assertRaises(FileCheckException) as context:
            PrecisionDebugger._check_input_params(args.config_path, args.task, args.dump_path, args.level)
        self.assertEqual(context.exception.code, FileCheckException.INVALID_FILE_ERROR)

        args = Args(task=1)
        with self.assertRaises(MsprobeException) as context:
            PrecisionDebugger._check_input_params(args.config_path, args.task, args.dump_path, args.level)
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)

        args = Args(dump_path=1)
        with self.assertRaises(MsprobeException) as context:
            PrecisionDebugger._check_input_params(args.config_path, args.task, args.dump_path, args.level)
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)

        args = Args(level=1)
        with self.assertRaises(MsprobeException) as context:
            PrecisionDebugger._check_input_params(args.config_path, args.task, args.dump_path, args.level)
        self.assertEqual(context.exception.code, MsprobeException.INVALID_PARAM_ERROR)

    def test_start_statistics(self):
        PrecisionDebugger._instance = None
        with patch.object(BasePrecisionDebugger, "_parse_config_path",
                          return_value=(self.statistics_common_config, self.statistics_task_config)):
            debugger = PrecisionDebugger(dump_path="./dump_path")
        debugger.service = MagicMock()
        debugger.config = MagicMock()
        debugger.task = 'statistics'
        debugger.start()
        debugger.service.start.assert_called_once()

    def test_stop_statistics(self):
        PrecisionDebugger._instance = None
        debugger = PrecisionDebugger(dump_path="./dump_path")
        debugger.service = MagicMock()
        debugger.task = ''
        debugger.stop()
        debugger.service.stop.assert_called_once()

    def test_step_statistics(self):
        debugger = PrecisionDebugger(dump_path="./dump_path")
        debugger.service = MagicMock()
        debugger.task = ''
        debugger.step()
        debugger.service.step.assert_called_once()
