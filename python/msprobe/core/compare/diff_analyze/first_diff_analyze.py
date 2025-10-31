# Copyright (c) 2025-2025, Huawei Technologies Co., Ltd.
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

import os

from tqdm import tqdm

from msprobe.core.common.const import Const, CompareConst
from msprobe.core.common.utils import logger, CompareException
from msprobe.core.common.file_utils import load_yaml
from msprobe.core.compare.config import ModeConfig
from msprobe.core.compare.utils import gen_api_batches


cur_dir = os.path.dirname(os.path.realpath(__file__))
diff_threshold_yaml_path = os.path.join(cur_dir, 'diff_analyze_threshold.yaml')
ignore_op_list_yaml_path = os.path.join(cur_dir, 'ignore_op_list.yaml')
ignore_list = load_yaml(ignore_op_list_yaml_path)
thresholds = load_yaml(diff_threshold_yaml_path)
cmp_metrics = thresholds.get('compare_metrics')


class FirstDiffAnalyze:
    def __init__(self, mode_config: ModeConfig, rank):
        self.mode_config = mode_config
        self.rank = rank

    @staticmethod
    def single_metric_diff_check(cmp_metric, metric_value):
        threshold = thresholds.get(cmp_metric, None)
        if threshold is None:
            logger.error(f"Check diff or {cmp_metric} need to configure the threshold. "
                         f"Please configure it in 'diff_analyze_threshold.yaml'.")
            raise CompareException(CompareException.MISSING_THRESHOLD_ERROR)
        if not isinstance(threshold, list) or len(threshold) != 1:
            logger.error(f"{cmp_metric} threshold configure wrong. Please check.")
            raise CompareException(CompareException.WRONG_THRESHOLD_ERROR)
        if isinstance(metric_value, str) and metric_value.endswith('%'):
            metric_value_float = float(metric_value[:-1]) / 100
            if metric_value_float > threshold[0]:
                return True
        return False

    def single_api_check(self, result_slice, header, api_name=None):
        """
        单个api差异检查

        :param result_slice: 数据切片
        :param header: 列名列表
        :return: {'is_same': bool, 'op_items': list[dict]}
        """
        single_check_result = {
            'is_same': True,
            'op_items': []
        }

        column_indices = {name: idx for idx, name in enumerate(header)}
        output_idx = -1
        for line in result_slice:
            op_item = {
                column_name: line[column_indices[column_name]]
                for column_name in header
            }
            single_check_result['op_items'].append(op_item)
            if op_item['state'] != 'output':
                continue
            output_idx += 1
            if output_idx in ignore_list.get(api_name, []):
                continue
            # set is_same
            if self.mode_config.dump_mode == Const.MD5:
                if line[column_indices[CompareConst.RESULT]] == CompareConst.DIFF:
                    single_check_result['is_same'] = False
            else:
                for cmp_metric in cmp_metrics:
                    metric_value = line[column_indices[cmp_metric]]
                    if self.single_metric_diff_check(cmp_metric, metric_value):
                        single_check_result['is_same'] = False
                        break
        return single_check_result

    def check(self, result_df):
        """
        比对后循环遍历api检查差异
        example：
        {
            'Functional.conv2d.0.forward': {
                'is_same': true,
                'op_items': [
                    {
                        'NPU name': 'Functional.conv2d.0.forward.input.0',
                        'Bench name': 'Functional.conv2d.0.forward.input.0',
                        'xxx': 1,
                        'NormRelativeErr': 2,
                        'yyy': 3,
                        ...
                    }
                ]
            }
        }
        """
        result = result_df.values
        header = result_df.columns.tolist()

        api_batches = gen_api_batches(result, header)

        check_result = {}

        default_bar_desc = 'API/Module diff check Progress'
        bar_desc_add_rank = f'[{self.rank}]' + default_bar_desc if self.rank else default_bar_desc
        with tqdm(total=len(api_batches), desc=bar_desc_add_rank, unit="api/module", ncols=100) as progress_bar:
            for api_batch in api_batches:
                result_slice = result[api_batch.start: api_batch.params_grad_end_index]
                api_compo = api_batch.api_name.split('.')
                # suppose name is Tensor.MatMul.0.forward
                if len(api_compo) < 4:
                    continue
                # get MatMul as api_name
                api_name = api_compo[-3]
                check_result[api_batch.api_name] = self.single_api_check(result_slice, header, api_name)
                progress_bar.update(1)

        return check_result
