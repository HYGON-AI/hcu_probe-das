# -*- coding: utf-8 -*-
# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
import subprocess

from msprobe.core.common.log import logger
from msprobe.infer.utils.security_check import is_enough_disk_space_left
from msprobe.infer.offline.compare.msquickcmp.adapter_cli.args_adapter import CmpArgsAdapter
from msprobe.infer.offline.compare.msquickcmp.cmp_process import cmp_process
from msprobe.infer.offline.compare.msquickcmp.common.args_check import check_model_path_legality, \
    check_target_model_path_legality, check_input_path_legality, check_cann_path_legality, check_output_path_legality, \
    check_dict_kind_string, check_rank_range_valid, check_number_list, check_dym_range_string, \
    check_fusion_cfg_path_legality, check_quant_json_path_legality, safe_string, str2bool, \
    check_input_json_path, check_input_data_path
from msprobe.infer.utils.util import filter_cmd
from msprobe.infer.offline.compare.msquickcmp.dump.dump_process import dump_process
from msprobe.infer.offline.compare.msquickcmp.dump.args_adapter import DumpArgsAdapter

CANN_PATH = os.environ.get('ASCEND_TOOLKIT_HOME', "/usr/local/Ascend/ascend-toolkit/latest")


def check_normal_dump_param(args):
    if args.opname:
        raise NotImplementedError("\'--operation-name\' or \'-opname\' only support "
                                  "MindIE-Torch dump scenario.")
    if args.exec:
        raise NotImplementedError("\'--exec\' only support MindIE-Torch dump scenario.")


def _offline_dump_parser(parser):
    parser.add_argument(
        '-m',
        '--model',
        required=False,
        dest="model_path",
        type=check_model_path_legality,
        help='The original model (.onnx or .pb or saved_model) file path')
    parser.add_argument(
        '--input_data',
        default='',
        dest="input_data_path",
        type=check_input_path_legality,
        help='The input data path of the model. Separate multiple inputs with commas(,).'
             ' E.g: input_0.bin,input_1.bin')
    parser.add_argument(
        '-c',
        '--cann_path',
        default=CANN_PATH,
        dest="cann_path",
        type=check_cann_path_legality,
        help='The CANN installation path')
    parser.add_argument(
        '-o',
        '--output',
        dest="out_path",
        default='',
        type=check_output_path_legality,
        help='The output path')
    parser.add_argument(
        '--input_shape',
        type=check_dict_kind_string,
        dest="input_shape",
        default='',
        help="Shape of input shape. Separate multiple nodes with semicolons(;)."
             " E.g: \"input_name1:1,224,224,3;input_name2:3,300\"")
    parser.add_argument(
        '--rank',
        type=check_rank_range_valid,
        dest="rank",
        default='0',
        help='Input rank ID [0, 255].')
    parser.add_argument(
        '-dr',
        '--dym_shape_range',
        type=check_dym_range_string,
        dest="dym_shape_range",
        default='',
        help="Dynamic shape range using in dynamic model, "
             "using this means ignore input_shape"
             " E.g: \"input_name1:1,3,200\~224,224-230;input_name2:1,300\"")
    parser.add_argument(
        '-ofs',
        '--onnx_fusion_switch',
        dest="onnx_fusion_switch",
        default=True,
        type=str2bool,
        help='Onnxruntime fusion switch, set False for dump complete onnx data when '
             'necessary.Usage: -ofs False')
    parser.add_argument(
        '--saved_model_signature',
        dest="saved_model_signature",
        default='serving_default',
        help="Enter the signature of the model")
    parser.add_argument(
        '--saved_model_tag_set',
        dest="saved_model_tag_set",
        default='serve',
        help="Enter the tagSet of the model.Currently, multiple tagSets can be transferred, "
             "for example, --saved_model_tag_set ['serve', 'general_parser']")
    parser.add_argument(
        '-dp',
        '--device_pattern',
        required=False,
        dest="device_pattern",
        choices=["cpu", "npu"],
        help="Enter inference in npu or cpu device. For example: -dp cpu")
    parser.add_argument(
        '--tf_json',
        required=False,
        dest="tf_json_path",
        type=check_input_json_path,
        help="When dump saved_model, you need provide tf-ops-json file path.")
    parser.add_argument(
        "--exec",
        dest="exec",
        required=False,
        type=safe_string,
        help="Exec command to run acltransformer model inference, "
             "only support MindIE-Torch dump scenario. "
             "For example: --exec \'bash run.sh patches/models/modeling_xxx.py\' ")
    parser.add_argument(
        "-opname",
        "--operation_name",
        required=False,
        dest="opname",
        type=safe_string,
        default=None,
        help="Operation names need to dump, only support MindIE-Torch dump scenario.")
    parser.add_argument(
        '--fusion_switch_file',
        dest="fusion_switch_file",
        type=check_fusion_cfg_path_legality,
        help='You can disable selected fusion patterns in the configuration file in TF2.x')


def offline_dump_cli(args):
    output_path = "./" if args.out_path == '' else args.out_path
    if not is_enough_disk_space_left(output_path):
        raise OSError("Please make sure that the remaining disk space in the dump path is greater than 2 GB")
    if args.exec:
        from msprobe.infer.offline.compare.msquickcmp.dump.mietorch.dump_config import DumpConfig
        DumpConfig(dump_path=args.out_path, api_list=args.opname)
        cmds = args.exec.split()
        cmds = filter_cmd(cmds)
        subprocess.run(cmds, shell=False)
    else:
        if (not args.model_path) or (not args.device_pattern):
            raise NotImplementedError("If you do not inference with MindIE-Torch, "
                                      "must use arguments '-m' and '-dp' to do next.")
        check_normal_dump_param(args)
        cmp_args = DumpArgsAdapter(args.model_path,
                                   input_data_path=args.input_data_path, cann_path=args.cann_path,
                                   out_path=args.out_path, input_shape=args.input_shape, device=args.device,
                                   dym_shape_range=args.dym_shape_range, onnx_fusion_switch=args.onnx_fusion_switch,
                                   saved_model_signature=args.saved_model_signature,
                                   saved_model_tag_set=args.saved_model_tag_set, device_pattern=args.device_pattern,
                                   tf_json_path=args.tf_json_path, fusion_switch_file=args.fusion_switch_file)
        dump_process(cmp_args, True)


def _compare_offline_model_parser(parser):
    parser.add_argument(
        '-tp',
        '--target_path',
        required=False,
        dest="target_path",
        type=check_target_model_path_legality,
        help='The offline model (.om or saved_model) file path')
    parser.add_argument(
        '-gp',
        '--golden_path',
        required=False,
        dest="golden_path",
        type=check_model_path_legality,
        help='The original model (.pb or saved_model or .onnx or .om) file path')
    parser.add_argument(
        '-o',
        '--output_path',
        dest="output_path",
        default='./',
        type=check_output_path_legality,
        help='The output path')
    parser.add_argument(
        '--input_data',
        default='',
        dest="input_data",
        type=check_input_data_path,
        help='The input data path of the model. Separate multiple inputs with commas(,).'
             ' E.g: input_0.bin,input_1.bin')
    parser.add_argument(
        '--input_shape',
        type=check_dict_kind_string,
        dest="input_shape",
        default='',
        help="Shape of input shape. Separate multiple nodes with semicolons(;)."
             " E.g: \"input_name1:1,224,224,3;input_name2:3,300\"")
    parser.add_argument(
        '--rank',
        type=check_rank_range_valid,
        dest="rank",
        default='0',
        help='Input device ID [0, 255].')
    parser.add_argument(
        '--output_size',
        type=check_number_list,
        dest="output_size",
        default='',
        help='The size of output. Separate multiple sizes with commas(,). E.g: 10200,34000')
    parser.add_argument(
        '--output_nodes',
        type=check_dict_kind_string,
        dest="output_nodes",
        default='',
        help="Output nodes designated by user. Separate multiple nodes with semicolons(;)."
             " E.g: \"node_name1:0;node_name2:1;node_name3:0\"")
    parser.add_argument(
        '--dym_shape_range',
        type=check_dym_range_string,
        dest="dym_shape_range",
        default='',
        help="Dynamic shape range using in dynamic model, "
             "using this means ignore input_shape"
             " E.g: \"input_name1:1,3,200\~224,224-230;input_name2:1,300\"")
    parser.add_argument(
        '-ofs',
        '--onnx_fusion_switch',
        dest="onnx_fusion_switch",
        default=True,
        type=str2bool,
        help='Onnxruntime fusion switch, set False for dump complete onnx data when '
             'necessary.Usage: -ofs False')
    parser.add_argument(
        '-qfr',
        '--quant_fusion_rule_file',
        type=check_quant_json_path_legality,
        dest="quant_fusion_rule_file",
        default='',
        help="the quant fusion rule file path")
    parser.add_argument(
        '--saved_model_signature',
        dest="saved_model_signature",
        default='serving_default',
        help="Enter the signature of the model")
    parser.add_argument(
        '--saved_model_tag_set',
        dest="saved_model_tag_set",
        default='serve',
        help="Enter the tagSet of the model.Currently, multiple tagSets can be transferred, "
             "for example, --saved_model_tag_set ['serve', 'general_parser']")


offline_model_compare_allowed_keys = {
    'mode',
    'target_path', 'golden_path', 'output_path', 'input_data', 'input_shape', 'rank',
    'output_size', 'output_nodes', 'dym_shape_range', 'onnx_fusion_switch', 'quant_fusion_rule_file',
    'saved_model_signature', 'saved_model_tag_set'
}


def compare_offline_model_mode(args):
    args = set_args_default(args)
    check_compare_args(args)

    if not args.golden_path:
        logger.error("The following args are required: -gp/--golden_path")
        return
    cmp_args = CmpArgsAdapter(args.golden_path, args.target_path, args.input_data, CANN_PATH, args.output_path,
                              args.input_shape, args.rank, args.output_size, args.output_nodes, args.dym_shape_range,
                              args.onnx_fusion_switch, args.quant_fusion_rule_file, args.saved_model_signature,
                              args.saved_model_tag_set)
    cmp_process(cmp_args, True)


def set_args_default(args):
    if not hasattr(args, 'rank') or not args.rank:
        args.rank = '0'
    if not hasattr(args, 'input_data'):
        args.input_data = ''
    if not hasattr(args, 'input_shape'):
        args.input_shape = ''
    if not hasattr(args, 'output_size'):
        args.output_size = ''
    if not hasattr(args, 'output_nodes'):
        args.output_nodes = ''
    if not hasattr(args, 'dym_shape_range'):
        args.dym_shape_range = ''
    if not hasattr(args, 'onnx_fusion_switch'):
        args.onnx_fusion_switch = True
    if not hasattr(args, 'quant_fusion_rule_file'):
        args.quant_fusion_rule_file = ''
    if not hasattr(args, 'saved_model_signature'):
        args.saved_model_signature = 'serving_default'
    if not hasattr(args, 'saved_model_tag_set'):
        args.saved_model_tag_set = 'serve'
    return args


def check_compare_args(args):
    if args.target_path:
        args.target_path = check_target_model_path_legality(args.target_path)
    if args.golden_path:
        args.golden_path = check_model_path_legality(args.golden_path)
    if args.output_path:
        args.output_path = check_output_path_legality(args.output_path)
    if args.input_data:
        args.input_data = check_input_data_path(args.input_data)
    if args.input_shape:
        args.input_shape = check_dict_kind_string(args.input_shape)
    if args.rank:
        args.rank = check_rank_range_valid(args.rank)
    if args.output_size:
        args.output_size = check_number_list(args.output_size)
    if args.output_nodes:
        args.output_nodes = check_dict_kind_string(args.output_nodes)
    if args.dym_shape_range:
        args.dym_shape_range = check_dym_range_string(args.dym_shape_range)
    if args.onnx_fusion_switch:
        args.onnx_fusion_switch = str2bool(args.onnx_fusion_switch)
    if args.quant_fusion_rule_file:
        args.quant_fusion_rule_file = check_quant_json_path_legality(args.quant_fusion_rule_file)


def _install_offline_deps_parser(parser):
    parser.add_argument("--no_check", dest="no_check", action="store_true",
                        help="<optional> Whether to skip checking the target website's certificate information "
                             "when installing aclruntime and ais_bench poses a certain security risk. "
                             "This poses a certain security risk, "
                             "and users should use it with caution and bear the consequences themselves.",
                        required=False)


def install_offline_deps_cli(args):
    offline_extra_install_cmd = [
        "/bin/bash",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "install_aclruntime_aisbench.sh")),
        str(args.no_check)
    ]
    subprocess.run(offline_extra_install_cmd, shell=False)
