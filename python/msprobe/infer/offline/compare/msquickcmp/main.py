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

from msprobe.infer.utils.parser import BaseCommand
from msprobe.infer.utils.security_check import is_enough_disk_space_left
from msprobe.infer.offline.compare.msquickcmp.adapter_cli.args_adapter import CmpArgsAdapter
from msprobe.infer.offline.compare.msquickcmp.cmp_process import cmp_process
from msprobe.infer.offline.compare.msquickcmp.common.args_check import check_model_path_legality, \
    check_om_path_legality, check_input_path_legality, check_cann_path_legality, check_output_path_legality, \
    check_dict_kind_string, check_device_range_valid, check_number_list, check_dym_range_string, \
    check_fusion_cfg_path_legality, check_quant_json_path_legality, safe_string, str2bool, \
    check_alone_compare_dir_path, check_input_json_path, check_debug_compare_input_data_path
from msprobe.infer.offline.compare.msquickcmp.common.utils import logger
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


class CompareCommand(BaseCommand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parser = None

    def add_arguments(self, parser):
        parser.add_argument(
            '-gm',
            '--golden-model',
            required=False,
            dest="golden_model",
            type=check_model_path_legality,
            help='The original model (.pb or saved_model or .onnx or .om) file path')
        parser.add_argument(
            '-om',
            '--om-model',
            required=False,
            dest="om_model",
            type=check_om_path_legality,
            help='The offline model (.om or saved_model) file path')
        parser.add_argument(
            '-i',
            '--input',
            default='',
            dest="input_data_path",
            type=check_debug_compare_input_data_path,
            help='The input data path of the model. Separate multiple inputs with commas(,).'
                 ' E.g: input_0.bin,input_1.bin')
        parser.add_argument(
            '-c',
            '--cann-path',
            default=CANN_PATH,
            dest="cann_path",
            type=check_cann_path_legality,
            help='The CANN installation path')
        parser.add_argument(
            '-o',
            '--output',
            dest="out_path",
            default='./',
            type=check_output_path_legality,
            help='The output path')
        parser.add_argument(
            '-is',
            '--input-shape',
            type=check_dict_kind_string,
            dest="input_shape",
            default='',
            help="Shape of input shape. Separate multiple nodes with semicolons(;)."
                 " E.g: \"input_name1:1,224,224,3;input_name2:3,300\"")
        parser.add_argument(
            '-d',
            '--device',
            type=check_device_range_valid,
            dest="device",
            default='0',
            help='Input device ID [0, 255].')
        parser.add_argument(
            '-outsize',
            '--output-size',
            type=check_number_list,
            dest="output_size",
            default='',
            help='The size of output. Separate multiple sizes with commas(,). E.g: 10200,34000')
        parser.add_argument(
            '-n',
            '--output-nodes',
            type=check_dict_kind_string,
            dest="output_nodes",
            default='',
            help="Output nodes designated by user. Separate multiple nodes with semicolons(;)."
                 " E.g: \"node_name1:0;node_name2:1;node_name3:0\"")
        parser.add_argument(
            '--advisor',
            action='store_true',
            dest="advisor",
            help='Enable advisor after compare.')
        parser.add_argument(
            '-dr',
            '--dym-shape-range',
            type=check_dym_range_string,
            dest="dym_shape_range",
            default='',
            help="Dynamic shape range using in dynamic model, "
                 "using this means ignore input_shape"
                 " E.g: \"input_name1:1,3,200\~224,224-230;input_name2:1,300\"")
        parser.add_argument(
            '--dump',
            dest="dump",
            default=True,
            type=str2bool,
            help="Whether to dump all the operations' output.")
        parser.add_argument(
            '--convert',
            dest="bin2npy",
            default=False,
            type=str2bool,
            help='Enable npu dump data conversion from bin to npy after compare.Usage: --convert True')
        parser.add_argument(
            '--locat',
            default=False,
            dest="locat",
            type=str2bool,
            help='Enable accuracy interval location when needed.E.g: --locat True')
        parser.add_argument(
            '-cp',
            '--custom-op',
            type=safe_string,
            dest="custom_op",
            default='',
            help='Op name witch is not registered in onnxruntime, only supported by Ascend')
        parser.add_argument(
            '-ofs',
            '--onnx-fusion-switch',
            dest="onnx_fusion_switch",
            default=True,
            type=str2bool,
            help='Onnxruntime fusion switch, set False for dump complete onnx data when '
                 'necessary.Usage: -ofs False')
        parser.add_argument(
            '--fusion-switch-file',
            dest="fusion_switch_file",
            type=check_fusion_cfg_path_legality,
            help='You can disable selected fusion patterns in the configuration file')
        parser.add_argument(
            "-single",
            "--single-op",
            default=False,
            dest="single_op",
            type=str2bool,
            help='Comparison mode:single operator compare.Usage: -single True')
        parser.add_argument(
            "-max",
            "--max-cmp-size",
            dest="max_cmp_size",
            default=0,
            type=int,
            help="Max size of tensor array to compare. Usage: --max-cmp-size 1024")
        parser.add_argument(
            '-q',
            '--quant-fusion-rule-file',
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
        # alone compare parameters
        parser.add_argument(
            '-mp',
            '--my-path',
            required=False,
            dest="my_path",
            type=check_alone_compare_dir_path,
            help='The npu dump data path')
        parser.add_argument(
            '-gp',
            '--golden-path',
            required=False,
            dest="golden_path",
            type=check_alone_compare_dir_path,
            help='The cpu(golden) dump data path')
        parser.add_argument(
            '--ops-json',
            required=False,
            dest="ops_json",
            type=check_alone_compare_dir_path,
            help='The npu and cpu ops matching rule json')
        self.parser = parser

    def handle(self, args):
        if not args.golden_model and not args.golden_path:
            logger.error("The following args are required: -gm/--golden-model or -gp/--golden-path")
            self.parser.print_help()
            return
        if args.ops_json is not None:
            mindie_rt_op_mapping = os.path.join(args.ops_json, "mindie_rt_op_mapping.json")
            mindie_torch_op_mapping = os.path.join(args.ops_json, "mindie_torch_op_mapping.json")
            if os.path.exists(mindie_rt_op_mapping) and os.path.exists(mindie_torch_op_mapping):
                from msprobe.infer.offline.compare.msquickcmp.mie_torch.mietorch_comp import MIETorchCompare
                args.golden_path = check_input_path_legality(args.golden_path)
                args.my_path = check_input_path_legality(args.my_path)
                args.ops_json = check_input_path_legality(args.ops_json)
                comparer = MIETorchCompare(args.golden_path, args.my_path, args.ops_json, args.out_path)
                comparer.compare()
                return
        cmp_args = CmpArgsAdapter(args.golden_model, args.om_model, args.weight_path, args.input_data_path,
                                  args.cann_path, args.out_path,
                                  args.input_shape, args.device, args.output_size, args.output_nodes, args.advisor,
                                  args.dym_shape_range,
                                  args.dump, args.bin2npy, args.custom_op, args.locat,
                                  args.onnx_fusion_switch, args.single_op, args.fusion_switch_file,
                                  args.max_cmp_size, args.quant_fusion_rule_file, args.saved_model_signature,
                                  args.saved_model_tag_set, args.my_path, args.golden_path, args.ops_json)
        cmp_process(cmp_args, True)


def get_compare_cmd_ins():
    help_info = "one-click network-wide accuracy analysis of golden models."
    compare_instance = CompareCommand("compare", help_info)
    return compare_instance


def _offline_dump_parser(parser):
    parser.add_argument(
        '-m',
        '--model',
        required=False,
        dest="model_path",
        type=check_model_path_legality,
        help='The original model (.onnx or .pb or saved_model) file path')
    parser.add_argument(
        '-i',
        '--input',
        default='',
        dest="input_data_path",
        type=check_input_path_legality,
        help='The input data path of the model. Separate multiple inputs with commas(,).'
             ' E.g: input_0.bin,input_1.bin')
    parser.add_argument(
        '-c',
        '--cann-path',
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
        '-is',
        '--input-shape',
        type=check_dict_kind_string,
        dest="input_shape",
        default='',
        help="Shape of input shape. Separate multiple nodes with semicolons(;)."
             " E.g: \"input_name1:1,224,224,3;input_name2:3,300\"")
    parser.add_argument(
        '-d',
        '--device',
        type=check_device_range_valid,
        dest="device",
        default='0',
        help='Input device ID [0, 255].')
    parser.add_argument(
        '-dr',
        '--dym-shape-range',
        type=check_dym_range_string,
        dest="dym_shape_range",
        default='',
        help="Dynamic shape range using in dynamic model, "
             "using this means ignore input_shape"
             " E.g: \"input_name1:1,3,200\~224,224-230;input_name2:1,300\"")
    parser.add_argument(
        '-ofs',
        '--onnx-fusion-switch',
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
        '--device-pattern',
        required=False,
        dest="device_pattern",
        choices=["cpu", "npu"],
        help="Enter inference in npu or cpu device. For example: -dp cpu")
    parser.add_argument(
        '--tf-json',
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
        "--operation-name",
        required=False,
        dest="opname",
        type=safe_string,
        default=None,
        help="Operation names need to dump, only support MindIE-Torch dump scenario.")
    parser.add_argument(
        '--fusion-switch-file',
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
