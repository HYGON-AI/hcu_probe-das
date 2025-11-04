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

"""
Function:
This class mainly involves the main function.
"""

import csv
import logging
import os
import shutil
import site
import stat
import subprocess
import time

import onnxruntime
import pandas as pd
from msprobe.infer.offline.surgeon.auto_optimizer.graph_refactor.onnx.graph import OnnxGraph
from msprobe.infer.offline.surgeon.auto_optimizer.graph_refactor.interface.base_node import Node
from msprobe.infer.offline.compare.msquickcmp.accuracy_locat import accuracy_locat as al
from msprobe.infer.offline.compare.msquickcmp.adapter_cli.args_adapter import CmpArgsAdapter
from msprobe.infer.offline.compare.msquickcmp.atc import atc_utils
from msprobe.infer.offline.compare.msquickcmp.common import utils
from msprobe.infer.offline.compare.msquickcmp.common.args_check import is_saved_model_valid
from msprobe.infer.offline.compare.msquickcmp.common.convert import convert_bin_dump_data_to_npy
from msprobe.infer.offline.compare.msquickcmp.common.convert import convert_npy_to_bin
from msprobe.infer.offline.compare.msquickcmp.common.utils import AccuracyCompareException, \
    get_shape_to_directory_name, safe_delete_path_if_exists, OPTYPE_WHITWLIST, find_om_files
from msprobe.infer.offline.compare.msquickcmp.net_compare import analyser
from msprobe.infer.offline.compare.msquickcmp.net_compare.net_compare import NetCompare
from msprobe.infer.offline.compare.msquickcmp.npu.npu_dump_data import NpuDumpData, DynamicInput
from msprobe.infer.offline.compare.msquickcmp.npu.om_parser import OmParser
from msprobe.infer.offline.compare.msquickcmp.single_op import single_op as sp

from msprobe.infer.utils.security_check import check_write_directory, ms_makedirs
from msprobe.infer.utils.file_open_check import ms_open, sanitize_csv_value
from msprobe.infer.utils.check.rule import Rule
from msprobe.infer.utils.util import load_file_to_read_common_check, filter_cmd
from msprobe.infer.utils.constants import TENSOR_MAX_SIZE

try:
    import acl
except ImportError:
    acl = None
    utils.logger.error("Please verify that the CANN environment is properly configured.")

WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR
READ_WRITE_FLAGS = os.O_RDWR | os.O_CREAT
WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
ERROR_INTERVAL_INFO_FILE = "error_interval_info.txt"
MAX_MEMORY_USE = 6 * 1024 * 1024 * 1024
COSINE_SIMILARITY = 0.99
RELATIVE_EUCLIDEAN_DISTANCE = 0.05
KULLBACK_LEIBLER_DIVERGENCE = 0.005
ROOT_MEAN_SQUARE_ERROR = 1.0
MEAN_RELATIVE_ERROR = 1.0
YES = "YES"
NO = "NO"


def _generate_golden_data_model(args, npu_dump_npy_path):
    if is_saved_model_valid(args.model_path):
        from msprobe.infer.offline.compare.msquickcmp.tf.tf_save_model_dump_data import TfSaveModelDumpData

        return TfSaveModelDumpData(args, args.model_path), None
    model_name, extension = utils.get_model_name_and_extension(args.model_path)
    if ".pb" == extension:
        from msprobe.infer.offline.compare.msquickcmp.tf.tf_dump_data import TfDumpData

        return TfDumpData(args), extension
    elif ".onnx" == extension:
        from msprobe.infer.offline.compare.msquickcmp.onnx_model.onnx_dump_data import OnnxDumpData

        return OnnxDumpData(args, npu_dump_npy_path), extension
    elif ".om" == extension:
        return NpuDumpData(arguments=args, is_golden=True), extension

    else:
        utils.logger.error("Only model files whose names end with .pb or .onnx are supported")
        raise AccuracyCompareException(utils.ACCURACY_COMPARISON_MODEL_TYPE_ERROR)


def _correct_the_wrong_order(left_index, right_index, golden_net_output_info):
    if left_index not in golden_net_output_info.keys() or right_index not in golden_net_output_info.keys():
        return
    if left_index != right_index:
        tmp = golden_net_output_info[left_index]
        golden_net_output_info[left_index] = golden_net_output_info[right_index]
        golden_net_output_info[right_index] = tmp
        utils.logger.info('swap the %s and %s item in golden_net_output_info!', left_index, right_index)


def _check_output_node_name_mapping(original_net_output_node, golden_net_output_info):
    for left_index, node_name in original_net_output_node.items():
        match = False
        for right_index, dump_file_path in golden_net_output_info.items():
            dump_file_name = os.path.basename(dump_file_path)
            if dump_file_name.startswith(node_name.replace("/", "_").replace(":", ".")):
                match = True
                _correct_the_wrong_order(left_index, right_index, golden_net_output_info)
                break
        if not match:
            utils.logger.warning("the original name: {} of net output maybe not correct!".format(node_name))
            break


def _get_single_csv_in_folder(csv_path):
    for file_name in os.listdir(csv_path):
        if file_name.endswith('.csv'):
            return os.path.join(csv_path, file_name)
    raise IOError(f"None csv file exists in folder {csv_path}")


def _read_and_process_csv(csv_path, process_func, node_output_show_list):
    if Rule.input_file().check(csv_path):
        with ms_open(csv_path, 'r', max_size=TENSOR_MAX_SIZE) as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
        if len(rows) < 1:
            utils.logger.error("csv is empty, please check.")
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_EMPTY_CSV_ERROR)
        header = rows[0]
        rows = process_func(header, rows, node_output_show_list)
    return rows


def _write_csv(csv_path, rows):
    with ms_open(csv_path, mode='w') as f:
        writer = csv.writer(f)
        for line in rows:
            for ele in line:
                _ = sanitize_csv_value(ele)
        writer.writerows(rows)


def _process_is_npu_and_is_precision_error_ops(header, rows, node_output_name_list):
    ground_truth_col = header.index("GroundTruth")
    optype_col = header.index("OpType")
    cosine_similarity_col = header.index("CosineSimilarity")
    relative_euclidean_distance_col = header.index("RelativeEuclideanDistance")
    kullback_leibler_divergence_col = header.index("KullbackLeiblerDivergence")
    root_mean_square_error_col = header.index("RootMeanSquareError")
    mean_relative_error_col = header.index("MeanRelativeError")

    header.append('IsNpuOps')
    header.append('IsOutputNode')
    header.append('IsPrecisionError')
    for row in rows[1:]:
        try:
            is_npu_ops = YES if row[ground_truth_col] == "*" else NO
            row.append(is_npu_ops)

            optype = row[optype_col]
            cosine_similarity = row[cosine_similarity_col]
            relative_euclidean_distance = float(row[relative_euclidean_distance_col])
            kullback_leibler_divergence = float(row[kullback_leibler_divergence_col])
            root_mean_square_error = float(row[root_mean_square_error_col])
            mean_relative_error = float(row[mean_relative_error_col])
            if _is_output_node(row[ground_truth_col], node_output_name_list):
                row.append(YES)
            else:
                row.append(NO)
            if optype in OPTYPE_WHITWLIST or cosine_similarity.lower() == 'nan':
                row.append(NO)
                continue
            if _is_row_precision_error(cosine_similarity,
                                       relative_euclidean_distance,
                                       kullback_leibler_divergence,
                                       root_mean_square_error,
                                       mean_relative_error):
                row.append(YES)
            else:
                row.append(NO)

        except ValueError as e:
            utils.logger.warning(f"Skipping row due to invalid data: {row}. Error: {e}")
            continue
    return rows


def _is_row_precision_error(cosine_similarity,
                            relative_euclidean_distance,
                            kullback_leibler_divergence,
                            root_mean_square_error,
                            mean_relative_error):
    return (float(cosine_similarity) < COSINE_SIMILARITY or
            relative_euclidean_distance > RELATIVE_EUCLIDEAN_DISTANCE or
            kullback_leibler_divergence > KULLBACK_LEIBLER_DIVERGENCE or
            root_mean_square_error > ROOT_MEAN_SQUARE_ERROR or
            mean_relative_error > MEAN_RELATIVE_ERROR)


def _is_output_node(groundtruth, onnxnode_output_name_list):
    return groundtruth.strip() in onnxnode_output_name_list


def _append_column_to_csv(csv_path, node_output_show_list=None):
    if node_output_show_list is None:
        node_output_show_list = []
    csv_path = _get_single_csv_in_folder(csv_path)
    rows = _read_and_process_csv(csv_path, _process_is_npu_and_is_precision_error_ops, node_output_show_list)
    _write_csv(csv_path, rows)


def check_dump_and_compare(args: CmpArgsAdapter):
    if not args.my_path and not args.golden_path and not args.ops_json:
        return True
    else:
        if args.my_path and args.golden_path and args.ops_json:
            return False
        else:
            raise Exception("If you want to alone compare, you need to provide parameters like: "
                            "-mp -gp and --ops-json")


def cmp_process(args: CmpArgsAdapter, use_cli: bool):
    """
    Function Description:
        main process function
    Exception Description:
        exit the program when an AccuracyCompare Exception  occurs
    """
    if check_dump_and_compare(args):
        dump_and_compare(args, use_cli)
    else:
        args.cann_path = os.path.realpath(args.cann_path)
        compare_run(args)


def dump_and_compare(args, use_cli):
    args.model_path = os.path.realpath(args.model_path)
    args.offline_model_path = os.path.realpath(args.offline_model_path)
    args.cann_path = os.path.realpath(args.cann_path)
    args.input_path = convert_npy_to_bin(args.input_path)
    args.fusion_switch_file = os.path.realpath(args.fusion_switch_file) if args.fusion_switch_file else None
    try:
        check_and_run(args, use_cli)
    except utils.AccuracyCompareException as error:
        raise error


def compare_run(args: CmpArgsAdapter):
    res = compare_process(args)
    if res and args.locat:
        endnode_names_list = res[0]["GroundTruth"].split(",")
        endnode_name = endnode_names_list[0]
        error_node_list = find_accuracy_interval(args, endnode_name, input_shape="")
        error_interval_info_file = os.path.join(args.out_path, ERROR_INTERVAL_INFO_FILE)
        with ms_open(error_interval_info_file, "a+") as fp_writer:
            output_error_interval_info(fp_writer, error_node_list)


def compare_process(args: CmpArgsAdapter):
    # compare the entire network
    if args.my_path is None or args.golden_path is None or args.ops_json is None:
        raise ValueError("when compare alone. Please ensure that both --my-path and --golden-path and --ops-json are "
                         "provided.")
    net_compare = NetCompare(args.my_path, args.golden_path,
                             args.ops_json, args, golden_json_path=None)
    net_compare.accuracy_network_compare()

    if not args.locat:
        invalid_rows, _ = analyser.Analyser(args.out_path)()
    else:
        invalid_rows, _ = analyser.Analyser(args.out_path)('ALL_INVALID')
    print_advisor_info(args.out_path)
    _append_column_to_csv(args.out_path)

    return invalid_rows


def run(args: CmpArgsAdapter, input_shape, original_out_path, use_cli: bool):
    if input_shape:
        args.input_shape = input_shape
        args.out_path = os.path.join(original_out_path, get_shape_to_directory_name(args.input_shape))

    if is_saved_model_valid(args.offline_model_path):
        # npu dump
        from msprobe.infer.offline.compare.msquickcmp.npu.npu_tf_adapter_dump_data import NpuTfAdapterDumpData
        npu_dump = NpuTfAdapterDumpData(args, args.offline_model_path)
        npu_dump.generate_inputs_data()
        npu_dump_data_path, output_json_path = npu_dump.generate_dump_data()
        # gpu dump
        from msprobe.infer.offline.compare.msquickcmp.tf.tf_save_model_dump_data import TfSaveModelDumpData
        golden_dump = TfSaveModelDumpData(args, args.model_path)
        golden_dump.generate_inputs_data(False)
        golden_dump_data_path = golden_dump.generate_dump_data(output_json_path)
        # compare the entire network
        net_compare = NetCompare(npu_dump_data_path, golden_dump_data_path,
                                 output_json_path, args, golden_json_path=None)
        net_compare.accuracy_network_compare()
        if not args.locat:
            invalid_rows, _ = analyser.Analyser(args.out_path)()
        else:
            invalid_rows, _ = analyser.Analyser(args.out_path)('ALL_INVALID')
        print_advisor_info(args.out_path)
        _append_column_to_csv(args.out_path)
    else:
        invalid_rows = run_om_model_compare(args, use_cli)

    return invalid_rows


def run_om_model_compare(args, use_cli):
    # whether use aipp
    output_json_path = atc_utils.convert_model_to_json(args.cann_path, args.offline_model_path, args.out_path)
    golden_json_path = None
    if args.model_path.endswith('.om'):
        golden_json_path = atc_utils.convert_model_to_json(args.cann_path, args.model_path, args.out_path)

    temp_om_parser = OmParser(output_json_path)
    use_aipp = True if temp_om_parser.get_aipp_config_content() else False

    if use_aipp and args.fusion_switch_file is not None:
        utils.logger.error("if .om model is using aipp config, --fusion-switch-file arg is not support.")
        raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PARAM_ERROR)

    npu_dump = NpuDumpData(args, is_golden=False)

    # generate npu inputs data
    npu_dump.generate_inputs_data(use_aipp=use_aipp)

    # generate npu dump data
    npu_dump_data_path, npu_net_output_data_path = npu_dump.generate_dump_data(use_cli=use_cli)

    # convert data from bin to npy if --convert is used, or if custom_op is not empty
    if args.bin2npy or args.custom_op != "":
        npu_dump_npy_path = convert_bin_dump_data_to_npy(npu_dump_data_path, npu_net_output_data_path, args.cann_path)
    else:
        npu_dump_npy_path = ""

    # generate onnx inputs data
    golden_dump, model_extension = _generate_golden_data_model(args, npu_dump_npy_path)
    expect_net_output_node = npu_dump.get_expect_output_name()

    # generate dump data by golden model
    if is_saved_model_valid(args.model_path):
        golden_dump.generate_inputs_data(True)
        golden_dump_data_path = golden_dump.generate_dump_data(output_json_path)
    else:
        golden_dump.generate_inputs_data(npu_dump_data_path, use_aipp)
        if isinstance(golden_dump, NpuDumpData):
            golden_dump_data_path, _ = golden_dump.generate_dump_data(npu_dump_npy_path, npu_dump.om_parser)
        else:
            golden_dump_data_path = golden_dump.generate_dump_data(npu_dump_npy_path, npu_dump.om_parser)
    golden_net_output_info = golden_dump.get_net_output_info()

    # if it's dynamic batch scenario, golden data files should be renamed
    utils.handle_ground_truth_files(npu_dump.om_parser, npu_dump_data_path, golden_dump_data_path)

    if not args.dump:
        # only compare the final output
        net_compare = NetCompare(npu_net_output_data_path, golden_dump_data_path,
                                 output_json_path, args, golden_json_path)
        net_compare.net_output_compare(npu_net_output_data_path, golden_net_output_info)
    else:
        # compare the entire network
        net_compare = NetCompare(npu_dump_data_path, golden_dump_data_path,
                                 output_json_path, args, golden_json_path)
        net_compare.accuracy_network_compare()

    # Check and correct the mapping of net output node name.
    if len(expect_net_output_node) == 1:
        _check_output_node_name_mapping(expect_net_output_node, golden_net_output_info)
    if not args.locat:
        invalid_rows, _ = analyser.Analyser(args.out_path)()
    else:
        invalid_rows, _ = analyser.Analyser(args.out_path)('ALL_INVALID')

    node_output_show_list = None
    if model_extension == ".onnx":
        node_output_show_list = _get_model_output_node_name_list(golden_dump.model_with_inputs_session,
                                                                 golden_dump.origin_model)
    print_advisor_info(args.out_path)
    _append_column_to_csv(args.out_path, node_output_show_list)
    return invalid_rows


def _get_model_output_node_name_list(model_with_inputs_session, origin_model):
    net_output_node_name_list = [item.name for item in model_with_inputs_session.get_outputs()]
    node_output_show_list = []
    for node_name in net_output_node_name_list:
        pre_node = _find_previous_node(origin_model.graph, node_name)
        if pre_node is None:
            return None
        node_output_show_list.append(pre_node)
    return node_output_show_list


def _find_previous_node(graph, output_name):
    # 遍历所有节点
    for node in graph.node:
        if output_name in [output for output in node.output]:
            # 找到目标输出节点
            return node.name
    return None


def print_advisor_info(out_path):
    advisor_info_txt_path = os.path.join(out_path, 'advisor_summary.txt')
    if Rule.input_file().check(advisor_info_txt_path):
        utils.logger.info(f"The advisor summary (.txt) is saved in :\"{advisor_info_txt_path}\"")
        advisor_info_txt_path = load_file_to_read_common_check(advisor_info_txt_path)
        with ms_open(advisor_info_txt_path, 'r', max_size=TENSOR_MAX_SIZE) as advisor_file:
            lines = advisor_file.readlines()
            for line in lines:
                utils.logger.info(line.strip())


def fusion_close_model_convert(args: CmpArgsAdapter):
    if args.fusion_switch_file:
        args.fusion_switch_file = os.path.realpath(args.fusion_switch_file)
        utils.check_file_or_directory_path(args.fusion_switch_file)

        om_json_path = atc_utils.convert_model_to_json(args.cann_path, args.offline_model_path, args.out_path)
        om_parser = OmParser(om_json_path)
        atc_input_shape_in_offline_model = DynamicInput.get_input_shape_from_om(om_parser)

        close_fusion_om_file = os.path.join(args.out_path, 'close_fusion_om_model')
        atc_command_file_path = atc_utils.get_atc_path(args.cann_path)
        atc_cmd = [
            atc_command_file_path, "--framework=5",
            "--soc_version=" + acl.get_soc_name(),
            "--model=" + args.model_path,
            "--output=" + close_fusion_om_file,
            "--fusion_switch_file=" + args.fusion_switch_file
        ]
        if atc_input_shape_in_offline_model:
            atc_cmd.append("--input_shape=" + atc_input_shape_in_offline_model)

        utils.execute_command(atc_cmd)

        om_files = find_om_files(args.out_path)
        if len(om_files) != 1:
            utils.logger.error('The number of .om files in output directory is not 1, please check.')
            raise AccuracyCompareException(utils.ACCURACY_COMPARISON_INVALID_PATH_ERROR)
        args.model_path = os.path.realpath(om_files[0])


def check_and_run(args: CmpArgsAdapter, use_cli: bool):
    utils.check_file_or_directory_path(args.model_path, is_saved_model_valid(args.model_path))
    utils.check_file_or_directory_path(args.offline_model_path, is_saved_model_valid(args.offline_model_path))
    if args.weight_path:
        utils.check_file_or_directory_path(args.weight_path)
    utils.check_device_param_valid(args.device)
    utils.check_file_or_directory_path(os.path.realpath(args.out_path), True)
    if args.fusion_switch_file:
        utils.check_file_or_directory_path(args.fusion_switch_file)
    utils.check_convert_is_valid_used(args.dump, args.bin2npy, args.custom_op)
    utils.check_locat_is_valid(args.dump, args.locat)
    sp.check_single_op_is_valid(args.single_op, args.dump, args.custom_op, args.locat)
    utils.check_max_size_param_valid(args.max_cmp_size)

    time_dir = time.strftime("%Y%m%d%H%M%S", time.localtime())
    original_out_path = os.path.realpath(os.path.join(args.out_path, time_dir))
    args.out_path = original_out_path
    # close fusion if not saved_model
    if not is_saved_model_valid(args.offline_model_path):
        fusion_close_model_convert(args)

    # deal with the dymShape_range param if exists
    input_shapes = []
    if args.dym_shape_range:
        input_shapes = utils.parse_dym_shape_range(args.dym_shape_range)
    if not input_shapes:
        input_shapes.append("")
    for input_shape in input_shapes:
        res = run(args, input_shape, original_out_path, use_cli)
        if args.single_op:
            single_op_compare(args, input_shape)
            continue
        if res and args.locat:
            endnode_names_list = res[0]["GroundTruth"].split(",")
            endnode_name = endnode_names_list[0]
            error_node_list = find_accuracy_interval(args, endnode_name, input_shape)
            error_interval_info_file = os.path.join(args.out_path, ERROR_INTERVAL_INFO_FILE)
            with ms_open(error_interval_info_file, "a+") as fp_writer:
                output_error_interval_info(fp_writer, error_node_list)
    if args.dym_shape_range:
        csv_sum(original_out_path)


def single_op_compare(args, input_shape):
    # load onnx model
    og = OnnxGraph.parse(args.model_path)
    og.infer_shape()

    # set broken single operator onnx file path
    subgraph_onnx_file = os.path.join(args.out_path, "broken.onnx")
    sp.broken(og, subgraph_onnx_file)

    # load broken single operator onnx
    subog = OnnxGraph.parse(subgraph_onnx_file)
    single_op_dir = sp.generate_single_op_dir(args.out_path)
    memory_size = sp.get_memory_size_by_soc_type(args.device)

    # devide onnx into fixed size onnxs
    subonnx_list = sp.dynamic_divide_onnx(args.out_path, subog, memory_size)

    # set csv list
    csv_list = []

    # set golden dump data source file
    onnx_data_path = os.path.join(args.out_path, 'dump_data/onnx')

    # for each onnx run compare
    for idx, subonnx in enumerate(subonnx_list):
        # run atc to get om file
        subgraph_om_file = os.path.join(args.out_path, 'broken')
        sp.atc_conversion(subonnx, subgraph_om_file)

        # get onnx input data from golden dump data
        # load single operator onnx
        utils.logger.info("Start to loading input data")
        subog = OnnxGraph.parse(subonnx)

        # load onnx input description
        Rule.input_file().check(subonnx, will_raise=True)
        inputs_list = [(ii.name, ii.shape) for ii in onnxruntime.InferenceSession(subonnx).get_inputs()]

        # find all the data needed
        input_need_list = al.input_completion(og, inputs_list)
        pattern = '|'.join(input_need_list)
        try:
            matched_files = al.find_npy_files_with_prefix(onnx_data_path, pattern)
        except Exception as e:
            utils.logger.error("Failed to find onnx dump data, please check whether file path is right")
            raise AccuracyCompareException(utils.ACCRACY_COMPARISON_FETCH_DATA_ERROR) from e
        sort_matched_files = []
        for prefix in input_need_list:
            for match_file in matched_files:
                file_name = os.path.basename(match_file)
                if file_name.startswith(prefix):
                    sort_matched_files.append(match_file)
        bin_files_path = al.create_bin_file(args.out_path, sort_matched_files)
        tmp_bin_path = os.path.join(args.out_path, 'tmp')
        utils.logger.info("Loading data Finished!")

        # set single op output data
        tmp_out_path = os.path.join(single_op_dir, f"single_op_{idx}")
        ms_makedirs(tmp_out_path, exist_ok=True)
        time_dir = time.strftime("%Y%m%d%H%M%S", time.localtime())
        original_out_path = os.path.realpath(os.path.join(args.out_path, time_dir))

        # set compare run args
        cmg_args = CmpArgsAdapter(subonnx, os.path.join(args.out_path, "broken.om"),
                                  "", bin_files_path, args.cann_path, tmp_out_path, "", args.device,
                                  "", "", False, "", True, False, custom_op="", locat=False, single_op=True)

        utils.logger.info("Start to run comparision")

        # run compare
        utils.logger.setLevel(logging.ERROR)
        _ = run(cmg_args, input_shape, original_out_path, True)
        utils.logger.setLevel(logging.INFO)
        csv_list.extend(sp.find_all_csv(tmp_out_path))
        utils.logger.info("Comparision finished")
        # remove temp bin files
        shutil.rmtree(tmp_bin_path)

    # merge csv
    summary_csv_path = utils.merge_csv(csv_list, single_op_dir, 'single_op_summary.csv')
    # analyze csv and print
    analyser.Analyser(summary_csv_path)()


def output_error_interval_info(fp_writer, error_node_list):
    for [l_node, r_node] in error_node_list:
        fp_writer.write(f"{l_node}:{r_node}")


def find_accuracy_interval(args, endnode_name, input_shape):
    """
    Function:
        find accuracy interval of the error node
    Return:
        an error node interval list
    """
    if input_shape:
        args.out_path = os.path.join(args.out_path, get_shape_to_directory_name(input_shape))

    # 读入onnx数据文件的路径
    onnx_file_path = 'dump_data/onnx'
    onnx_data_path = os.path.join(args.out_path, onnx_file_path)

    # 读入onnx模型
    og = OnnxGraph.parse(args.model_path)
    og.infer_shape()

    # 获取精度异常节点
    endnode = og.get_node(endnode_name, node_type=Node)

    output_file = './accuracy_location_log.txt'
    output_file = os.path.realpath(output_file)
    error_node_list = []
    # 验证单层算子是否有问题
    node_interval = [endnode, endnode]
    # 单层算子无问题
    if not subgraph_check(og, node_interval, args, onnx_data_path, input_shape):
        for node in og.nodes:
            if al.check_input_node(og, node):
                input_node_interval = [node, endnode]
                l_node, r_node = bin_divide(og, input_node_interval, args, onnx_data_path, input_shape)
                utils.logger.info("Accumulated Error interval has been found.")
                error_node_list.append([l_node, r_node])
        return error_node_list
    return [[endnode, endnode]]


def subgraph_check(og, node_interval, args, onnx_data_path, input_shape):
    startnode, endnode = node_interval
    subgraph_onnx_file = os.path.join(args.out_path, 'tmp_for_accuracy_locat.onnx')
    try:
        og.extract_subgraph([startnode.name], [endnode.name], subgraph_onnx_file)
    except Exception as e:
        utils.logger.error("Failed to extract subgraph model")
        raise AccuracyCompareException(utils.ACCRACY_COMPARISON_EXTRACT_ERROR) from e
    utils.logger.info("Extracting model Sucess!")
    utils.logger.info("Start using atc to convert onnx to om file")
    subgraph_om_file = os.path.join(args.out_path, 'tmp_for_accuracy_locat')
    atc_cmd = [
        "atc", "--framework=5", "--soc_version=" + acl.get_soc_name(), "--model=" + subgraph_onnx_file,
                                "--output=" + subgraph_om_file
    ]
    atc_cmd = filter_cmd(atc_cmd)
    subprocess.run(atc_cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    utils.logger.info("atc conversion Success!")
    utils.logger.info("Start to loading input data")
    subog = OnnxGraph.parse(subgraph_onnx_file)
    if Rule.input_file().check(subgraph_onnx_file):
        inputs_list = [(ii.name, ii.shape) for ii in onnxruntime.InferenceSession(subgraph_onnx_file).get_inputs()]
        input_need_list = al.input_completion(og, inputs_list)
        pattern = '|'.join(input_need_list)
    try:
        matched_files = al.find_npy_files_with_prefix(onnx_data_path, pattern)
    except Exception as e:
        utils.logger.error("Failed to find onnx dump data, please check whether file path is right")
        raise AccuracyCompareException(utils.ACCRACY_COMPARISON_FETCH_DATA_ERROR) from e
    sort_matched_files = []
    for prefix in input_need_list:
        for match_file in matched_files:
            file_name = os.path.basename(match_file)
            if file_name.startswith(prefix):
                sort_matched_files.append(match_file)
    bin_files_path = al.create_bin_file(args.out_path, sort_matched_files)
    tmp_bin_path = os.path.join(args.out_path, 'tmp')
    utils.logger.info("Loading data Finished!")
    tmp_out_path = os.path.join(args.out_path, 'tmpres')
    if not os.path.exists(tmp_out_path):
        ms_makedirs(tmp_out_path)
    time_dir = time.strftime("%Y%m%d%H%M%S", time.localtime())
    original_out_path = os.path.realpath(os.path.join(args.out_path, time_dir))
    cmg_args = CmpArgsAdapter(subgraph_onnx_file, os.path.join(args.out_path, "tmp_for_accuracy_locat.om"),
                              "", bin_files_path, args.cann_path, tmp_out_path, "", args.device,
                              "", "", False, "", True, False, custom_op=args.custom_op, locat=True)
    utils.logger.info("Start to run comparison")
    res = run(cmg_args, input_shape, original_out_path, True)
    utils.logger.info("Comparison finished")
    shutil.rmtree(tmp_out_path)
    if os.path.exists(tmp_bin_path):
        shutil.rmtree(tmp_bin_path)
    if al.check_res(res, endnode):
        return True
    return False


def bin_divide(og, node_interval, args, onnx_data_path, input_shape):
    """
    Function:
        using binary search to find the accuracy error interval
    Return:
        an accuracy error interval list
    """
    startnode, endnode = node_interval
    subgraph_model_path = os.path.join(args.out_path, 'tmp_for_subgraph.onnx')
    og.extract_subgraph([startnode.name], [endnode.name], subgraph_model_path)
    subog = OnnxGraph.parse(subgraph_model_path)

    utils.logger.info("Binary Search for error interval starts.")
    # 直线化
    satisfied_nodes = []
    satisfied_nodes = al.calculate_flow(subog, startnode, endnode)
    low = 0
    high = len(satisfied_nodes) - 1

    # 二分
    while low < high:
        mid = (low + high + 1) // 2
        input_node_interval = [satisfied_nodes[mid], endnode]
        if subgraph_check(og, input_node_interval, args, onnx_data_path, input_shape):
            low = mid
        else:
            high = mid - 1
    utils.logger.info("Binary Search for error interval ends.")
    return satisfied_nodes[low], endnode


def csv_sum(original_out_path):
    """
    Function:
        Summarize csv files under different shapes
        genrate a xlsx file   
    """
    csv_file_list = []
    sheet_name_list = []

    for files in os.listdir(original_out_path):
        if not os.path.isdir(os.path.join(original_out_path, files)):
            continue
        for sub_file in os.listdir(os.path.join(original_out_path, files)):
            if sub_file.endswith(".csv"):
                csv_file_list.append(os.path.join(original_out_path, files, sub_file))
                sheet_name_list.append(files)

    xlsx_file_summary = os.path.join(original_out_path, "result_summary.xlsx")

    if os.path.exists(xlsx_file_summary):
        utils.logger.error("Error, file already exists!")
        os.remove(xlsx_file_summary)

    with ms_open(xlsx_file_summary, 'wb') as fp_write:
        with pd.ExcelWriter(xlsx_file_summary) as writer:
            for i, csv_file in enumerate(csv_file_list):
                if Rule.input_file().check(csv_file):
                    data = pd.read_csv(csv_file, na_values=['NAN'])
                    data.to_excel(writer, sheet_name=sheet_name_list[i], index=False, na_rep='NAN')
