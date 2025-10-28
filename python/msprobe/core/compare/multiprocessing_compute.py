# Copyright (c) 2024-2025, Huawei Technologies Co., Ltd.
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

import multiprocessing
from dataclasses import dataclass
from functools import partial

import pandas as pd
from tqdm import tqdm

from python.msprobe.core.common.log import logger
from python.msprobe.core.common.utils import CompareException
from python.msprobe.core.common.const import CompareConst
from python.msprobe.core.common.exceptions import FileCheckException
from python.msprobe.core.compare.npy_compare import compare_ops_apply, get_error_flag_and_msg
from python.msprobe.core.compare.config import ModeConfig


@dataclass
class ComparisonResult:
    cos_result: list
    euc_dist_result: list
    max_err_result: list
    max_relative_err_result: list
    one_thousand_err_ratio_result: list
    five_thousand_err_ratio_result: list
    err_msgs: list


def _ms_graph_handle_multi_process(func, result_df, mode):
    process_num = max(int((multiprocessing.cpu_count() + 1) // 4), 1)
    df_chunk_size = len(result_df) // process_num
    if df_chunk_size > 0:
        df_chunks = [result_df.iloc[i:i + df_chunk_size] for i in range(0, len(result_df), df_chunk_size)]
    else:
        df_chunks = [result_df]

    results = []
    pool = multiprocessing.Pool(process_num)

    def err_call(args):
        logger.error('multiprocess compare failed! Reason: {}'.format(args))

    for df_chunk in df_chunks:
        result = pool.apply_async(func, args=(df_chunk, mode), error_callback=err_call)
        results.append(result)

    pool.close()

    try:
        final_results = [r.get(timeout=3600) for r in results]
    except Exception as e:
        logger.error(f"Task failed with exception: {e}")
        pool.terminate()
        raise CompareException(CompareException.MULTIPROCESS_ERROR) from e

    pool.join()
    return pd.concat(final_results, ignore_index=True)


def check_accuracy(cos, max_abs_err):
    if cos == CompareConst.SHAPE_UNMATCH:
        return CompareConst.ACCURACY_CHECK_UNMATCH
    if cos == CompareConst.NONE or max_abs_err == CompareConst.NONE:
        return CompareConst.NONE
    if cos == "N/A" or max_abs_err == "N/A":
        return CompareConst.ACCURACY_CHECK_NO
    try:
        cos, max_abs_err = float(cos), float(max_abs_err)
    except ValueError:
        logger.warning("Cosine or MaxAbsErr can not get float value.")
        return CompareConst.NONE
    if cos < CompareConst.COS_THRESHOLD and max_abs_err > CompareConst.MAX_ABS_ERR_THRESHOLD:
        return CompareConst.ACCURACY_CHECK_NO
    if cos < CompareConst.COS_MAX_THRESHOLD or max_abs_err > CompareConst.MAX_ABS_ERR_MAX_THRESHOLD:
        return CompareConst.ACCURACY_CHECK_NO
    return CompareConst.ACCURACY_CHECK_YES


class CompareRealData:
    def __init__(self, file_reader, mode_config: ModeConfig, cross_frame):
        self.file_reader = file_reader
        self.mode_config = mode_config
        self.cross_frame = cross_frame

    @staticmethod
    def read_dump_data(result_df):
        try:
            npu_dump_name_list = result_df.loc[0:, CompareConst.NPU_NAME].tolist()
            dump_tensor_pair_list = result_df.loc[0:, CompareConst.DATA_NAME].tolist()
            op_name_mapping_dict = {}
            for index, npu_dump_name in enumerate(npu_dump_name_list):
                dump_tensor_pair = dump_tensor_pair_list[index]
                op_name_mapping_dict[npu_dump_name] = dump_tensor_pair
            return op_name_mapping_dict
        except ValueError as e:
            logger.error('result dataframe is not found.')
            raise CompareException(CompareException.INVALID_DATA_ERROR) from e
        except KeyError as e:
            logger.error('result dataframe elements can not be access.')
            raise CompareException(CompareException.INVALID_KEY_ERROR) from e

    @staticmethod
    def _save_cmp_result(offset, result: ComparisonResult, result_df, lock):
        """
            Save comparison results into the result DataFrame with thread safety.
        Args:
            offset: offset for index
            result: data struct of ComparisonResult
            result_df: result of DataFrame
            lock: thread lock

        Returns:
            comparison results in DataFrame
        """

        lock.acquire()
        try:
            for i, cos_item in enumerate(result.cos_result):
                process_index = i + offset
                result_df.loc[process_index, CompareConst.COSINE] = cos_item
                result_df.loc[process_index, CompareConst.EUC_DIST] = result.euc_dist_result[i]
                result_df.loc[process_index, CompareConst.MAX_ABS_ERR] = result.max_err_result[i]
                result_df.loc[process_index, CompareConst.MAX_RELATIVE_ERR] = result.max_relative_err_result[i]
                result_df.loc[process_index, CompareConst.ONE_THOUSANDTH_ERR_RATIO] = (
                    result.one_thousand_err_ratio_result)[i]
                result_df.loc[process_index, CompareConst.FIVE_THOUSANDTHS_ERR_RATIO] = (
                    result.five_thousand_err_ratio_result)[i]
                result_df.loc[process_index, CompareConst.ACCURACY] = (
                    check_accuracy(result.cos_result[i], result.max_err_result[i]))
                result_df.loc[process_index, CompareConst.ERROR_MESSAGE] += result.err_msgs[i]
            return result_df
        except ValueError as e:
            logger.error('result dataframe is not found.')
            raise CompareException(CompareException.INVALID_DATA_ERROR) from e
        except IndexError as e:
            logger.error('result dataframe elements can not be access.')
            raise CompareException(CompareException.INDEX_OUT_OF_BOUNDS_ERROR) from e
        finally:
            lock.release()

    def compare_by_op(self, npu_op_name, bench_op_name, op_name_mapping_dict, input_param):
        """
        :param npu_op_name: excel中的NPU_Name，例如：MintFunctional.conv2d.0.forward.input.3.0
        :param bench_op_name: excel中的Bench_Name，例如：Functional.conv2d.0.forward.input.3.0
        :param op_name_mapping_dict: op_name和npy或pt文件的映射关系
        :param input_param: npu_path/bench_path/stack_path等参数
        :return: result_list，包含余弦相似度、最大绝对误差、最大相对误差、千分之一误差率、千分之五误差率和错误信息
        用于读取excel中的NPU_Name和Bench_Name，根据映射关系找到npy或pt文件，然后读取文件中的数据进行比较，计算余弦相似度、欧式距离
        最大绝对误差、最大相对误差、千分之一误差率、千分之五误差率并生成错误信息
        """
        relative_err, error_flag, err_msg = None, False, None

        data_name_pair = op_name_mapping_dict.get(npu_op_name)
        npu_data_name = data_name_pair[0]
        bench_data_name = data_name_pair[1]

        error_file = data_name_pair

        if str(npu_data_name) == CompareConst.NO_REAL_DATA_FLAG:  # 没有npu真实数据
            n_value, b_value, error_flag = CompareConst.NO_REAL_DATA, CompareConst.NO_REAL_DATA, True
            err_msg = "NPU does not have data file."
        elif str(bench_data_name) == CompareConst.NO_REAL_DATA_FLAG:  # 没有bench真实数据
            n_value, b_value, error_flag = CompareConst.NO_REAL_DATA, CompareConst.NO_REAL_DATA, True
            err_msg = "Bench does not have data file."
        elif str(bench_data_name) == CompareConst.N_A:  # bench没匹配
            n_value, b_value, error_flag = CompareConst.API_UNMATCH, CompareConst.API_UNMATCH, True
            err_msg = "Bench api/module unmatched."
        else:
            npu_dir = input_param.get(CompareConst.NPU_DUMP_DATA_DIR)
            bench_dir = input_param.get(CompareConst.BENCH_DUMP_DATA_DIR)
            try:
                n_value, b_value = self.file_reader(npu_dir, npu_data_name, bench_dir, bench_data_name,
                                                    self.cross_frame)
            except IOError as error:
                error_file = error.filename
                n_value, b_value = CompareConst.READ_NONE, CompareConst.READ_NONE
                error_flag = True
            except (FileCheckException, CompareException):
                error_file = data_name_pair
                n_value, b_value = CompareConst.READ_NONE, CompareConst.READ_NONE
                error_flag = True

        # 通过n_value, b_value同时得到错误标志和错误信息
        if not err_msg:
            n_value, b_value, error_flag, err_msg = get_error_flag_and_msg(n_value, b_value, error_flag=error_flag,
                                                                           error_file=error_file)

        result_list, err_msg = compare_ops_apply(n_value, b_value, error_flag, err_msg)

        if self.mode_config.fuzzy_match and npu_op_name != bench_op_name and bench_op_name != CompareConst.N_A:
            err_msg += " Fuzzy matching data, the comparison accuracy may be affected."
        result_list.append(err_msg)
        return result_list

    def compare_ops(self, idx, dump_path_dict, result_df, lock, input_param):
        cos_result = []
        euc_dist_result = []
        max_err_result = []
        max_relative_err_result = []
        one_thousand_err_ratio_result = []
        five_thousand_err_ratio_result = []
        err_mess = []

        is_print_compare_log = input_param.get("is_print_compare_log")

        for i in range(len(result_df)):
            npu_op_name = result_df.iloc[i, 0]
            bench_op_name = result_df.iloc[i, 1]
            if is_print_compare_log:
                logger.info("start compare: {}".format(npu_op_name))

            cos_sim, euc_dist, max_abs_err, max_relative_err, one_thousand_err_ratio, five_thousand_err_ratio, err_msg \
                = self.compare_by_op(npu_op_name, bench_op_name, dump_path_dict, input_param)

            if is_print_compare_log:
                if "does not have data file" in err_msg:
                    logger.info(f"[{npu_op_name}] Compare result: {err_msg} ")
                elif "Bench api/module unmatched" in err_msg:
                    logger.info(f"[{npu_op_name}] Compare result: {err_msg} ")
                else:
                    logger.info(
                        f"[{npu_op_name}] Compare result: cosine {cos_sim}, euc_dist {euc_dist}, "
                        f"max_abs_err {max_abs_err}, max_relative_err {max_relative_err}, "
                        f"one_thousand_err_ratio {one_thousand_err_ratio}, "
                        f"five_thousand_err_ratio {five_thousand_err_ratio}, {err_msg}"
                    )
            cos_result.append(cos_sim)
            euc_dist_result.append(euc_dist)
            max_err_result.append(max_abs_err)
            max_relative_err_result.append(max_relative_err)
            one_thousand_err_ratio_result.append(one_thousand_err_ratio)
            five_thousand_err_ratio_result.append(five_thousand_err_ratio)
            err_mess.append(err_msg)

        cr = ComparisonResult(
            cos_result=cos_result,
            euc_dist_result=euc_dist_result,
            max_err_result=max_err_result,
            max_relative_err_result=max_relative_err_result,
            one_thousand_err_ratio_result=one_thousand_err_ratio_result,
            five_thousand_err_ratio_result=five_thousand_err_ratio_result,
            err_msgs=err_mess
        )

        return self._save_cmp_result(idx, cr, result_df, lock)

    def do_multi_process(self, input_param, result_df):
        try:
            result_df = self._handle_multi_process(self.compare_ops, input_param, result_df,
                                                   multiprocessing.Manager().RLock())
            return result_df
        except ValueError as e:
            logger.error('result dataframe is not found.')
            raise CompareException(CompareException.INVALID_DATA_ERROR) from e

    def _handle_multi_process(self, func, input_param, result_df, lock):
        process_num = max(int((multiprocessing.cpu_count() + 1) // 4), 1)
        op_name_mapping_dict = self.read_dump_data(result_df)

        df_chunk_size = len(result_df) // process_num
        if df_chunk_size > 0:
            df_chunks = [result_df.iloc[i:i + df_chunk_size] for i in range(0, len(result_df), df_chunk_size)]
        else:
            df_chunks = [result_df]

        results = []
        pool = multiprocessing.Pool(process_num)

        def err_call(args):
            logger.error('multiprocess compare failed! Reason: {}'.format(args))

        progress_bar = tqdm(total=len(result_df), desc="API/Module Item Compare Process", unit="row", ncols=100)

        def update_progress(size, progress_lock, extra_param=None):
            with progress_lock:
                progress_bar.update(size)

        for process_idx, df_chunk in enumerate(df_chunks):
            idx = df_chunk_size * process_idx
            chunk_size = len(df_chunk)
            result = pool.apply_async(func,
                                      args=(idx, op_name_mapping_dict, df_chunk, lock, input_param),
                                      error_callback=err_call,
                                      callback=partial(update_progress, chunk_size, lock)
                                      )
            results.append(result)

        pool.close()

        try:
            final_results = [r.get(timeout=3600) for r in results]
        except Exception as e:
            logger.error(f"Task failed with exception: {e}")
            pool.terminate()
            raise CompareException(CompareException.MULTIPROCESS_ERROR) from e

        pool.join()
        return pd.concat(final_results, ignore_index=True)
