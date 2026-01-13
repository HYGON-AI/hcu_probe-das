# -------------------------------------------------------------------------
#  This file is part of the MindStudio project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from typing import Dict
from tqdm import tqdm

from msprobe.core.common.file_utils import save_json, check_path_before_create, check_path_not_exists, \
    check_file_or_directory_path
from msprobe.core.common.log import logger
from msprobe.core.config_check.ckpt_compare.megatron_loader import load_megatron_weights
from msprobe.core.config_check.ckpt_compare.metrics import METRIC_FUNC



def compare_checkpoints(ckpt_path1, ckpt_path2, output_path) -> Dict:
    """Compare weights between two checkpoints using cosine similarity and L2 distance.
    
    Args:
        ckpt_path1 (str): Path to first checkpoint directory
        ckpt_path2 (str): Path to second checkpoint directory 
        output_path (str): Path to save comparison results JSON file

    Returns:
        Dict: Dictionary containing comparison metrics for each parameter. The dictionary has the following structure:
            {
                "param_name": {
                    "cosine_similarity": float,  # Cosine similarity between parameter tensors
                    "l2_distance": float,        # L2 distance between parameter tensors  
                    "shape": List[int]           # Shape of the parameter tensors
                },
                ...
            }
    """

    # Load both checkpoints
    check_file_or_directory_path(ckpt_path1, isdir=True)
    check_file_or_directory_path(ckpt_path2, isdir=True)
    check_path_before_create(output_path)
    check_path_not_exists(output_path)
    weights1 = load_megatron_weights(ckpt_path1)
    weights2 = load_megatron_weights(ckpt_path2)
    
    # Initialize results dictionary
    results = {}
    
    # Compare weights with matching keys
    common = set(weights1) & set(weights2)
    logger.warning(f'Parameters not in ckpt2: {set(weights1) - set(weights2)}')
    logger.warning(f'Parameters not in ckpt1: {set(weights2) - set(weights1)}')
    for key in tqdm(common):
        tensor1 = weights1[key]
        tensor2 = weights2[key]
        
        results[key] = {}
        for metric, func in METRIC_FUNC.items():
            try:
                results[key][metric] = func(tensor1, tensor2)
            except Exception as e:
                results[key][metric] = 'error'
                logger.warning(f'Error when calculate {metric} for reason: {e}')

    # Write results to JSON file
    save_json(output_path, results, indent=4)
    logger.info(f"Comparison results written to {output_path}")
    return results
