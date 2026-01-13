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

import torch
import numpy as np
import matplotlib.pyplot as plt
from msprobe.pytorch.monitor.features import cal_histc


class HeatmapVisualizer:
    def __init__(self) -> None:
        self.histogram_bins_num = 30
        self.min_val = -1
        self.max_val = 1
        self.histogram_edges = None
        self.histogram_sum_data_np = None  # matrix shape is [bins_num * total_step]
        self.cur_step_histogram_data = None
        self.histogram_edges = torch.linspace(self.min_val, self.max_val, self.histogram_bins_num)

    def pre_cal(self, tensor):
        self.cur_step_histogram_data = cal_histc(tensor_cal=tensor, bins_total=self.histogram_bins_num,
                                                 min_val=self.min_val, max_val=self.max_val)

    def visualize(self, tag_name: str, step, summary_writer):
        if self.histogram_sum_data_np is None or self.histogram_sum_data_np.size == 0:
            self.histogram_sum_data_np = np.expand_dims(self.cur_step_histogram_data.cpu(), 0).T
        else:
            # add new data along a different axis because we transposed early
            # matrix shape is [bins_num * total_step]
            self.histogram_sum_data_np = np.concatenate((self.histogram_sum_data_np, np.expand_dims(
                self.cur_step_histogram_data.cpu(), 1)), axis=1)

        fig, ax = plt.subplots()
        cax = ax.matshow(self.histogram_sum_data_np, cmap='hot', aspect='auto')
        fig.colorbar(cax)

        lbs = [f'{self.histogram_edges[i]:.2f}' for i in range(self.histogram_bins_num)]
        plt.yticks(ticks=range(self.histogram_bins_num), labels=lbs)
        ax.set_xlabel('Step')
        ax.set_ylabel('Value Range')
        plt.title(f'Total Step: {step}')

        # Convert matplotlib figure to an image format suitable for TensorBoard
        fig.canvas.draw()
        image = torch.from_numpy(np.array(fig.canvas.renderer.buffer_rgba()))
        plt.close(fig)
        summary_writer.add_image(tag_name, image.permute(2, 0, 1), global_step=step, dataformats='CHW')
