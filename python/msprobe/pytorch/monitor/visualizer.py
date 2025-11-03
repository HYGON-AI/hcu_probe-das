# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
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
