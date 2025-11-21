/* Copyright (c) 2025, Huawei Technologies.
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0  (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import '@vaadin/tabs';
import '@vaadin/tabsheet';
import { Notification } from '@vaadin/notification';
import { PolymerElement, html } from '@polymer/polymer';
import { observe, customElement, property } from '@polymer/decorators';
import useNodeInfo from './useNodeInfo';
import './components/tf_vaadin_table/index';
import './components/tf_vaddin_text_table/index';
import './components/tf_resize_height/index';
import type { UseNodeInfoType } from './useNodeInfo';
import type { NodeInfoType } from './type';
import { BENCH_PREFIX, NPU_PREFIX } from '../common/constant';
import i18next from '../common/i18n';

@customElement('graph-info-board')
class TfGraphNodeInfo extends PolymerElement {
  static readonly template = html`
    <style>
      :host {
        --selected-color: rgb(255, 255, 255);
        --matched-color: rgb(236, 235, 235);
      }
      .tab-content-wrapper {
        height: 100%;
        display: flex;
        justify-content: space-between;
      }
      .io-vaadin-table {
        width: 100%;
        height: 100%;
      }
      .vaadin-tabs {
        background-color: white;
      }
      .vaadin-tab {
        font-size: 14px;
      }
      .table-wrapper {
        height: 100%;
        width: 100%;
        display: flex;
        flex-direction: column;
        overflow: hidden;
      }
      .node-info-wrapper {
        display: flex;
        justify-content: space-between;
        background-color: rgb(199, 199, 199);
      }
      .node-info {
        font-family: Roboto, sans-serif;
        padding-left: 20px;
        display: flex;
        justify-content: flex-start;
        font-weight: 400;
        font-size: 14px;
      }
      .node-info-item {
        margin-right: 20px;
        display: flex;
        align-items: center;
      }
      .legend-wrapper {
        display: flex;
        align-items: center;
      }
      .legend-selected {
        display: inline-block;
        width: 12px;
        height: 12px;
        background-color: var(--selected-color);
        margin-right: 5px;
      }
      .legend-matched {
        display: inline-block;
        width: 12px;
        height: 12px;
        background-color: var(--matched-color);
        margin-right: 5px;
      }
      .matched-yes {
        display: inline-block;
        width: 10px;
        height: 8px;
        background: #52c41a;
        padding-top: 2px;
        margin-right: 5px;
        border-radius: 50%;
      }
      .matched-no {
        display: inline-block;
        width: 10px;
        height: 8px;
        background: #ff4d4f;
        padding-top: 2px;
        margin-right: 5px;
        border-radius: 50%;
      }
      vaadin-tabsheet::part(content) {
        background-color: white;
      }
    </style>
    <vaadin-tabsheet>
      <vaadin-tabs slot="tabs" class="vaadin-tabs">
        <vaadin-tab id="io-tab" class="vaadin-tab">
          <template is="dom-if" if="[[!isSingleGraph]]"> [[t('comparison_details')]] </template>
          <template is="dom-if" if="[[isSingleGraph]]"> [[t('node_details')]] </template>
        </vaadin-tab>
        <vaadin-tab id="stack-info-tab" class="vaadin-tab">[[t('node_information')]]</vaadin-tab>
      </vaadin-tabs>

      <div tab="io-tab" class="vaadin-tab-content">
        <tf-resize-height height="{{height}}">
          <div class="table-wrapper">
            <div class="node-info-wrapper">
              <div class="node-info">
                <template is="dom-if" if="[[npuNodeName]]">
                  <p class="node-info-item selected-node">
                    <span class="legend-selected"></span>
                    [[t('debug_node')]]：[[npuNodeName]]
                  </p>
                </template>
                <template is="dom-if" if="[[benchNodeName]]">
                  <p class="node-info-item match-node">
                    <span class="legend-matched"></span>
                    [[t('bench_node')]]：[[benchNodeName]]
                  </p>
                </template>
              </div>
              <template is="dom-if" if="[[!isSingleGraph]]">
                <div class="node-info">
                  <p class="node-info-item ">
                    <span class="matched-yes"></span>
                    [[t('matched')]]
                  </p>
                  <p class="node-info-item match-node">
                    <span class="matched-no"></span>
                    [[t('unMatched')]]
                  </p>
                </div>
              </template>
            </div>

            <tf-vaadin-table
              id="main-table"
              class="io-vaadin-table"
              io-dataset="[[ioDataset]]"
              tooltips="[[tooltips]]"
              handle-cell-click="[[handleGridCellClick]]"
              is-single-graph-node="[[isSingleGraph]]"
            >
            </tf-vaadin-table>
          </div>
        </tf-resize-height>
      </div>
      <div tab="stack-info-tab" class="vaadin-tab-content">
        <tf-resize-height height="{{height}}">
          <div class="table-wrapper">
            <tf-vaadin-text-table
              id="main-table"
              class="io-vaadin-table"
              dataset="[[detailData]]"
              handle-cell-click="[[handleGridCellClick]]"
            >
            </tf-vaadin-text-table>
          </div>
        </tf-resize-height>
      </div>
    </vaadin-tabsheet>
  `;
  @property({ type: Object })
  get t() {
    return (key: string) => i18next.t(key);
  }

  @property({ type: String, notify: true })
  selectedNode: string = '';

  @property({ type: Object })
  selection: any;

  @property({ type: Array })
  ioDataset: any[] = [];

  @property({ type: Array })
  detailData: any[] = [];

  @property({ type: String })
  npuNodeName?: string;

  @property({ type: String })
  benchNodeName?: string;

  @property({ type: Boolean })
  isSingleGraph = false;

  useNodeInfo: UseNodeInfoType = useNodeInfo();

  constructor() {
    super();
    this.setupLanguageListener();
  }

  setupLanguageListener() {
    i18next.on('languageChanged', () => {
      //更新语言后重新渲染
      const t = this.t;
      this.set('t', null);
      this.set('t', t);
    });
  }

  @observe('selectedNode')
  observeToUpdateTableData() {
    this.updateTableData(this.selectedNode);
  }

  async updateTableData(selectedNode) {
    if (!selectedNode) {
      this.set('ioDataset', []);
      this.set('detailData', []);
      this.set('npuNodeName', '');
      this.set('benchNodeName', '');
      return;
    }
    const { npuNode, benchNode } = await this._updateNodeInfo(selectedNode);
    // 考虑选中的节点是匹配节点的情况
    this.set('npuNodeName', npuNode?.name?.replace(NPU_PREFIX, ''));
    this.set('benchNodeName', benchNode?.name?.replace(BENCH_PREFIX, ''));
    const inputDataset = this.useNodeInfo.getIoDataSet(npuNode, benchNode, 'inputData');
    const outputDataSet = this.useNodeInfo.getIoDataSet(npuNode, benchNode, 'outputData');
    const ioDataset = [
      ...inputDataset.matchedIoDataset,
      ...outputDataSet.matchedIoDataset,
      ...inputDataset.unMatchedNpuIoDataset,
      ...outputDataSet.unMatchedNpuIoDataset,
      ...inputDataset.unMatchedBenchIoDataset,
      ...outputDataSet.unMatchedBenchIoDataset,
    ];
    this.set('ioDataset', ioDataset);
    const detailData = this.useNodeInfo.getDetailDataSet(npuNode, benchNode);
    this.set('detailData', detailData);
  }

  async _updateNodeInfo(selectedNode: string): Promise<{ npuNode: NodeInfoType; benchNode: NodeInfoType }> {
    let nodeType; // 节点类型
    if (this.isSingleGraph) {
      nodeType = 'Single';
    } else {
      nodeType = this.selectedNode?.startsWith(NPU_PREFIX) ? 'NPU' : 'Bench';
    }
    const nodeInfo = {
      nodeType,
      nodeName: selectedNode?.replace(new RegExp(`^(${NPU_PREFIX}|${BENCH_PREFIX})`), ''), // 去掉前缀
    };
    const { success, data, error } = await this.useNodeInfo.getNodeInfo(nodeInfo, this.selection);
    if (success) {
      return { npuNode: data?.npu, benchNode: data?.bench };
    } else {
      Notification.show(`${this.t('retrieve_node_information_fail')}：${error}`, {
        position: 'middle',
        duration: 2000,
        theme: 'error',
      });
    }
    return { npuNode: null, benchNode: null };
  }

  // 点击单元格高亮
  handleGridCellClick(e: MouseEvent, syncGrid: HTMLElement): void {
    const target = e.composedPath()[0] as HTMLElement; // 获取点击的目标元素
    const slotValue = target.getAttribute('slot'); // 提取 slot 属性
    if (!slotValue || !slotValue.startsWith('vaadin-grid-cell-content-')) {
      return;
    }
    const cellIndex = parseInt(slotValue.split('-').pop() || '0', 10);
    // 前8个元素是表头不可选中，所以跳过
    if (cellIndex <= 8) {
      return;
    }
    const highlightedCells = this.shadowRoot?.querySelectorAll('.highlight-cell');
    highlightedCells?.forEach((cell) => cell.classList.remove('highlight-cell')); // 清除所有高亮样式
    target.classList.add('highlight-cell'); // 添加高亮样式到当前单元格
  }
}
