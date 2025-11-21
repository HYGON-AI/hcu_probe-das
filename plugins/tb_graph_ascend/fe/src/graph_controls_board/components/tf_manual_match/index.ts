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

import '@vaadin/button';
import '@vaadin/details';
import '@vaadin/combo-box';
import '@vaadin/tooltip';
import '@vaadin/progress-bar';
import '@vaadin/checkbox';
import { isEmpty } from 'lodash';
import { Notification } from '@vaadin/notification';
import { PolymerElement, html } from '@polymer/polymer';
import { customElement, property, observe } from '@polymer/decorators';
import { NPU_PREFIX, BENCH_PREFIX, DB_TYPE } from '../../../common/constant';
import useMatched from './useMatched';
import '../tf_search_combox/index';
import i18next from '../../../common/i18n'
import type { UseMatchedType } from '../../type';
import type { SelectionType } from '../../../graph_ascend/type';


@customElement('tf-manual-match')
class Legend extends PolymerElement {
  // 定义模板
  static readonly template = html`
    <style>
      .matched-button {
        display: flex;
        justify-content: end;
        margin-top: 10px;
      }
      .matched-button button {
        border: 1px solid #ccc;
        cursor: pointer;
      }
      .vaadin-details-selected {
        display: flex;
        padding-top: 0;
      }
      vaadin-combo-box::part(input-field) {
        height: 30px;
        border: 1px solid var(--paper-input-container-color, var(--secondary-text-color));
        background-color: white;
        font-size: 14px;
        border-radius: 0px;
      }
      vaadin-combo-box::part(toggle-button) {
        font-size: 14px;
      }
      vaadin-combo-box {
        width: 100%;
        border-radius: 0px;
      }
      tf-search-combox.matched-node::part(arraw-button) {
        margin-top: 28px !important;
      }
      .match-checkbox {
        font-size: 14px;
      }

      .vaadin-details-title {
        font-size: 14px;
        color: #333333;
        font-weight: 600;
        margin-bottom: 0;
      }

      .vaadin-details vaadin-details-summary {
        font-size: 15px;
        color: #333333;
        font-weight: 600;
      }
      .match-button {
        display: flex;
        justify-content: center;
      }
      .match-button vaadin-button {
        width: 100%;
        font-weight: 600;
      }
      .warning {
        display: flex;
        align-items: center;
        font-size: 14px;
        position: relative;
      }
      #question {
        cursor: pointer;
        position: absolute;
        font-size: 10px;
        top: 14px;
        left: var(--question-config-left, 120px);
      }
      .button-wrapper {
        position: relative;
        margin-top: 30px;
      }
      .button-wrapper vaadin-button {
        width: 100%;
        font-weight: 600;
      }
      #match-config {
        color: white;
        position: absolute;
        top: 10px;
        left: 286px;
        font-size: 10px;
        cursor: pointer;
      }

      #match-save {
        color: white;
        position: absolute;
        top: 50px;
        left: 286px;
        font-size: 10px;
        cursor: pointer;
      }
    </style>
    <vaadin-details class="vaadin-details" summary="[[t('node_match')]]" opened>
      <div class="warning">
        <vaadin-combo-box
          label="[[t('select_match_config_file')]]"
          items="[[matchedConfigFiles]]"
          value="{{selectedConfigFile}}"
          on-change="_addMatchedNodesLinkByConfigFile"
        ></vaadin-combo-box>
        <vaadin-icon id="question" icon="vaadin:question-circle"></vaadin-icon>
        <vaadin-tooltip
          for="question"
          text="[[t('select_match_config_file_desc')]]"
          position="end"
        ></vaadin-tooltip>
      </div>
      <template is="dom-if" if="[[matchConfigLoading]]">
        <vaadin-progress-bar indeterminate></vaadin-progress-bar>
      </template>
      <div class="unmatched-node">
        <p class="vaadin-details-title">[[t('unmatched_node')]]</p>
        <tf-search-combox
          label="[[t('debug')]]([[npuUnMatchedNodes.length]])"
          items="[[npuUnMatchedNodes]]"
          selected-value="{{selectedNpuUnMatchedNode}}"
          on-select-change="[[_changeNpuUnMatchedNode]]"
          is-compare-graph="[[isCompareGraph]]"
        ></tf-search-combox>
        <tf-search-combox
          label="[[t('bench')]]([[benchUnMatchedNodes.length]])"
          items="[[benchUnMatchedNodes]]"
          selected-value="{{selectedBenchUnMatchedNode}}"
          on-select-change="[[_changeBenchUnMatchedNode]]"
          is-compare-graph="[[isCompareGraph]]"
        ></tf-search-combox>
        <template is="dom-if" if="[[matchLoading]]">
          <vaadin-progress-bar indeterminate></vaadin-progress-bar>
        </template>
        <vaadin-checkbox class="match-checkbox" label="[[t('manage_nodes_and_subs')]]" checked={{isMatchChildren}}></vaadin-checkbox>
        <div class="match-button">
          <vaadin-button theme="secondary small" on-click="_addMatchedNodesLink">[[t('click_to_match')]]</vaadin-button>
        </div>
      </div>
      <div class="matched-node">
        <p class="vaadin-details-title">[[t('matched_nodes')]]</p>
        <tf-search-combox
          class="matched-node"
          label="[[t('debug')]]([[npuMatchedNodes.length]])"
          items="[[npuMatchedNodes]]"
          selected-value="{{selectedNpuMatchedNode}}"
          on-select-change="[[_changeNpuMatchedNode]]"
          is-compare-graph="[[isCompareGraph]]"
        ></tf-search-combox>
        <tf-search-combox
          class="matched-node"
          label="[[t('bench')]]([[benchMatchedNodes.length]])"
          items="[[benchMatchedNodes]]"
          selected-value="{{selectedBenchMatchedNode}}"
          on-select-change="[[_changeBenchMatchedNode]]"
          is-compare-graph="[[isCompareGraph]]"
        ></tf-search-combox>
        <template is="dom-if" if="[[unmatchLoading]]">
          <vaadin-progress-bar indeterminate></vaadin-progress-bar>
        </template>
        <vaadin-checkbox class="match-checkbox" label="[[t('manage_nodes_and_subs')]]" checked={{isUnMatchChildren}}></vaadin-checkbox>
        <div class="match-button">
          <vaadin-button theme="secondary small" on-click="_deletelMatchedNodesLink">[[t('cancel_matching')]]</vaadin-button>
        </div>
      </div>

      <div class="button-wrapper">
        <vaadin-button
          class="save-button"
          theme="primary small"
          on-click="_saveMatchedRelations"
          disabled="[[saveLoading]]"
          >[[t('generate_matching_configuration_file')]]</vaadin-button
        >
        <vaadin-icon id="match-config" icon="vaadin:question-circle"></vaadin-icon>
        <vaadin-tooltip
          for="match-config"
          text="[[t('generate_matching_configuration_file_tooltip')]]"
          position="end"
        ></vaadin-tooltip>
        <template is="dom-if" if="[[!isDBType]]">
          <vaadin-button
            class="save-button"
            theme="primary contrast small"
            on-click="_saveMatchedNodesLink"
            disabled="[[saveLoading]]"
            >[[t('save')]]</vaadin-button
          >
          <vaadin-icon id="match-save" icon="vaadin:question-circle"></vaadin-icon>
          <vaadin-tooltip
            for="match-save"
            text="[[t('save_tooltip')]]"
            position="end"
          ></vaadin-tooltip>
          <template is="dom-if" if="[[saveLoading]]">
            <vaadin-progress-bar indeterminate></vaadin-progress-bar>
          </template>
        </template>
      </div>
    </vaadin-details>
  `;

  @property({ type: Object })
  t: Function = (key) => i18next.t(key);

  @property({ type: Object })
  unmatched: any = [];

  @property({ type: Boolean })
  isDBType = false;

  @property({ type: Object })
  selection: SelectionType = {} as SelectionType;

  @property({ type: Boolean })
  isCompareGraph: boolean = true;

  @property({ type: String, notify: true })
  selectedNode: string = '';

  @property({ type: Array })
  npuMatchedNodes: Array<string> = [];

  @property({ type: Array })
  benchMatchedNodes: Array<string> = [];

  @property({ type: Array })
  npuUnMatchedNodes: Array<string> = [];

  @property({ type: Array })
  benchUnMatchedNodes: Array<string> = [];

  @property({ type: String })
  selectedNpuMatchedNode: string = '';

  @property({ type: String })
  selectedBenchMatchedNode: string = '';

  @property({ type: String })
  selectedNpuUnMatchedNode: string = '';

  @property({ type: String })
  selectedBenchUnMatchedNode: string = '';

  @property({ type: Boolean })
  saveLoading: boolean = false;

  @property({ type: Boolean })
  matchLoading: boolean = false;

  @property({ type: Boolean })
  matchConfigLoading: boolean = false;

  @property({ type: Boolean })
  unmatchLoading: boolean = false;

  @property({ type: Object })
  colorset: object = {};

  @property({ type: Object })
  npuMatchNodes: object = {};

  @property({ type: Object })
  benchMatchNodes: object = {};

  @property({ type: Object })
  matchedConfigFiles: Array<string> = ['未选择'];

  @property({ type: String })
  selectedConfigFile: string = '';

  @property({ type: Boolean })
  isMatchChildren: boolean = true;

  @property({ type: Boolean })
  isUnMatchChildren: boolean = true;

  useMatched: UseMatchedType = useMatched();
  npuMatchedNodeList = {};
  benchMatchedNodeList = {};

  @observe('selection')
  _observeSelection(): void {
    this.set('isDBType', this.selection?.type === DB_TYPE);
  }

  @observe('t')
  _observeT(): void {
    if (this.t) {
      this.updateStyles({
        '--question-config-left': i18next.language === 'zh-CN' ? '120px' : '162px',
      });
    }
  }

  @observe('unmatched')
  _observeUnmatchedNode(): void {
    this.set('npuUnMatchedNodes', this.unmatched.npuNodeList || []);
    this.set('benchUnMatchedNodes', this.unmatched.benchNodeList || []);
    this.set('selectedNpuUnMatchedNode', '');
    this.set('selectedBenchUnMatchedNode', '');
  }

  @observe('npuMatchNodes', 'benchMatchNodes')
  _observeMatchNodes(): void {
    if (!this.isCompareGraph) {
      return;
    }
    // 初始化已匹配节点列表
    this.npuMatchedNodeList = this.npuMatchNodes;
    this.benchMatchedNodeList = this.benchMatchNodes;
    this.set('npuMatchedNodes', Object.keys(this.npuMatchNodes || {}));
    this.set('benchMatchedNodes', Object.keys(this.benchMatchNodes || {}));
    this.set('selectedNpuMatchedNode', '');
    this.set('selectedBenchMatchedNode', '');
  }

  @observe('matchedConfigFiles')
  _observeMatchedConfigFiles(): void {
    this.set('selectedConfigFile', '未选择');
  }

  @observe('selectedNode')
  _observeSelectedNode(): void {
    if (isEmpty(this.selectedNode) || !this.isCompareGraph) {
      return;
    }
    if (this.selectedNode.startsWith(NPU_PREFIX)) {
      this.set('selectedNpuUnMatchedNode', this.selectedNode.replace(NPU_PREFIX, ''));
      const selectedNpuMatchedNode = this.selectedNode.replace(NPU_PREFIX, '');
      this.set('selectedNpuMatchedNode', selectedNpuMatchedNode);
      this.set('selectedBenchMatchedNode', this.npuMatchedNodeList?.[selectedNpuMatchedNode]);
    } else if (this.selectedNode.startsWith(BENCH_PREFIX)) {
      this.set('selectedBenchUnMatchedNode', this.selectedNode.replace(BENCH_PREFIX, ''));
      const selectedBenchMatchedNode = this.selectedNode.replace(BENCH_PREFIX, '');
      this.set('selectedBenchMatchedNode', selectedBenchMatchedNode);
      this.set('selectedNpuMatchedNode', this.benchMatchedNodeList?.[selectedBenchMatchedNode]);
    }
  }

  // 一定要写箭头函数，不然父子组件传值this指向有问题
  _changeNpuUnMatchedNode = (): void => {
    if (this.isCompareGraph) {
      const node = NPU_PREFIX + this.selectedNpuUnMatchedNode;
      this.set('selectedNode', node);
    } else {
      Notification.show(this.t('build_not_support_match'), {
        position: 'middle',
        duration: 2000,
        theme: 'contrast',
      });
    }
  };

  _changeBenchUnMatchedNode = (): void => {
    if (this.isCompareGraph) {
      const node = BENCH_PREFIX + this.selectedBenchUnMatchedNode;
      this.set('selectedNode', node);
    } else {
      Notification.show(this.t('build_not_support_match'), {
        position: 'middle',
        duration: 2000,
        theme: 'contrast',
      });
    }
  };

  _changeNpuMatchedNode = (): void => {
    if (this.isCompareGraph) {
      const node = NPU_PREFIX + this.selectedNpuMatchedNode;
      this.set('selectedBenchMatchedNode', this.npuMatchedNodeList[this.selectedNpuMatchedNode]);
      this.set('selectedNode', node);
      this.set('selectedNode', BENCH_PREFIX + this.selectedBenchMatchedNode); // 展开对应侧节点
    } else {
      Notification.show(this.t('build_not_support_match'), {
        position: 'middle',
        duration: 2000,
        theme: 'contrast',
      });
    }
  };

  _changeBenchMatchedNode = (): void => {
    if (this.isCompareGraph) {
      const node = BENCH_PREFIX + this.selectedBenchMatchedNode;
      this.set('selectedNpuMatchedNode', this.benchMatchedNodeList[this.selectedBenchMatchedNode]);
      this.set('selectedNode', node);
      this.set('selectedNode', NPU_PREFIX + this.selectedNpuMatchedNode); // 展开对应侧节点
    } else {
      Notification.show(this.t('build_not_support_match'), {
        position: 'middle',
        duration: 2000,
        theme: 'contrast',
      });
    }
  };

  // 取消匹配
  async _deletelMatchedNodesLink(): Promise<void> {
    if (!this.isCompareGraph) {
      Notification.show(this.t('build_not_support_match'), {
        position: 'middle',
        duration: 2000,
        theme: 'contrast',
      });
      return;
    }
    if (!this.selectedNpuMatchedNode || !this.selectedBenchMatchedNode) {
      Notification.show(this.t('unmatch_first'), {
        position: 'middle',
        duration: 2000,
        theme: 'contrast',
      });
      return;
    }
    this.set('unmatchLoading', true);
    const { success, data, error } = await this.useMatched.deleteMatchedNodesLink(
      this.selectedNpuMatchedNode,
      this.selectedBenchMatchedNode,
      this.selection,
      this.isUnMatchChildren
    );
    this.set('unmatchLoading', false);
    if (success) {
      const npuMatchNodes = data?.npuMatchNodes || {};
      const benchMatchNodes = data?.benchMatchNodes || {};
      const npuUnMatchNodes = data?.npuUnMatchNodes || [];
      const benchUnMatchNodes = data?.benchUnMatchNodes || [];
      // 更新节点之间的匹配关系,更新匹配精度,节点重新上色
      const updateHierarchyData = new CustomEvent('updateHierarchyData', { bubbles: true, composed: true });
      const processedNodeNum = Math.abs(npuUnMatchNodes?.length - this.npuUnMatchedNodes?.length);
      this.dispatchEvent(updateHierarchyData);
      // 更新匹配关系
      this.npuMatchedNodeList = npuMatchNodes;
      this.benchMatchedNodeList = benchMatchNodes;
      // 未匹配列表更新
      this.set('npuUnMatchedNodes', npuUnMatchNodes);
      this.set('benchUnMatchedNodes', benchUnMatchNodes);
      // 已匹配列表更新
      this.set('npuMatchedNodes', Object.keys(npuMatchNodes || {}));
      this.set('benchMatchedNodes', Object.keys(benchMatchNodes || {}));
      // 未匹配列表选择取消匹配的节点
      this.set('selectedNpuUnMatchedNode', this.selectedNpuMatchedNode);
      this.set('selectedBenchUnMatchedNode', this.selectedBenchMatchedNode);
      // 已匹配列表清空选中的节点
      this.set('selectedNpuMatchedNode', '');
      this.set('selectedBenchMatchedNode', '');
      Notification.show(`${this.t('cancel_match_success_part1')}${processedNodeNum}${this.t('cancel_match_success_part2')}`, {
        position: 'middle',
        duration: 4000,
        theme: 'success',
      });
    } else {
      Notification.show(`${this.t('cancel_match_fail')}${error}`, {
        position: 'middle',
        duration: 3000,
        theme: 'error',
      });
    }
  }

  _addMatchedNodesLinkByConfigFile = async (): Promise<void> => {
    if (!this.isCompareGraph) {
      Notification.show(this.t('build_not_support_match'), {
        position: 'middle',
        duration: 2000,
        theme: 'contrast',
      });
      return;
    }
    if (!this.selectedConfigFile || this.selectedConfigFile === '未选择') {
      return;
    }
    this.set('matchConfigLoading', true);
    const { success, data, error } = await this.useMatched.addMatchedNodesLinkByConfigFile(
      this.selectedConfigFile,
      this.selection,
    );
    this.set('matchConfigLoading', false);
    if (success) {
      const matchResult = data?.matchResult || [];
      const npuMatchNodes = data?.npuMatchNodes || {};
      const benchMatchNodes = data?.benchMatchNodes || {};
      const npuUnMatchNodes = data?.npuUnMatchNodes;
      const benchUnMatchNodes = data?.benchUnMatchNodes;
      // 更新节点之间的匹配关系,更新匹配精度,节点重新上色
      const updateHierarchyData = new CustomEvent('updateHierarchyData', { bubbles: true, composed: true });
      const processedNodeNum = matchResult?.length;
      const matchSuccessNum = matchResult?.filter(Boolean).length;
      const matchFailedNum = processedNodeNum - matchSuccessNum;
      // 更新匹配关系
      this.npuMatchedNodeList = npuMatchNodes;
      this.benchMatchedNodeList = benchMatchNodes;
      this.dispatchEvent(updateHierarchyData);
      // 已匹配列表添加匹配成功的节点
      this.set('npuMatchedNodes', Object.keys(npuMatchNodes || {}));
      this.set('benchMatchedNodes', Object.keys(benchMatchNodes || {}));
      // 未匹配列表删除匹配成功的节点
      this.set('npuUnMatchedNodes', npuUnMatchNodes);
      this.set('benchUnMatchedNodes', benchUnMatchNodes);
      Notification.show(
        this.t('match_success_part1') + processedNodeNum + 
        this.t('match_success_part2') + matchSuccessNum + 
        this.t('match_success_part3') + matchFailedNum,
        {
          position: 'middle',
          duration: 4000,
          theme: 'success',
        },
      );
    } else {
      Notification.show(`${this.t('match_fail')}:${error}`, {
        position: 'middle',
        duration: 3000,
        theme: 'error',
      });
    }
  };

  // 匹配节点
  async _addMatchedNodesLink(): Promise<void> {
    if (!this.isCompareGraph) {
      Notification.show(this.t('build_not_support_match'), {
        position: 'middle',
        duration: 2000,
        theme: 'contrast',
      });
      return;
    }
    if (!this.selectedNpuUnMatchedNode || !this.selectedBenchUnMatchedNode) {
      Notification.show(this.t('select_match_node'), {
        position: 'middle',
        duration: 2000,
        theme: 'contrast',
      });
      return;
    }

    this.set('matchLoading', true);
    const { success, data, error } = await this.useMatched.addMatchedNodesLink(
      this.selectedNpuUnMatchedNode,
      this.selectedBenchUnMatchedNode,
      this.selection,
      this.isMatchChildren,
    );
    this.set('matchLoading', false);
    if (success) {
      const npuMatchNodes = data?.npuMatchNodes || {};
      const benchMatchNodes = data?.benchMatchNodes || {};
      const npuUnMatchNodes = data?.npuUnMatchNodes || [];
      const benchUnMatchNodes = data?.benchUnMatchNodes || [];
      // 更新节点之间的匹配关系,更新匹配精度,节点重新上色
      const updateHierarchyData = new CustomEvent('updateHierarchyData', { bubbles: true, composed: true });
      const processedNodeNum = Math.abs(npuUnMatchNodes?.length - this.npuUnMatchedNodes.length);
      // 更新匹配关系
      this.npuMatchedNodeList = npuMatchNodes;
      this.benchMatchedNodeList = benchMatchNodes;
      this.dispatchEvent(updateHierarchyData);
      // 已匹配列表添加匹配成功的节点
      this.set('npuMatchedNodes', Object.keys(npuMatchNodes || {}));
      this.set('benchMatchedNodes', Object.keys(benchMatchNodes || {}));
      // 未匹配列表删除匹配成功的节点
      this.set('npuUnMatchedNodes', npuUnMatchNodes);
      this.set('benchUnMatchedNodes', benchUnMatchNodes);
      // 已匹配列表选择匹配成功的节点
      this.set('selectedNpuMatchedNode', this.selectedNpuUnMatchedNode);
      this.set('selectedBenchMatchedNode', this.selectedBenchUnMatchedNode);
      // 未匹配列表清空选中的节点
      this.set('selectedNpuUnMatchedNode', '');
      this.set('selectedBenchUnMatchedNode', '');
      Notification.show(`${this.t('match_success_part1')}${processedNodeNum}${this.t('match_success_part4')}`, {
        position: 'middle',
        duration: 4000,
        theme: 'success',
      });
    } else {
      Notification.show(`${this.t('operation_fail')}:${error}`, {
        position: 'middle',
        duration: 3000,
        theme: 'error',
      });
    }
  }

  // 保存
  async _saveMatchedNodesLink(): Promise<void> {
    this.set('saveLoading', true);
    const result = await this.useMatched.saveMatchedNodesLink(this.selection);
    this.set('saveLoading', false);
    if (result.success) {
      Notification.show(this.t('save_success'), {
        position: 'middle',
        duration: 3000,
        theme: 'success',
      });
    } else {
      Notification.show(`${this.t('operation_fail')}${result.error}`, {
        position: 'middle',
        duration: 3000,
        theme: 'error',
      });
    }
  }
  async _saveMatchedRelations(): Promise<void> {
    if (!this.isCompareGraph) {
      Notification.show(this.t('build_not_support_match'), {
        position: 'middle',
        duration: 2000,
        theme: 'contrast',
      });
      return;
    }
    this.set('saveLoading', true);
    const { success, data, error } = await this.useMatched.saveMatchedRelations(this.selection);
    this.set('saveLoading', false);
    if (success) {
      const configFile = data;
      const matchedConfigFiles = [...new Set(['未选择', configFile, ...this.matchedConfigFiles])];
      this.set('matchedConfigFiles', matchedConfigFiles);
      Notification.show(`${this.t('file_generate_success')}${configFile}`, {
        position: 'middle',
        duration: 3000,
        theme: 'success',
      });
    } else {
      Notification.show(`${this.t('operation_fail')}:${error}`, {
        position: 'middle',
        duration: 3000,
        theme: 'error',
      });
    }
  }
}
