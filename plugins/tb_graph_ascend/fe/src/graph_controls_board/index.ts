/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright (c) 2025, Huawei Technologies.
Adapt to the model hierarchical visualization data collected by the msprobe tool
==============================================================================*/

import '@vaadin/icon';
import '@vaadin/icons';
import '@vaadin/select';
import '@vaadin/button';


import * as _ from 'lodash';
import { customElement, property } from '@polymer/decorators';
import { html, PolymerElement } from '@polymer/polymer';
import { DarkModeMixin } from '../polymer/dark_mode_mixin';
import { LegacyElementMixin } from '../polymer/legacy_element_mixin';
import { PaperCheckboxElement } from '../polymer/irons_and_papers';
import '../polymer/irons_and_papers';

import i18next from '../common/i18n'
import './components/tf_main_controler';
import './components/tf_manual_match/index';
import './components/tf_color_select/index';
import './components/tf_linkage_search_combox/index';
import type { MetaDirType, MinimapVis } from './type';
import type { SelectionType } from '../graph_ascend/type';

@customElement('graph-controls-board')
class TfGraphControls extends LegacyElementMixin(DarkModeMixin(PolymerElement)) {
  static readonly template = html`
    <style>
      :host {
        color: #555;
        display: flex;
        flex-direction: column;
        font-size: 12px;
        width: 100%;
        --tb-graph-controls-title-color: #000;
        --tb-graph-controls-legend-text-color: #000;
        --tb-graph-controls-text-color: #555;
        --tb-graph-controls-title-font-size: 14px;
        --tb-graph-controls-subtitle-font-size: 14px;
        --paper-input-container-shared-input-style_-_font-size: 14px;
        --paper-font-subhead_-_font-size: 14px;
      }

      .holder {
        background: rgb(246, 246, 246);
        box-sizing: border-box;
        color: var(--tb-graph-controls-text-color);
        width: 100%;
      }

      .title {
        font-size: var(--tb-graph-controls-title-font-size);
        margin: 8px 5px 0px 0;
        color: var(--tb-graph-controls-title-color);
      }
      .title small {
        font-weight: normal;
      }
      .container-wrapper {
        margin: 10px 0 20px 0;
        border-top: 1px #bfbfbf dashed;
      }
      .minimap-control {
        font-size: var(--tb-graph-controls-title-font-size);
        height: 42px;
        display: flex;
        flex-wrap: wrap;
      }

      .left-checkbox {
        margin-right: 12px;
      }

      .icon-button {
        font-size: var(--tb-graph-controls-title-font-size);
      }
      .button-text {
        padding-left: 12px;
        text-transform: none;
      }
      .button-icon {
        width: 26px;
        height: 26px;
        margin-right: 5px;
        color: var(--paper-orange-500);
        font-weight: 500;
      }

      .allcontrols .control-holder {
        clear: both;
        display: flex;
        justify-content: space-between;
      }

      .allcontrols .control-holder.control-options {
        padding: 0 0 15px 15px;
        flex-direction: column;
      }

      .tabs {
        display: flex;
        height: 40px;
        border-bottom: 1px solid #ccc;
      }

      .tab-button {
        flex: 1;
        padding: 10px;
        background: rgb(246, 246, 246);
        border: none;
        cursor: pointer;
        border-bottom: 3px solid transparent; /* 初始状态下的底部线条透明 */
      }

      .tab-button.active {
        border-bottom: 3px solid orange; /* 聚焦时显示橙色线条 */
      }

      .tab-content {
        position: relative;
        background: rgb(255, 255, 255);
        padding: 0 20px;
        flex-grow: 1;
        overflow-y: auto;
      }
      .fit-screen {
        display: flex;
        align-items: center;
      }
      .fit-screen vaadin-icon {
        margin-right: 10px;
        cursor: pointer;
        color: var(--paper-orange-500);
      }
      .fit-screen vaadin-button {
        margin-top: 6px;
        font-size: var(--tb-graph-controls-title-font-size);
        font-weight: 400;
        cursor: pointer;
      }
      .setting-title {
        display: flex;
        justify-content: space-between;
        align-items: center;
      }
      .language-button{
        font-size: var(--tb-graph-controls-title-font-size);
        font-weight: 400;
        cursor: pointer;
        color: black;
      }

      .hidden {
        display: none;
      }

      .sync-expand-checkbox {
        font-size: var(--tb-graph-controls-title-font-size);
        margin-left: -4px;
      }

      .vaadin-details-selected {
        display: flex;
        padding-top: 0;
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

      .loading-wrapper{
          position: absolute;
          width: 90%;
          height: 90%;
          z-index: 999;
          color: rgba(37, 37, 37, 0.8);
          background-color: rgba(255, 255, 255, 0.76);
          display: flex;
          justify-content: center;
          align-items: center;
          font-size: 20px;
          font-weight: 600;
      }
      paper-checkbox {
        --paper-checkbox-unchecked-color: gray; /* 选中时的颜色 */
        user-select: none;
      }
    </style>

    <div class="tabs">
      <button class="tab-button" on-tap="_showNodeControls">[[t('settings')]]</button>
      <button class="tab-button" on-tap="_showMatch">[[t('function')]]</button>
    </div>
    <div id="nodes-content" class="tab-content">
      <div class='setting-title'>
        <div class="fit-screen">
          <vaadin-icon icon="vaadin:viewport" on-click="_clickSetting"></vaadin-icon>
          <vaadin-button theme="tertiary contrast" on-click="_fit">[[t('fit')]]</vaadin-button>
        </div>
        <vaadin-button class='language-button' theme="tertiary-inline" on-click="changeLanguage">中|en</vaadin-button>
      </div>
      <div class="minimap-control">
        <paper-checkbox class="left-checkbox" checked on-change="_toggleNpuMinimap">[[t('show_debug_minimap')]]</paper-checkbox>
        <template is="dom-if" if="[[!isSingleGraph]]">
          <paper-checkbox  checked on-click="_toggleBenchMinimap">[[t('show_bench_minimap')]]</paper-checkbox>
        </template>
      </div>
      <vaadin-checkbox class="sync-expand-checkbox" label="[[t('shouldExpandNodesSync')]]" checked={{isSyncExpand}}></vaadin-checkbox>
      <div class="container-wrapper">
        <tf-main-controler
          t="[[t]]"
          meta-dir="[[metaDir]]"
          selection="{{selection}}"
          microsteps="[[microsteps]]"
          ranks="[[ranks]]"
          steps="[[steps]]"
        ></tf-main-controler>
      </div>
   
      <tf-color-select
        t="[[t]]"
        colors="{{colors}}"
        task=[[task]]
        is-overflow-filter="{{isOverflowFilter}}"
        overflowcheck="[[overflowcheck]]"
        selected-node="{{selectedNode}}"
        selection="[[selection]]"
        colorset="[[colorset]]"
        is-single-graph="[[isSingleGraph]]"
      ></tf-color-select>
    </div>
    <div id="directory-content" class="tab-content hidden"></div>
    <div id="match-content" class="tab-content hidden">
        <template is='dom-if' if='[[loading]]'>
          <div class='loading-wrapper'>
              Loading......
          </div>
      </template>
      <vaadin-details class="vaadin-details" summary="[[t('node_search')]]" opened>
        <tf-linkage-search-combox
          t="[[t]]"
          nodelist="[[nodelist]]"           
          selected-node="{{selectedNode}}"
          is-compare-graph="[[!isSingleGraph]]"
        >
        </tf-linkage-search-combox>
      </vaadin-details>
      <tf-manual-match
        t="[[t]]"
        unmatched="[[unmatched]]"
        selected-node="{{selectedNode}}"
        selection="[[selection]]"
        is-compare-graph="[[!isSingleGraph]]"
        npu-match-nodes="[[npuMatchNodes]]"
        bench-match-nodes="[[benchMatchNodes]]"
        matched-config-files="[[matchedConfigFiles]]"
      ></tf-manual-match>

    </div>
  `;

  @property({ type: Object })
  t: Function = (key) => i18next.t(key);

  @property({ type: Boolean })
  loading: boolean = false;

  @property({ type: Object })
  metaDir: MetaDirType = {} as MetaDirType;

  @property({ type: Boolean, notify: true })
  isOverflowFilter: boolean = false;

  @property({ type: Boolean })
  isSingleGraph = false;

  @property({ type: Object, notify: true })
  selection: SelectionType = {} as SelectionType;

  // 全量节点数据，支撑各种节点的搜索
  @property({ type: Object })
  nodelist: object = {};

  // 颜色图例
  @property({ type: Object })
  colorset;

  // 溢出检测标志
  @property({ type: Boolean })
  overflowcheck;

  // 节点匹配，未匹配部分节点
  @property({ type: Object })
  unmatched: any = [];

  @property({ type: String, notify: true })
  selectedNode: string | null = null;

  @property({ type: Object, notify: true })
  minimapVis: MinimapVis = { npu: true, bench: true };

  @property({ type: Boolean, notify: true })
  isSyncExpand: boolean = true;

  // 颜色数据
  @property({ type: Object, notify: true })
  colors: any;

  @property({ type: Object, notify: true })
  loadAllNodeList: Function = () => { };

  @property({ type: Object, notify: true })
  needLoadAllNodeList: boolean = true;



  override ready(): void {
    super.ready();
    this._showTabContent(this.t('settings'), 'nodes-content');
  }

  changeLanguage() {
    const currentLang = i18next.language === 'en' ? 'zh-CN' : 'en';
    i18next.changeLanguage(currentLang).then(() => {
      //更新语言后重新渲染
      const t = this.t;
      this.set('t', null);
      this.set('t', t);
      const selection = { ...this.selection, lang: currentLang }
      this.set('selection', selection);

    });
  }

  _showTabContent(buttonText, contentId): void {
    // Remove 'active' class from all buttons
    this.shadowRoot?.querySelectorAll('.tab-button').forEach((button) => {
      button.classList.remove('active');
    });

    // Add 'active' class to the clicked button
    const buttons = this.shadowRoot?.querySelectorAll('.tab-button');
    buttons?.forEach((button) => {
      if ((button as HTMLElement).innerHTML === buttonText) {
        button?.classList.add('active');
      }
    });

    // Hide all content
    this.shadowRoot?.querySelectorAll('.tab-content').forEach((content) => {
      content.classList.add('hidden');
    });

    // Show the selected content
    const selectedContent = this.shadowRoot?.getElementById(contentId);
    if (selectedContent) {
      selectedContent.classList.remove('hidden');
    }
  }

  // 使用示例
  _showNodeControls(): void {
    this._showTabContent(this.t('settings'), 'nodes-content');
  }

  async _showMatch(): Promise<void> {
    this._showTabContent(this.t('function'), 'match-content');
    if (this.loadAllNodeList && this.needLoadAllNodeList) {
      this.set('loading', true)
      await this.loadAllNodeList(this.selection);
      this.set('needLoadAllNodeList', false) //已经加载过一次,不需要再加载
      this.set('loading', false)
    }
  }

  _fit(): void {
    this.fire('fit-tap');
  }

  _toggleNpuMinimap(event: CustomEvent): void {
    const checkbox = event.target as PaperCheckboxElement;
    this.set('minimapVis.npu', checkbox.checked);
  }

  _toggleBenchMinimap(event: CustomEvent): void {
    const checkbox = event.target as PaperCheckboxElement;
    this.set('minimapVis.bench', checkbox.checked);
  }
}
