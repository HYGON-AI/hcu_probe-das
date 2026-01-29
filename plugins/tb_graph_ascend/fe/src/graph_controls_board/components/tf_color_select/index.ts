/*
 * -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */

import '@vaadin/combo-box';
import '@vaadin/text-field';
import '@vaadin/checkbox';
import * as _ from 'lodash';
import { PolymerElement, html } from '@polymer/polymer';
import { Notification } from '@vaadin/notification';
import { customElement, property, observe } from '@polymer/decorators';
import { NPU_PREFIX, UNMATCHED_COLOR, defaultColorSetting, defaultColorSelects } from '../../../common/constant';
import request from '../../../utils/request';
import { DarkModeMixin } from '../../../polymer/dark_mode_mixin';
import { LegacyElementMixin } from '../../../polymer/legacy_element_mixin';
import i18next from '../../../common/i18n';
import { t } from 'i18next';
import { PaperButtonElement } from '@polymer/paper-button';
@customElement('tf-color-select')
class Legend extends LegacyElementMixin(DarkModeMixin(PolymerElement)) {
  // 定义模板
  static readonly template = html`
      <style>
        /* 定义 CSS 变量 */
        :root {
          --tb-graph-controls-legend-text-color: #333;
          --tb-graph-controls-subtitle-font-size: 12px;
          --border-color: #bfbfbf;
          --hover-background-color: rgb(201, 200, 199);
          --default-background-color: rgb(238, 238, 238);
          --default-text-color: rgb(87, 86, 86);
        }

        /* 通用图标样式 */
        vaadin-icon {
          cursor: pointer;
          height: 19px;
        }

        /* 通用工具栏样式 */
        .toolbar {
          appearance: none;
          background-color: inherit;
          padding: 10px 0 8px 0;
          border-right: none;
          border-left: none;
          color: var(--tb-graph-controls-legend-text-color);
          font: inherit;
          display: flex;
          align-items: center;
          justify-content: space-between;
          width: 100%;
          outline: none;
        }

        /* 容器包裹样式 */
        .container-wrapper {
          margin: 20px 0;
          border-top: 1px dashed var(--border-color);
        }

        /* 下拉菜单样式 */
        .dropdown {
          position: absolute;
          top: 100%;
          left: 0;
          width: 50px;
          border: 1px solid #ccc;
          background-color: white;
          z-index: 10;
        }

        /* 搜索容器样式 */
        .container-search {
          align-items: center;
        }

        /* 计数器样式 */
        .counter,
        .counter-total {
          font-size: var(--tb-graph-controls-subtitle-font-size);
          color: gray;
          margin-left: 4px;
        }

        .counter-total {
          width: 60px;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        /* 自定义选择框样式 */
        .custom-select {
          position: relative;
          display: inline-block;
        }

        .select-box {
          width: 40px;
          height: 10px;
          margin-right: 13px;
          background-color: white;
          padding: 5px;
          border: 1px solid black;
          cursor: pointer;
        }

        .option {
          padding: 10px;
          cursor: pointer;
        }

        /* 搜索箭头样式 */
        .search-arrow {
          margin-top: 26px;
          cursor: pointer;
          color: var(--default-text-color);
          background: var(--default-background-color);
          border: 1px solid black;
          padding: 4px;
          height: 30px;
          width: 22px;
        }

        .search-arrow:hover {
          background: var(--hover-background-color);
        }

        .toggle-legend-text {
          font-size: 15px;
        }
        #question {
          cursor: pointer;
          position: relative;
          font-size: 10px;
          top: -2px;
          left: 2px;
        }

        /* Vaadin 组合框样式 */
        vaadin-combo-box {
          flex: 1;
          font-size: small;
        }

        vaadin-combo-box::part(input-field) {
          background-color: white;
          border: 1px solid #0d0d0d;
          height: 30px;
          border-radius: 0;
        }

        #left {
          position: relative;
          left: var(--color-config-left);
        }

        #right {
          position: relative;
          left: var(--color-config-right);
        }

        #addButton {
          position: relative;
          left: var(--color-config-add);
        }
      </style>
      <template is="dom-if" if="[[enableConfig]]">
        <div class='container-wrapper'>
          <template is="dom-if" if="[[_colorSetting]]">
            <div>
              <template is="dom-if" if="[[!isOverflowFilter]]">
                <div class="toolbar">
                  <div style="font-size: 15px">
                    [[t("accuracy_error")]]
                  <vaadin-icon id="question" icon="vaadin:question-circle"></vaadin-icon>
                  <vaadin-tooltip
                    for="question"
                    text=[[precisionDesc]]
                    position="end"
                  ></vaadin-tooltip>
                  </div>
                  <div style="margin-left: auto; display: flex; gap: 8px;">
                    <template is="dom-if" if="[[showSwitchIcon]]">
                      <vaadin-icon icon="vaadin:exchange" on-click="_selectedTabChanged"></vaadin-icon>
                    </template>
                  </div>
                </div>
              </template>
              <template is="dom-if" if="[[isOverflowFilter]]">
                <div class="toolbar">
                  <div style="font-size: 15px">[[t("overflow")]]</div>
                  <template is="dom-if" if="[[showSwitchIcon]]">
                    <vaadin-icon icon="vaadin:exchange" on-click="_selectedTabChanged"></vaadin-icon>
                  </template>
                </div>
              </template>
              <template is="dom-if" if="[[!isOverflowFilter]]">
                <div class="run-dropdown" display: flex; flex-direction: column;">
                  <template is="dom-repeat" items="[[colorSetChanged]]">
                    <div class="color-option" style="display: flex; align-items: center;">
                      <vaadin-checkbox id="checkbox-[[index]]" on-click="_toggleCheckbox"></vaadin-checkbox>
                      <div
                        style="width: 12px; height: 12px; background-color: [[item.0]]; margin-right: 8px; border: 1px solid gray;"
                      ></div>
                      [[_getColorDisplayLabel(item.1)]]
                    </div>
                  </template>
                </div>
                <div class="container-search">
                  <tf-search-combox
                    label="[[t('match_accuracy_error')]]([[precisionmenu.length]])"
                    items="[[precisionmenu]]"
                    selected-value="{{selectedPrecisionNode}}"
                    on-select-change="[[_observePrecsionNode]]"
                  ></tf-search-combox>
                <div>
              </template>
              <template is="dom-if" if="[[isOverflowFilter]]">
                <template is="dom-if" if="{{overFlowSet.length}}">
                  <div class="container" style="display: flex; flex-direction: column;">
                    <div class="run-dropdown" display: flex; flex-direction: column;">
                      <template is="dom-repeat" items="[[overFlowSet]]">
                        <div class="color-option" style="display: flex; align-items: center;">
                          <vaadin-checkbox id="overflowCheckbox-[[index]]" on-click="_toggleCheckbox"></vaadin-checkbox>
                          <div
                            style="width: 12px; height: 12px; background-color: [[item.0]]; margin-right: 8px; border: 1px solid gray;"
                          ></div>
                          [[item.1.accuracy_level]]
                        </div>
                      </template>
                    </div>
                  </div>
                </template>
                <div class="container-search">
                  <tf-search-combox
                    label="[[t('overflow_filter_node')]]([[overflowmenu.length]])"
                    items="[[overflowmenu]]"
                    selected-value="{{selectedOverflowNode}}"
                    on-select-change="[[_observeOverFlowNode]]"
                  ></tf-search-combox>
                </div>
              </template>
            </div>
          </template>
        </div>
      </template>
    `;

  @property({ type: Object })
  t: Function = (key) => i18next.t(key);

  @property({ type: Boolean })
  _colorSetting: boolean = true; // 颜色设置按钮

  @property({ type: Boolean })
  isSingleGraph = false;

  @property({ type: Array })
  selectColor: any = [];

  @property({ type: String, notify: true })
  selectedPrecisionNode: string = '';

  @property({ type: String, notify: true })
  selectedOverflowNode: string = '';

  @property({ type: Object })
  precisionmenu: any = [];

  // 颜色图例
  @property({ type: Object })
  colorset;

  @property({ type: Object })
  colorSetChanged;

  // 溢出图例默认数据
  @property({ type: Object })
  overFlowSet: any = [
    ['#B6C7FC', 'medium'],
    ['#7E96F0', 'high'],
    ['#4668B8', 'critical'],
  ];

  // 自定义颜色设置
  @property({ type: Array })
  standardColorList = ['#FFFCF3', '#FFEDBE', '#FFDC7F', '#FFC62E', '#FF9B3D', '#FF704D', '#FF4118'];

  @property({ type: Array })
  colorList = _.cloneDeep(this.standardColorList);

  @property({ type: Array })
  colorSelects = defaultColorSelects;

  @property({ type: Object, notify: true })
  colors: any;

  @property({ type: Boolean, notify: true })
  isOverflowFilter: boolean = false;

  @property({ type: String, notify: true })
  selectedNode: string | null = null;

  // 溢出筛选
  @property({ type: Array })
  overflowLevel: any = [];

  @property({ type: Object })
  overflowmenu: any = [];

  @property({ type: Boolean })
  overflowcheck;

  @property({ type: Boolean })
  enableConfig = true;

  @property({ type: Boolean })
  showSwitchIcon = true;

  @property({ type: Object })
  selection: any = {};

  @property({ type: String })
  task: string = '';

  @property({ type: String })
  precisionDesc: string = '';

  @property({ type: String })
  unMatchedNodeName = t('no_matching_nodes')

  private closeButtonElements: Set<PaperButtonElement> = new Set();
  private titleElements: Set<HTMLHeadingElement> = new Set(); // 专门管理标题元素

  @observe('t')
  _observeT(): void {
    if (this.t) {
    const allCheckboxes = this.shadowRoot?.querySelectorAll('vaadin-checkbox');
    if (allCheckboxes) {
      allCheckboxes.forEach((checkbox) => {
        checkbox.checked = false; // 清空每个 checkbox 的选中状态
      });
    }
      this.set('precisionDesc', this.t('precision_desc'));
      this.updateStyles({
        '--color-config-left': i18next.language === 'zh-CN' ? '16px' : '36px',
        '--color-config-right': i18next.language === 'zh-CN' ? '56px' : '93px',
        '--color-config-add': i18next.language === 'zh-CN' ? '100px' : '143px',
      });
      this.closeButtonElements.forEach(button => {
        button.textContent = this.t('close');
      });
      this.titleElements.forEach(title => {
        title.textContent = this.t('info');
      });
    }
  }

  @observe('colorset')
  _observeColorSet(): void {
    if (_.isEmpty(this.colorset)) {
      return;
    } // 如果colorset为空，直接返回
    if (this.colorset.length !== 0) {
      const colorsets = this.colorset;
      for (const item of colorsets) {
        if (item[1].value.length === 0) {
          item[1].value = this.unMatchedNodeName;
        }
      }
      this.colorSetChanged = colorsets;
    } else {
      return;
    }
  }
  @observe('task')
  _observeTask(): void {
    this.set('precisionDesc', this.t('precision_desc'));
  }

  // 写一个如果切换数据清除所有checkbox和所有this.selectColor
  @observe('selection')
  _clearAllToggleCheckboxAndInputField(): void {
    this.set('selectedSide', '0');
    const allCheckboxes = this.shadowRoot?.querySelectorAll('vaadin-checkbox');
    if (allCheckboxes) {
      allCheckboxes.forEach((checkbox) => {
        checkbox.checked = false; // 清空每个 checkbox 的选中状态
      });
    }
    this.selectColor = [];
    this.precisionmenu = [];
    this.overflowLevel = [];
    // 清除精度筛选输入框
    this.set('selectedPrecisionNode', '');
    // 清除精度溢出输入框
    this.set('selectedOverflowNode', '');
    this.set('selectedNode', '');
    this.updateColorSetting();
  }

  @observe('isSingleGraph', 'overflowcheck')
  updateColorSetting(): void {
    if (!this.isSingleGraph) {
      this.set('enableConfig', true);
      this.set('showSwitchIcon', !!this.overflowcheck);
      this.set('isOverflowFilter', false);
    } else {
      if (this.overflowcheck) {
        this._selectedTabChanged();
        this.set('enableConfig', true);
        // 隐藏切换按钮
        this.set('showSwitchIcon', false);
        // 切换至精度溢出，隐藏精度筛选
        this.set('isOverflowFilter', true);
      } else {
        this.set('enableConfig', false);
      }
    }
  }
  // 请求后端接口，更新筛选数据
  updateFilterData = async () => {
    if (_.isEmpty(this.selectColor)) {
      return;
    }
    try {
      const values = this.selectColor.map((item) => item === this.unMatchedNodeName ? -1 : item);
      const params = {
        metaData: this.selection,
        type: 'precision',
        values,
      };
      const { success, data, error } = await request({ url: 'screen', method: 'POST', data: params })

      if (success) {
        this.set('precisionmenu', data);
        this.set('selectedPrecisionNode', data?.[0] || '');
      }
      else {
        Notification.show(`Error:${error}`, {
          position: 'middle',
          duration: 4000,
          theme: 'error',
        });
      }

    }
    catch (error) {
      Notification.show(this.t('retrieve_precision_menu_fail'), {
        position: 'middle',
        duration: 4000,
        theme: 'error',
      });
    }
  }

  async _toggleCheckbox(this, event): Promise<void> {
    const item = event.model.item;
    let checkbox;
    let overflowCheckbox;
    if (item[1].value) {
      checkbox = this.shadowRoot?.getElementById(`checkbox-${event.model.index}`) as HTMLInputElement;
    } else {
      overflowCheckbox = this.shadowRoot?.getElementById(`overflowCheckbox-${event.model.index}`) as HTMLInputElement;
    }
    // 更新 selectColor 数组
    if (checkbox) {
      if (!checkbox.checked) {
        this.selectColor.push(item[1].value); // 添加选中的颜色
      } else {
        const index = this.selectColor.findIndex(
          (color) => color[0] === item[1].value[0] && color[1] === item[1].value[1],
        );
        if (index !== -1) {
          this.selectColor.splice(index, 1); // 取消选中的颜色
        }
      }
      if (this.selectColor.length === 0) {
        this.precisionmenu = [];
        this.set('selectedPrecisionNode', '');
        return;
      }
      const values = this.selectColor.map((item) => item === this.unMatchedNodeName ? -1 : item);
      const params = {
        metaData: this.selection,
        type: 'precision',
        values
      };
      const { success, data, error } = await request({ url: 'screen', method: 'POST', data: params });

      if (success) {
        this.set('precisionmenu', data);
        // 更新数据绑定
        this.notifyPath(`menu.${event.model.index}.checked`, checkbox.checked);
        // 清除精度筛选输入框
        this.set('selectedPrecisionNode', data?.[0] || '');
        // 选中第一个选项
        setTimeout(() => {
          this._observePrecsionNode();
        }, 200)
      }
      else {
        Notification.show(`Error:${error}`, {
          position: 'middle',
          duration: 4000,
          theme: 'error',
        });
      }
    } else {
      if (overflowCheckbox.checked) {
        this.overflowLevel.push(item[1]); // 添加选中的颜色
      } else {
        const index = this.overflowLevel.findIndex((overflow) => overflow === item[1]);
        if (index !== -1) {
          this.overflowLevel.splice(index, 1); // 取消选中的颜色
        }
      }
      if (this.overflowLevel.length === 0) {
        this.overflowmenu = [];
        return;
      }

      const params = {
        metaData: this.selection,
        type: 'overflow',
        values: this.overflowLevel,
      };
      const { success, data, error } = await request({ url: 'screen', method: 'POST', data: params });
      if (success) {
        this.set('overflowmenu', data);
        // 更新数据绑定
        this.notifyPath(`menu.${event.model.index}.checked`, overflowCheckbox.checked);
        // 清除精度筛选输入框
        this.set('selectedOverflowNode', data?.[0] || '');
        // 选中第一个选项
        setTimeout(() => {
          this._observeOverFlowNode();
        }, 200)
      }
      else {
        Notification.show(`Error:${error}`, {
          position: 'middle',
          duration: 4000,
          theme: 'error',
        });
      }
    }
  }

  _selectedTabChanged(): void {
    this.set('isOverflowFilter', !this.isOverflowFilter);
  }

  _observePrecsionNode = () => {
    if (!this.selectedPrecisionNode) {
      return;
    }
    let prefix = NPU_PREFIX;
    const node = prefix + this.selectedPrecisionNode;
    this.set('selectedNode', node);
  };

  _observeOverFlowNode = () => {
    if (!this.selectedOverflowNode) {
      return;
    }
    const prefix = this.isSingleGraph ? '' : NPU_PREFIX;
    const node = prefix + this.selectedOverflowNode;
    this.set('selectedNode', node);
  };

  _getColorDisplayLabel = (obj: any): string => {
    return obj.accuracy_level ?? obj.value;
  };
}
