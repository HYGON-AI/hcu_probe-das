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

import '@vaadin/combo-box';
import '@vaadin/text-field';
import * as _ from 'lodash';
import { PolymerElement, html } from '@polymer/polymer';
import { Notification } from '@vaadin/notification';
import { customElement, property, observe } from '@polymer/decorators';
import { NPU_PREFIX, UNMATCHED_COLOR, defaultColorSetting, defaultColorSelects } from '../../../common/constant';
import request from '../../../utils/request';
import { DarkModeMixin } from '../../../polymer/dark_mode_mixin';
import { LegacyElementMixin } from '../../../polymer/legacy_element_mixin';
import '../tf_filter_precision_error/index'
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
          padding: 10px 0;
          border-bottom: 1px solid var(--border-color);
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

        .search-number{
          display: inline-block;
          width: 80px;
          height: 14px;
          background-color: #fff;
          font-size: 14px;
          color: red;
          font-weight: bold;
          position: relative;
          top: 30px;
          left: 114px;
          z-index: 10;
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
                    <vaadin-icon icon="vaadin:funnel" on-click="_clickFilter"></vaadin-icon>
                    <vaadin-icon icon="vaadin:cog-o" on-click="_clickSetting"></vaadin-icon>
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
                <div class="run-dropdown" style="margin-top: 8px; display: flex; flex-direction: column;">
                  <template is="dom-repeat" items="[[colorSetChanged]]">
                    <div class="color-option" style="display: flex; align-items: center;">
                      <paper-checkbox id="checkbox-[[index]]" on-click="_toggleCheckbox"></paper-checkbox>
                      <div
                        style="width: 12px; height: 12px; background-color: [[item.0]]; margin-right: 8px; border: 1px solid gray;"
                      ></div>
                      [[item.1.value]]
                    </div>
                  </template>
                </div>
                <span class="search-number">([[precisionmenu.length]])</span>
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
                    <div class="run-dropdown" style="margin-top: 8px; display: flex; flex-direction: column;">
                      <template is="dom-repeat" items="[[overFlowSet]]">
                        <div class="color-option" style="display: flex; align-items: center;">
                          <paper-checkbox id="overflowCheckbox-[[index]]" on-click="_toggleCheckbox"></paper-checkbox>
                          <div
                            style="width: 12px; height: 12px; background-color: [[item.0]]; margin-right: 8px; border: 1px solid gray;"
                          ></div>
                          [[item.1]]
                        </div>
                      </template>
                    </div>
                  </div>
                </template>
                <span class="search-number">([[overflowmenu.length]])</span>
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
          <template is="dom-if" if="[[!_colorSetting]]">
            <div class="toolbar" id="colorSetting-toolbar" style="width: auto; cursor: default;">
              <span class="toggle-legend-text">
                [[t('color_setting')]]
                <div class="legend-clarifier">
                  <paper-tooltip animation-delay="0" position="right" offset="0">
                    <div class="custom-tooltip">
                      <p>精度区间左闭右开</p>
                      <p>精度区间至多5个档位</p>
                      <p>区间内输入至多可保留小数点后5位</p>
                      <p>不符合输入会被清空</p>
                    </div>
                  </paper-tooltip>
                </div>
              </span>
              <div style="display: flex; margin-left: auto;">
              <button on-click="_defaultSetting" style="margin-right: 7px;">
                  <span>[[t('preset_configuration')]]</span>
                </button>
                <button on-click="_confirmAction" style="margin-right: 7px;">
                  <span>[[t('confirm')]]</span>
                </button>
                <button on-click="_cancelAction">
                  <span>[[t('cancel')]]</span>
                </button>
              </div>
            </div>
            <iron-collapse opened="[[_colors]]" class="legend-content" id="colorSetting-content" style="height: 150px; padding-top: 10px;">
              <div style="display: flex; align-items: center">
                <div>[[t('color_selection')]]</div>
                <div id="left">[[t('left_interval')]]</div>
                <div id="right">[[t('right_interval')]]</div>
                <button id="addButton" on-click="_addOption">[[t('add_range')]]</button>
              </div>
              <!-- 动态生成的隐藏选项 -->
              <template is="dom-repeat" items="[[colorSelects]]" as="item">
                <div style="display: flex; align-items: center; margin-top: 2px;">
                  <!-- 自定义颜色选择框 -->
                  <div class="custom-select">
                    <div class="select-box" style="background-color: [[item.key]];" on-click="_toggleDropdown">
                      <span></span>
                    </div>
                    <div class="dropdown" hidden>
                      <template is="dom-repeat" items="[[colorList]]" as="color">
                        <div class="option" 
                          style="background-color: [[color]]" 
                          on-mouseover="_onOptionHover"
                          on-mouseout="_outOptionHover" 
                          on-click="_changeColor" 
                          value="[[color]]">
                        </div>
                      </template>
                    </div>
                  </div>
                  <div style="display: flex; margin-right: 10px;">
                    <!-- 输入框 左 -->
                    <input
                      id="input-left"
                      on-change="_validateInputs"
                      style="display: inline-block; width: 60px; margin-right: 10px;"
                      value="[[_formatValue(item.values.0)]]"
                    />
                    <!-- 输入框 右 -->
                    <input
                      id="input-right"
                      on-change="_validateInputs"
                      style="display: inline-block; width: 60px;"
                      value="[[_formatValue(item.values.1)]]"
                    />
                  </div>
                  <!-- 删除按钮 -->
                  <button on-click="_removeOption">-</button>
                </div>
              </template>
            </iron-collapse>
          </template>
        </div>
      </template>
      <tf-filter-precision-error t="[[t]]" filter-dialog-opened="{{filterDialogOpened}}" update-filter-data="{{updateFilterData}}" selection="[[selection]]"/>
    `;

  @property({ type: Object })
  t: Function = (key) => i18next.t(key);

  @property({ type: Boolean })
  _colorSetting: boolean = true; // 颜色设置按钮

  @property({ type: Boolean })
  filterDialogOpened: boolean = false;

  @property({ type: Boolean })
  isSingleGraph = false;

  @property({ type: Boolean })
  _overFlowLevel: boolean = true; // 溢出筛选图例

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

  @property({ type: Number, notify: true })
  dropdownIndex;

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
      const newSelectColor = this.colorSetChanged?.map((item) => {
        if (item[1].value === this.unMatchedNodeName) {
          return [item[0], { value: this.t('no_matching_nodes'), description: item[1].description }];
        }
        return item;
      });
      this.set('colorSetChanged', newSelectColor);
      this.set('unMatchedNodeName', this.t('no_matching_nodes'));
      this.set('precisionDesc', this.t(`precision_desc.${this.task}`));
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
    this.set('precisionDesc', this.t(`precision_desc.${this.task}`));
  }

  // 写一个如果切换数据清除所有checkbox和所有this.selectColor
  @observe('selection')
  _clearAllToggleCheckboxAndInputField(): void {
    this.set('selectedSide', '0');
    const allCheckboxes = this.shadowRoot?.querySelectorAll('paper-checkbox');
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

  toggleVisibility(): void {
    this.set('_colorSetting', !this._colorSetting);
  }

  _clickFilter(event): void {
    event.stopPropagation();
    this.set('filterDialogOpened', true);
  }

  _clickSetting(event): void {
    event.stopPropagation();
    this.set('_colors', true);
    this.toggleVisibility();
  }

  _defaultSetting(): void {
    // 配置预设
    this.colorSelects = defaultColorSetting;
    this._confirmAction();
    // 清空并且还原至临时配置结构
    this.colorSelects = defaultColorSelects;
  }

  _cancelAction(): void {
    this.toggleVisibility();
  }

  async _confirmAction(): Promise<void> {
    const newColorsList = {};
    const len = this.colorSelects.length;
    if (len === 0) {
      this.showDynamicDialog(this.t('configuration_items_missing'));
      return;
    }

    // 遍历每一项，动态生成 newColorsList 对象
    for (let i = 0; i < len; i++) {
      const color = this.colorSelects[i].key;
      const leftValue = this.colorSelects[i].values[0];
      const rightValue = this.colorSelects[i].values[1];
      // 检查每个组中的所有输入框是否都有值
      if (isNaN(leftValue) || isNaN(rightValue) || color === 'NaN') {
        this.showDynamicDialog(this.t('unconfigured_items_exist'));
        return;
      }
      // 将每个 color 和其对应的 leftValue 和 rightValue 作为 value 数组，设置到 colors 对象中
      newColorsList[color] = {
        value: [leftValue, rightValue],
        description:
          '此节点所有输入输出的统计量相对误差，值越大代表测量值与标杆值的偏差越大，相对误差计算方式：|(测量值-标杆值)/标杆值|',
      };
    }
    // 无匹配节点图例一定存在
    newColorsList[UNMATCHED_COLOR] = {
      value: this.unMatchedNodeName,
      description: '对比过程中节点未匹配上',
    };
    const params = {
      colors: JSON.stringify(newColorsList),
      metaData: this.selection,
    };
    const { success, data, error } = await request({ url: 'updateColors', method: "POST", data: params });
    if (success) {
      // 更新颜色列表

      this.set('colors', newColorsList);
      let newColorSetChanged: any[] = [];
      this.toggleVisibility();
      Object.entries(newColorsList).forEach(([color, details]) => {
        let detailsTyped = details as { value: string };
        const colorset: any[] = [color, detailsTyped];
        newColorSetChanged.push(colorset);
      });
      this.set('colorSetChanged', newColorSetChanged);
    } else {
      this.showDynamicDialog(error);
    }
  }

  _toggleDropdown(event): void {
    const selectBox = event.target.closest('.select-box'); // 获取最近的父元素 .select-box
    const dropdown = selectBox.nextElementSibling; // 获取下一个兄弟元素，即 .dropdown
    dropdown.hidden = !dropdown.hidden;
    this.dropdownIndex = event.model.index;
    function maybeCloseMenu(eventCloseMenu?: any): void {
      if (eventCloseMenu?.composedPath().includes(selectBox)) {
        return;
      }
      dropdown.hidden = true;
      document.body.removeEventListener('click', maybeCloseMenu, {
        capture: true,
      });
    }
    if (!dropdown.hidden) {
      document.body.addEventListener('click', maybeCloseMenu, {
        capture: true,
      });
    }
  }

  _onOptionHover(event): void {
    event.target.style.border = 'solid 1px black';
  }

  _outOptionHover(event): void {
    event.target.style.border = '';
  }

  _changeColor(event): void {
    const dropdown = event.target.closest('.dropdown');
    const select = dropdown.previousElementSibling;
    dropdown.hidden = true;
    const selectedColor = event.target.value;
    select.style.backgroundColor = selectedColor;
    this.set(`colorSelects.${this.dropdownIndex}.key`, selectedColor);
    this.notifyPath('colorSelects');
    this._setColorList();
  }

  // 不显示NaN 而显示空
  _formatValue(value): string {
    return isNaN(value) ? '' : value;
  }

  _validateInputs(event: any): void {
    const index = event.model.index;
    const { values } = this.colorSelects[index];

    // 显式定义 leftInputSet 和 rightInputSet 的类型为 number[]
    const [leftInputSet, rightInputSet] = this.colorSelects.reduce<[number[], number[]]>(
      (acc, item) => {
        acc[0].push(item.values[0]);
        acc[1].push(item.values[1]);
        return acc;
      },
      [[], []], // 初始值为两个空数组
    );

    let value = parseFloat(event.target.value);
    // 输入值验证 NaN值防护 限制输入范围
    if (isNaN(value) || value < 0 || value > 1) {
      this._clearInput(event, index);
      return;
    }

    const valueStr = value.toString();

    // 检查是否存在小数点
    const parts = valueStr.split('.');

    // 如果存在小数点且小数部分长度超过最大限制
    if (parts.length > 1 && parts[1].length > 5) {
      // 使用 toFixed 保留最多5位小数
      value = parseFloat(value.toFixed(5));
    }

    const isLeftInput = event.target.id === 'input-left';
    const otherSide = isLeftInput ? values[1] : values[0];
    const [left, right] = isLeftInput ? [value, otherSide] : [otherSide, value];

    // 检查输入值是否有效
    const isLeftInputGreater = isLeftInput && left > right;
    const isRightInputGreater = !isLeftInput && right < left;

    if (isLeftInputGreater || isRightInputGreater) {
      this._clearInput(event, index);
      return;
    }

    // 检查输入值是否与其他区间冲突
    const isConflict = this.colorSelects.some((item, i) => {
      // 排除当前输入框
      if (i === index) {
        return false;
      }

      const [leftInput, rightInput] = item.values;
      return (
        (isLeftInput && left !== leftInput && left >= leftInput && left < rightInput) ||
        (!isLeftInput && right !== rightInput && right > leftInput && right <= rightInput) ||
        (isLeftInput && leftInputSet.includes(left)) ||
        (!isLeftInput && rightInputSet.includes(right))
      );
    });

    if (isConflict) {
      this._clearInput(event, index);
      return;
    }

    // 0！@#￥ 也可以被float转换为0，阻止这种情况发生
    event.target.value = value;
    // 更新值
    this.set(`colorSelects.${index}.values.${isLeftInput ? 0 : 1}`, value);
  }

  _clearInput(event: any, index: number): void {
    event.target.value = ''; // 清空输入框
    this.set(`colorSelects.${index}.values.${event.target.id === 'input-left' ? 0 : 1}`, NaN); // 更新 colorSelects
  }

  _addOption(): void {
    if (this.colorSelects.length < 5) {
      const obj = {
        key: 'NaN',
        values: [NaN, NaN],
      };
      this.push('colorSelects', obj);
    }
    // 确保它在当前同步操作this.push()之后才执行.
    this.async(() => {
      this._setColorList();
    }, 0);
  }

  _removeOption(event): void {
    const index = event.model.index;

    // 删除项
    this.splice('colorSelects', index, 1);

    // 恢复其他输入框的值
    this.colorSelects.forEach((item, i) => {
      if (i >= index) {
        this.set(`colorSelects.${i}.values`, item.values);
      }
    });
    this._setColorList();
  }

  _setColorList(): void {
    let colorSelectElements = this.shadowRoot?.querySelectorAll('[id^="color-select"]');
    let backgroundColors: string[] = [];
    this.colorSelects.forEach((item) => {
      // 获取计算后的背景色
      const backgroundColor = item.key;
      backgroundColors.push(backgroundColor);
    });
    let newColorList = this.standardColorList.filter((color) => !backgroundColors.includes(color));
    this.set('colorList', newColorList);
    // 清除选中，否则再次选中不同列表的同一顺位的值的时候不会触发on-change
    this.async(() => {
      colorSelectElements?.forEach((element) => {
        if (element instanceof HTMLSelectElement) {
          element.selectedIndex = -1;
        }
      });
    }, 0);
  }

  _toggleOverflowLevelOpen(): void {
    this.set('_overFlowLevel', !this._overFlowLevel);
  }

  showDynamicDialog(message): void {
    // 检查是否已经有显示的对话框，避免重复添加
    let existingDialog = this.shadowRoot?.querySelector('#dynamicDialog');
    if (existingDialog) {
      existingDialog.remove(); // 删除旧的对话框
    }
    // 创建新的对话框
    const dialog = document.createElement('paper-dialog');
    dialog.id = 'dynamicDialog';
    // 添加标题
    const title = document.createElement('h2');
    title.textContent = 'Info';
    this.titleElements.add(title);
    dialog.appendChild(title);
    // 添加提示内容
    const content = document.createElement('div');
    content.textContent = message;
    dialog.appendChild(content);
    // 添加按钮
    const buttonContainer = document.createElement('div');
    buttonContainer.classList.add('buttons');
    const closeButton = document.createElement('paper-button');
    closeButton.setAttribute('dialog-dismiss', '');
    closeButton.textContent = 'close';
    this.closeButtonElements.add(closeButton);
    buttonContainer.appendChild(closeButton);
    dialog.appendChild(buttonContainer);
    // 添加到 shadow DOM
    this.shadowRoot?.appendChild(dialog);
    // 打开对话框
    dialog.open();
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
      if (checkbox.checked) {
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
}
