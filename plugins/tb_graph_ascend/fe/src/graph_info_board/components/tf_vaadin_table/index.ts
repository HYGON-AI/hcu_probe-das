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

import { PolymerElement, html } from '@polymer/polymer';
import { customElement, property, query } from '@polymer/decorators';
import '@vaadin/grid'; // 引入新的 Vaadin Grid 组件
import '@vaadin/tooltip';
import type { GridEventContext } from '@vaadin/grid';
import { escapeHTML } from '../../../utils';
import i18next from 'i18next';
@customElement('tf-vaadin-table')
class TfVaadinTable extends PolymerElement {
  static readonly template = html`
    <style>
      :host {
        display: block;
      }

      .vaadin-grid {
        height: 100%;
        font-size: 14px;
      }

      vaadin-grid-cell-content {
        height: 100%;
        display: flex;
        align-items: center;
      }

      vaadin-grid::part(header-cell) {
        border-bottom: 1px solid rgb(66, 66, 66);
      }

      .highlight-cell {
        border: 1px solid #005fdb;
        border-radius: 4px;
      }
      .no-data {
        font-size: 14px;
        color: #999;
      }
      .avater-matched:before {
        content: '';
        display: inline-block;
        width: 10px;
        height: 8px;
        background: #52c41a;
        padding-top: 2px;
        margin-right: 10px;
        border-radius: 50%;
      }
      .avater-unmatched:before {
        content: '';
        display: inline-block;
        width: 10px;
        height: 8px;
        background: #ff4d4f;
        padding-top: 2px;
        margin-right: 10px;
        border-radius: 50%;
      }
      .splitter {
        border-bottom: 1px solid rgb(66, 66, 66);
      }
    </style>
    <template is="dom-if" if="[[!isEmptyGrid]]">
      <vaadin-grid
        id="grid"
        on-click="handleGridClick"
        items="[[ioDataset]]"
        class="vaadin-grid"
        theme="force-outline no-border"
      >
        <!-- 动态生成列 -->
        <template is="dom-repeat" items="[[headers]]" as="header">
          <vaadin-grid-column
            path="[[header]]"
            header="[[header]]"
            resizable
            renderer="[[renderDefaultValue]]"
          ></vaadin-grid-column>
        </template>
      </vaadin-grid>
    </template>
    <template is="dom-if" if="[[isEmptyGrid]]">
      <p class="no-data">[[t('no_iodata')]]</p>
    </template>
  `;

  @property({ type: Object })
  syncGrid?: HTMLElement; // 点击高亮需要同步的表格元素

  @property({ type: Boolean })
  isSingleGraphNode = false; // 是否是单节点图

  @property({ type: Object })
  tooltips: any;

  @property({ type: Object })
  handleCellClick!: (e: MouseEvent, syncGrid: HTMLElement) => void;

  @property({
    type: Array,
    computed: '_computeHeaders(ioDataset)',
  })
  headers!: any[];

  @property({
    type: Boolean,
    computed: '_isEmptyGrid(ioDataset)',
  })
  isEmptyGrid!: false;

  @property({ type: Object })
  t: Function = (key) => i18next.t(key);

  renderDefaultValue!: (root: HTMLElement, column: any, rowData: any) => void;

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

  override connectedCallback(): void {
    super.connectedCallback();
    this.renderDefaultValue = this._renderDefaultValue.bind(this);
  }

  /**
   * 计算表头（所有唯一的键）
   * @param {Array<Object>} data 数据源
   * @return {Array<string>} 表头数组
   */
  _computeHeaders(data): any[] {
    if (this._isEmptyGrid(data)) {
      return [];
    }
    const ignoreDataIndex = ['data_name', 'isBench', 'isMatched', 'value'];
    const headers = Array.from(
      data.reduce((keys, item) => {
        // 只取前5个数据项，避免性能问题
        Object.keys(item).forEach((key) => {
          if (!ignoreDataIndex.includes(key)) {
            keys.add(String(key));
          }
        });
        return keys;
      }, new Set()),
    );
    return headers;
  }

  _isEmptyGrid(data): boolean {
    return !Array.isArray(data) || data.length === 0;
  }

  _renderDefaultValue(root: HTMLElement, column: any, rowData: any): void {
    const selectedColor = this._getCssVariable('--selected-color');
    const matchedColor = this._getCssVariable('--matched-color');
    const isBench = 'isBench';
    const isMatched = 'isMatched';
    root.classList.remove('splitter');
    if (rowData.item[isBench]) {
      root.style.backgroundColor = matchedColor;
      if (rowData.item[isMatched]) {
        root.classList.add('splitter');
      }
    } else {
      root.style.backgroundColor = selectedColor;
    }
    if (column.path === 'name' && !this.isSingleGraphNode) {
      const className = rowData.item[isMatched] ? 'avater-matched' : 'avater-unmatched';
      root.innerHTML = `<span class='${className}'>${escapeHTML(rowData.item[column.path])}</span>`;
      return;
    }
    let tooltip = rowData.item[column.path] ?? '-';
    if (this.tooltips?.[column.path]) {
      tooltip = `${this.tooltips[column.path]}:\n${tooltip}`;
    }
    root.title = tooltip;
    root.textContent = rowData.item[column.path] ?? '-';
  }

  handleGridClick(e: MouseEvent): void {
    this.handleCellClick(e, this.syncGrid as HTMLElement); // 调用后方法的this会指向当前组件，无法拿到同级别的表格组件，所以需要回传
  }

  _getCssVariable(variableName): string {
    const computedStyle = getComputedStyle(this);
    return computedStyle.getPropertyValue(variableName).trim(); // 去掉多余的空格
  }
}
