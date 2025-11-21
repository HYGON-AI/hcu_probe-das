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
import { customElement, property } from '@polymer/decorators';
import '@vaadin/grid'; // 引入新的 Vaadin Grid 组件
import '@vaadin/tooltip';
import type { GridEventContext } from '@vaadin/grid';
import { Notification } from '@vaadin/notification';
import i18next from 'i18next';
@customElement('tf-vaadin-text-table')
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
      .no-data {
        font-size: 14px;
        color: #999;
      }
      .copyable-input {
        gap: 8px;
        width: 100%;
        height: 260px;
        font-family: Roboto, sans-serif;
        font-weight: 400;
      }

      .copyable-input textarea {
        flex-grow: 1;
        width: 100%;
        height: 80%;
        font-family: Roboto, sans-serif;
        box-sizing: border-box;
        resize: none;
        padding: 8px;
        font-size: 14px;
        border: 1px solid #ddd;
        border-radius: 4px;
        background: #f9f9f9;
        overflow-wrap: break-word; /* 自动换行 */
        white-space: pre-wrap; /* 保留换行符 */
      }

      .copy-button {
        padding: 4px 8px;
        font-size: 12px;
        background: rgb(117, 122, 128);
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        position: relative;
        right: 58px;
        bottom: 180px;
      }
      .copy-button:hover {
        background: #0056b3;
      }
    </style>
    <template is="dom-if" if="[[!isEmptyGrid]]">
      <vaadin-grid id="grid" items="[[dataset]]" class="vaadin-grid" theme="force-outline">
        <!-- 动态生成列 -->
        <template is="dom-repeat" items="[[headers]]" as="header">
          <vaadin-grid-column path="[[header]]" header="[[header]]" resizable renderer="[[renderDefaultValue]]">
            <template> [[item[header]]] </template>
          </vaadin-grid-column>
        </template>
        <vaadin-tooltip slot="tooltip" generator="[[tooltipGenerator]]"></vaadin-tooltip>
      </vaadin-grid>
    </template>
    <template is="dom-if" if="[[isEmptyGrid]]">
      <p class="no-data">[[t('no_data')]]</p>
    </template>
  `;

  @property({ type: Object })
  syncGrid!: HTMLElement; // 点击高亮需要同步的表格元素

  @property({ type: Object })
  handleCellClick!: (e: MouseEvent, syncGrid: HTMLElement) => void;

  @property({
    type: Array,
    computed: '_computeHeaders(dataset)',
  })
  headers: any[] = [];

  @property({
    type: Boolean,
    computed: '_isEmptyGrid(dataset)',
  })
  isEmptyGrid: boolean = false;

  @property({ type: Object })
  t: Function = (key) => i18next.t(key);

  renderDefaultValue!: (root: HTMLElement, column: any, rowData: any) => void;
  tooltipGenerator!: (context: GridEventContext<Record<string, string>>) => string;

  private copyButtonElements: Set<HTMLButtonElement> = new Set();

  constructor() {
    super();
    this.setupLanguageListener();
  }

  setupLanguageListener() {
    i18next.on('languageChanged', () => {
      // 更新语言后重新渲染
      const t = this.t;
      this.set('t', null);
      this.set('t', t);
      // 更新copy按钮文本
      this.copyButtonElements.forEach(button => {
          button.textContent = this.t('copy');
      });
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
    const ignoreDataIndex = ['title'];
    const headers = Array.from(
      data.slice(0, 5).reduce((keys, item) => {
        // 只取前5个数据项，避免性能问题
        Object.keys(item).forEach((key) => {
          if (!ignoreDataIndex.includes(key)) {
            keys.add(key);
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
    const propertyName = column.path;
    const titleName = rowData.item.title;
    if (!root.firstElementChild) {
      switch (titleName) {
        case 'stackInfo':
        case 'suggestions':
        case 'parallelMergeInfo':
          this._createCopyableTextarea(root, propertyName, rowData);
          break;
        default:
          root.style.fontWeight = 'bold';
          root.textContent = `${titleName}：${rowData.item[propertyName] || 'null'}`;
          break;
      }
    } else {
      switch (titleName) {
        case 'stackInfo':
        case 'suggestions':
        case 'parallelMergeInfo':
          this._updateCopyableTextarea(root, propertyName, rowData);
          break;
        default:
          root.textContent = `${titleName}：${rowData.item[propertyName] || 'null'}`;
      }
    }
  }

  _createCopyableTextarea(root: HTMLElement, propertyName: any, rowData: any): void {
    const container = document.createElement('div');
    container.className = 'copyable-input';

    const title = document.createElement('div');
    const textTitle = 'title';
    title.className = 'copyable-input-title';
    title.style.fontWeight = 'bold';
    title.style.marginTop = '8px';
    title.style.marginBottom = '8px';
    title.style.fontSize = '14px';
    title.textContent = `${rowData.item[textTitle]}:`;
    container.appendChild(title);

    const textarea = document.createElement('textarea');
    textarea.readOnly = true;
    textarea.value = rowData.item[propertyName] || 'null';
    textarea.onmouseenter = () => {
      button.style.display = 'unset';
    };
    textarea.onmouseleave = () => {
      button.style.display = 'none';
    };
    container.appendChild(textarea);

    const button = document.createElement('button');
    button.className = 'copy-button';
    button.textContent = 'copy';
    button.style.display = 'none';
    button.onmousemove = () => {
      button.style.display = 'unset';
    };
    button.onclick = () => {
      navigator.clipboard
        .writeText(textarea.value)
        .then(() => {
          Notification.show(this.t('copy_success'), {
            position: 'middle',
            duration: 1000,
            theme: 'success',
          });
        })
        .catch((err) => {
          Notification.show(this.t('copy_fail'), {
            position: 'middle',
            duration: 1000,
            theme: 'error',
          });
        });
    };
    container.appendChild(button);
    this.copyButtonElements.add(button);
    root.appendChild(container);
  }

  _updateCopyableTextarea(root: HTMLElement, propertyName: any, rowData: any): void {
    const title = root.querySelector('.copyable-input-title');
    const textTitle = 'title';
    if (title) {
      title.textContent = `${rowData.item[textTitle]}:`;
    }
    const textarea = root.querySelector('textarea');
    if (textarea) {
      textarea.value = rowData.item[propertyName] || 'null';
    }
  }

  _tooltipGenerator = (context: GridEventContext<Record<string, string>>): string => {
    const { column, item } = context;
    return item?.[column?.path || ''] || '';
  };

  handleGridClick(e: MouseEvent): void {
    this.handleCellClick(e, this.syncGrid); // 调用后方法的this会指向当前组件，无法拿到同级别的表格组件，所以需要回传
  }
}
