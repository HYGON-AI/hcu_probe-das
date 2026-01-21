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

import '@vaadin/button';
import '@vaadin/details';
import '@vaadin/combo-box';

import { isEmpty } from 'lodash';
import { Notification } from '@vaadin/notification';
import { PolymerElement, html } from '@polymer/polymer';
import { customElement, property } from '@polymer/decorators';
import '@vaadin/progress-bar';
import i18next from 'i18next';
@customElement('tf-search-combox')
class Legend extends PolymerElement {
  // 定义模板
  static readonly template = html`
    <style>
      .search-arrow {
        font-size: 10px;
        margin-top: 31px;
        cursor: pointer;
        color: rgb(87, 86, 86);
        background: rgb(238, 238, 238);
        border: 0.5px solid var(--paper-input-container-color, var(--secondary-text-color));
        padding: 2px 2px;
        height: 30px;
        width: 22px;
      }
      .search-arrow:hover {
        background: rgb(201, 200, 199);
      }
      .container-search {
        display: flex;
        align-items: center;
        padding-right: 8px;
      }
      vaadin-combo-box {
        width: 100%;
        padding-bottom: 0;
      }
      vaadin-combo-box::part(input-field) {
        height: 30px;
        border: 1px solid var(--paper-input-container-color, var(--secondary-text-color));
        background-color: white;
        font-size: 14px;
        border-radius: 0;
      }
      vaadin-combo-box::part(toggle-button) {
        font-size: 14px;
      }
      .search-combox {
        display: flex;
        align-items: center;
        width: 100%;
      }
    </style>
    <div class="search-combox">
      <vaadin-combo-box
        label="[[label]]"
        items="[[items]]"
        value="{{selectedValue}}"
        on-change="_onChange"
        renderer="[[_comboBoxRenderer]]"
      ></vaadin-combo-box>
      <vaadin-icon
        title="[[t('search_previous')]]"
        icon="vaadin:arrow-up"
        class="search-arrow"
        on-click="_selectPrevious"
      ></vaadin-icon>
      <vaadin-icon
        title="[[t('search_next')]]"
        icon="vaadin:arrow-down"
        class="search-arrow"
        on-click="_selectNext"
      ></vaadin-icon>
    </div>
  `;
  @property({ type: Object })
  t: Function = (key) => i18next.t(key);

  @property({ type: Object })
  onSelectChange!: () => void;

  @property({ type: String, notify: true })
  selectedValue: string = '';

  @property({ type: Array })
  items: string[] = [];

  @property({ type: Boolean })
  isCompareGraph: boolean = true;

  _comboBoxRenderer(root, owner, model) {
    const item = model.item;
    const text = item.name || item.label || item; // 根据你的数据结构调整
    
    // 创建内容
    if (!root.firstElementChild) {
      const content = document.createElement('vaadin-item');
      content.setAttribute('title', text); // 添加 tooltip
      root.appendChild(content);
      
      const span = document.createElement('span');
      span.textContent = text;
      span.style.cssText = `
        display: block;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      `;
      content.appendChild(span);
    } else {
      const content = root.firstElementChild;
      content.setAttribute('title', text);
      const span = content.firstElementChild;
      span.textContent = text;
    }
  }

  _onChange(): void {
    this.onSelectChange();
  }

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

  // 选择列表中的下一个节点
  _selectNext(): void {
    if (!this.isCompareGraph) {
      Notification.show(this.t('build_not_support_match'), {
        position: 'middle',
        duration: 2000,
        theme: 'contrast',
      });
      return;
    }

    if (isEmpty(this.items)) {
      Notification.show(this.t('list_empty'), {
        position: 'middle',
        duration: 2000,
        theme: 'contrast',
      });
      return;
    }
    if (isEmpty(this.selectedValue)) {
      this.set('selectedValue', this.items[0]);
      this.onSelectChange();
      return;
    }
    const index = this.items.indexOf(this.selectedValue);
    if (index + 1 >= this.items.length) {
      Notification.show(this.t('bottom_of_list'), {
        position: 'middle',
        duration: 2000,
        theme: 'contrast',
      });
      return;
    } else {
      this.set('selectedValue', this.items[index + 1]);
    }
    this.onSelectChange();
  }

  // 选择列表中的上一个节点
  _selectPrevious(): void {
    if (!this.isCompareGraph) {
      Notification.show(this.t('build_not_support_match'), {
        position: 'middle',
        duration: 2000,
        theme: 'contrast',
      });
      return;
    }

    if (isEmpty(this.items)) {
      Notification.show(this.t('list_empty'), {
        position: 'middle',
        duration: 2000,
        theme: 'contrast',
      });
      return;
    }
    if (isEmpty(this.selectedValue)) {
      this.set('selectedValue', this.items[0]);
      this.onSelectChange();
      return;
    }
    const index = this.items.indexOf(this.selectedValue);
    if (index - 1 < 0) {
      Notification.show(this.t('top_of_list'), {
        position: 'middle',
        duration: 2000,
        theme: 'contrast',
      });
      return;
    } else {
      this.set('selectedValue', this.items[index - 1]);
    }
    this.onSelectChange();
  }
}
