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
import '@vaadin/select';
import '@vaadin/text-field';
import { NPU_PREFIX, BENCH_PREFIX } from '../../../common/constant';
import { PolymerElement, html } from '@polymer/polymer';
import { customElement, property, observe } from '@polymer/decorators';
import '@vaadin/progress-bar';
import '../tf_search_combox/index';
@customElement('tf-linkage-search-combox')
class Legend extends PolymerElement {
  // 定义模板
  static readonly template = html`
    <style>
      :host {
        --select-border-color: #3b5998;
      }
      .condition {
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
      }

      vaadin-combo-box {
        flex: 1;
        font-size: small;
        width: 100%;
      }

      vaadin-text-field input {
        min-height: 0;
      }

      .result vaadin-combo-box {
        padding-top: 0;
      }
      .vaadin-field-container {
        width: 100%;
      }
      vaadin-combo-box::part(input-field) {
        background-color: white;
        border: 1px solid #0d0d0d;
        height: 30px;
        border-radius: 0;
      }

      vaadin-text-field::part(input-field) {
        background-color: white;
        border: 1px solid #0d0d0d;
        height: 30px;
        border-radius: 0;
        font-size: 14px;
      }

      vaadin-select {
        width: 100px;
      }

      vaadin-select::part(input-field) {
        background-color: white;
        border: 1px solid var(--select-border-color);
        height: 30px;
        border-radius: 0;
      }

      vaadin-select-item {
        font-size: smaller;
      }
    </style>
    <div class="control-holder">
      <div class="condition">
        <vaadin-select
          label="[[t('data_side')]]"
          readonly="[[!isCompareGraph]]"
          items="[[menuSideItem]]"
          value="{{selectedSide}}"
        ></vaadin-select>
        <vaadin-text-field
          label="[[t('search_node')]]"
          value="{{searchText}}"
          on-change="_onChangeSearchText"
          clear-button-visible
        ></vaadin-text-field>
      </div>
      <div class="result">
        <tf-search-combox
          label="[[t('node_list')]]([[menuItem.length]])"
          items="[[menuItem]]"
          selected-value="{{selectedMenuNode}}"
          on-select-change="[[_onSelectedMenuNode]]"
        ></tf-search-combox>
      </div>
    </div>
  `;
  @property({ type: Object })
  t: Function = () => '';
  @property({ type: String, notify: true })
  selectedNode = '';

  @property({ type: Boolean })
  isCompareGraph: boolean = true;

  @property({ type: Object })
  nodelist;

  @property({ type: String })
  selectedMenuNode = '';

  @property({ type: Array })
  menuItem = [];

  @property({ type: String })
  searchText = '';

  @property({ type: String })
  selectedSide = 'npu';

  @property({ type: Array })
  menuSideItem = [
    { label: this.t('debug'), value: 'npu' },
    { label: this.t('bench'), value: 'bench' },
  ];

  @observe('selectedSide')
  _observeSelectSide(): void {
    if (this.nodelist) {
      this.set('menuItem', this.nodelist[this.selectedSide] || []);
      this.set('searchText', '');
      this.set('selectedMenuNode', '');
    }
  }
  @observe('t')
  _observeT(): void {
    if (this.t) {
      this.set('menuSideItem', [
        { label: this.t('debug'), value: 'npu' },
        { label: this.t('bench'), value: 'bench' },
      ]);
    }
  }


  @observe('nodelist')
  _observeMenu(): void {
    this.set('selectedMenuNode', '');
    this.set('selectedSide', 'npu');
    this.set('searchText', '');
    this.set('menuItem', this.nodelist[this.selectedSide] || []);
  }

  @observe('isCompareGraph')
  _observeRenderHierarchy(): void {
    this.updateStyles({ '--select-border-color': this.isCompareGraph ? '#0d0d0d' : 'white' });
  }

  _onSelectedMenuNode = (): void => {
    let prefix = '';
    if (this.isCompareGraph) {
      if (this.selectedSide === 'npu') {
        prefix = NPU_PREFIX;
      } else {
        prefix = BENCH_PREFIX;
      }
    }
    const node = prefix + this.selectedMenuNode;
    this.set('selectedNode', node);
  };

  _onChangeSearchText(): void {
    const allNodeItems = this.nodelist[this.selectedSide] as Array<string>;
    if (!this.searchText) {
      this.set('menuItem', allNodeItems);
      return;
    }
    const searchTextLower = this.searchText.trim().toLowerCase(); // 将搜索文本转换为小写
    const filterItem = allNodeItems?.filter((item: string) => {
      return item.toLowerCase().indexOf(searchTextLower) !== -1; // 将目标文本转换为小写
    });
    this.set('selectedMenuNode', '');
    this.set('menuItem', filterItem);
  }
}
