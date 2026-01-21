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

import { PolymerElement, html } from '@polymer/polymer';
import { customElement, property } from '@polymer/decorators';
import i18next from '../../../common/i18n';
@customElement('scene-legend')
class Legend extends PolymerElement {
  static readonly template = html`
      <style>
        :host {
          --legend-border-color:rgb(99, 99, 99);
          --legend-fill-color: rgb(230, 230, 230);
        }
        .legend {
          display: flex;
          justify-content: center;
          align-items: center;
          background: #fff;
          height: 40px;
          border-bottom: 1px solid #e0e0e0;
          position: relative;
        }
        .legend-item {
          margin-right: 10px;
          display: flex;
          align-items: center;
        }
        .legend-item-value {
          margin-left: 5px;
          font-size: 12px;
        }
        .legend-clarifier {
          color: #266236;
          cursor: pointer;
          height: 16px;
          width: 16px;
          margin-left: 4px;
          display: inline-block;
          text-decoration: underline;
          background: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDIwMDAgMjAwMCI+PHBhdGggZmlsbD0iIzc5Nzk3OSIgZD0iTTk3MSAxMzY4cTE4IDAgMzEgMTIgMTIgMTMgMTIgMzF2MTQzcTAgMTgtMTIgMzEtMTMgMTItMzEgMTJIODI3cS0xNyAwLTMwLTEyLTEzLTEzLTEzLTMxdi0xNDNxMC0xOCAxMy0zMSAxMy0xMiAzMC0xMmgxNDR6bTEyMi03NDJxODYgNDMgMTM4IDExNSA1MiA3MSA1MiAxNjEgMCA5Ny01NyAxNjUtMzYgNDAtMTE5IDg3LTQzIDI1LTU3IDM5LTI1IDE4LTI1IDQzdjU5SDc3NHYtNzBxMC04MiA1Ny0xNDAgMzItMzIgMTA4LTc1bDctNHE1NC0zMiA3NS01MCAyNS0yNSAyNS01NyAwLTQzLTQ0LTcyLTQ1LTI5LTEwMS0yOXQtOTUgMjVxLTMyIDIyLTc5IDgzLTExIDExLTI3IDE0LTE2IDQtMzAtN2wtMTAxLTc1cS0xNC0xMS0xNi0yOXQ1LTMycTY4LTk3IDE1NS0xNDYgODYtNDggMjA4LTQ4IDg2IDAgMTcyIDQzem01NzYgMjlxMTIxIDIwNCAxMjEgNDQ1IDAgMjQwLTEyMSA0NDUtMTIwIDIwNC0zMjUgMzI1LTIwNCAxMjAtNDQ1IDEyMC0yNDAgMC00NDUtMTIwLTIwNC0xMjEtMzI1LTMyNVE5IDEzNDAgOSAxMTAwcTAtMjQxIDEyMC00NDUgMTIxLTIwNSAzMjUtMzI1IDIwNS0xMjEgNDQ1LTEyMSAyNDEgMCA0NDUgMTIxIDIwNSAxMjAgMzI1IDMyNXptLTE0OSA4MDRxOTctMTY1IDk3LTM1OXQtOTctMzU5cS05Ny0xNjUtMjYyLTI2MnQtMzU5LTk3cS0xOTQgMC0zNTkgOTdUMjc4IDc0MXEtOTcgMTY1LTk3IDM1OXQ5NyAzNTlxOTcgMTY1IDI2MiAyNjJ0MzU5IDk3cTE5NCAwIDM1OS05N3QyNjItMjYyeiIvPjwvc3ZnPg==');
        }

        .legend-clarifier paper-tooltip {
          width: 150px;
          display: flex;
          align-items: center;
        }

        .custom-tooltip {
          font-size: 14px;
        }

        /* 小键盘图标按钮样式 */
        .keyboard-button {
          position: absolute;
          right: 20px;
          top: 50%;
          transform: translateY(-50%);
          cursor: pointer;
          background: transparent;
          border: none;
          width: 24px;
          height: 24px;
          padding: 4px;
          border-radius: 4px;
          transition: background-color 0.2s;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        
        .keyboard-button:hover {
          background-color: rgba(38, 98, 54, 0.1);
        }
        
        .keyboard-icon {
          width: 18px;
          height: 18px;
          background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTgiIGhlaWdodD0iMTgiIHZpZXdCb3g9IjAgMCAxOCAxOCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTIuMjUgMy4zNzVIMTUuNzVWMTRIMi4yNVYzLjM3NVpNMi4yNSAyLjA2MjVDMS44NjI1IDIuMDYyNSAxLjUgMi4zMjUgMS41IDIuNjg3NVYxNC42ODc1QzEuNSAxNS4wNTEzIDEuODYyNSAxNS4zMTI1IDIuMjUgMTUuMzEyNUgxNS43NUMxNi4xMzc1IDE1LjMxMjUgMTYuNSAxNS4wNTEzIDE2LjUgMTQuNjg3NVYyLjY4NzVDMTYuNSAyLjMyNSAxNi4xMzc1IDIuMDYyNSAxNS43NSAyLjA2MjVIMi4yNVoiIGZpbGw9IiM3OTc5NzkiLz4KPHBhdGggZD0iTTUgNS42ODc1SDYuMzc1VjcuMDYyNUg1VjUuNjg3NVoiIGZpbGw9IiM3OTc5NzkiLz4KPHBhdGggZD0iTTcuMDYyNSA1LjY4NzVIOC40Mzc1VjcuMDYyNUg3LjA2MjVWNS42ODc1WiIgZmlsbD0iIzc5Nzk3OSIvPgo8cGF0aCBkPSJNOS4xMjUgNS42ODc1SDEwLjVWNy4wNjI1SDkuMTI1VjUuNjg3NVoiIGZpbGw9IiM3OTc5NzkiLz4KPHBhdGggZD0iTTExLjE4NzUgNS42ODc1SDEyLjU2MjVWNy4wNjI1SDExLjE4NzVWNS42ODc1WiIgZmlsbD0iIzc5Nzk3OSIvPgo8cGF0aCBkPSJNNSA4LjA2MjVINi4zNzVWOS40Mzc1SDVWOC4wNjI1WiIgZmlsbD0iIzc5Nzk3OSIvPgo8cGF0aCBkPSJNNy4wNjI1IDguMDYyNUg4LjQzNzVWOS40Mzc1SDcuMDYyNVY4LjA2MjVaIiBmaWxsPSIjNzk3OTc5Ii8+CjxwYXRoIGQ9Ik05LjEyNSA4LjA2MjVIMTAuNVY5LjQzNzVIOS4xMjVWOC4wNjI1WiIgZmlsbD0iIzc5Nzk3OSIvPgo8cGF0aCBkPSJNMTEuMTg3NSA4LjA2MjVIMTIuNTYyNVY5LjQzNzVIMTEuMTg3NVY4LjA2MjVaIiBmaWxsPSIjNzk3OTc5Ii8+CjxwYXRoIGQ9Ik01IDEwLjQzNzVINi4zNzVWMTEuODEyNUg1VjEwLjQzNzVaIiBmaWxsPSIjNzk3OTc5Ii8+CjxwYXRoIGQ9Ik03LjA2MjUgMTAuNDM3NUg4LjQzNzVWMTEuODEyNUg3LjA2MjVWMTAuNDM3NVoiIGZpbGw9IiM3OTc5NzkiLz4KPHBhdGggZD0iTTkuMTI1IDEwLjQzNzVIMTAuNVYxMS44MTI1SDkuMTI1VjEwLjQzNzVaIiBmaWxsPSIjNzk3OTc5Ii8+CjxwYXRoIGQ9Ik0xMS4xODc1IDEwLjQzNzVIMTIuNTYyNVYxMS44MTI1SDExLjE4NzVWMTAuNDM3NVoiIGZpbGw9IiM3OTc5NzkiLz4KPHBhdGggZD0iTTEzLjUgNS42ODc1SDE1LjA2MjVWNy4wNjI1SDEzLjVWNS42ODc1WiIgZmlsbD0iIzc5Nzk3OSIvPgo8cGF0aCBkPSJNMTMuNSAxMC40Mzc1SDE1LjA2MjVWMTEuODEyNUgxMy41VjEwLjQzNzVaIiBmaWxsPSIjNzk3OTc5Ii8+Cjwvc3ZnPgo=');
          background-size: contain;
          background-repeat: no-repeat;
          background-position: center;
        }
        
        /* 快捷键弹框样式 */
        .shortcut-panel {
          position: absolute;
          right: 20px;
          top: 45px;
          background: white;
          border: 1px solid #e0e0e0;
          border-radius: 6px;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
          z-index: 1000;
          min-width: 200px;
          padding: 16px;
          display: none;
        }
        
        .shortcut-panel.show {
          display: block;
        }
        
        .shortcut-panel::before {
          content: '';
          position: absolute;
          top: -6px;
          right: 20px;
          width: 12px;
          height: 12px;
          background: white;
          transform: rotate(45deg);
          border-left: 1px solid #e0e0e0;
          border-top: 1px solid #e0e0e0;
        }
        
        .shortcut-header {
          font-weight: bold;
          color: #333;
          margin-bottom: 12px;
          padding-bottom: 8px;
          border-bottom: 1px solid #eee;
          font-size: 14px;
        }
        
        .shortcut-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin: 8px 0;
          font-size: 13px;
        }
        
        /* 修改描述文字为黑色 */
        .shortcut-action {
          color: #000000; /* 改为黑色 */
        }
        
        /* 修改按键文字为浅灰色 */
        .shortcut-key {
          color: #999999; /* 改为浅灰色 */
          font-weight: bold;
          font-family: 'Courier New', monospace;
          min-width: 60px;
          text-align: center;
        }

        .module-rect {
          width: 46px;
          height: 16px;
          border-radius: 6px;
          border: 1px solid var(--legend-border-color);
          background: var(--legend-fill-color)
        }

        .unexpand-nodes{
          width: 46px;
          height: 16px;
          border-radius: 50%;
          border: 1px solid var(--legend-border-color);
          background: var(--legend-fill-color)
        }

        .api-list {
          width: 50px;
          height: 24px;
        }
        .api-list rect {
          fill:rgb(255, 255, 255); /* 内部无填充 */
          stroke: rgb(99, 99, 99); /* 边框颜色 */
          stroke-width: 1; /* 边框宽度 */
          stroke-dasharray: 10 1; /* 虚线样式 */
        }

        .fusion-node{
          width: 50px;
          height: 24px;
        }
        .fusion-node rect {
          fill: rgb(255, 255, 255); /* 内部无填充 */
          stroke: rgb(99, 99, 99); /* 边框颜色 */
          stroke-width: 1; /* 边框宽度 */
          stroke-dasharray: 2 1; /* 虚线样式 */
        }
      </style>
      <div class="legend">
        <div class="legend-item">
          <svg  class='module-rect'></svg>
          <span class="legend-item-value">[[t('module_or_operators')]]</span>
        </div>
        <div class="legend-item">
          <svg  class='unexpand-nodes'></svg>
          <span class="legend-item-value">[[t('unexpanded_nodes')]]</span>
          <div class="legend-item-value legend-clarifier">
            <paper-tooltip fit-to-visible-bounds animation-delay="0" position="right" offset="0">
              <div class="custom-tooltip">
                [[t('unexpanded_nodes_tooltip')]]
              </div>
            </paper-tooltip>
          </div>
        </div>
        <div class="legend-item">
          <svg class='api-list'>
              <rect width="46" height="18" rx='5' ry='5' x='2' y='4' />
          </svg>
          <span class="legend-item-value">[[t('api_list')]]</span>
          <div class="legend-item-value legend-clarifier">
            <paper-tooltip animation-delay="0" position="right" offset="0">
               <div class="custom-tooltip">[[t('api_list_tooltip')]]</div>
            </paper-tooltip>
          </div>
        </div>
        <div class="legend-item">
          <svg class='fusion-node'>
               <rect width="46" height="18" rx='5' ry='5' x='2' y='4' />
          </svg>
          <span class="legend-item-value">[[t('multi_collection')]]</span>
          <div class="legend-item-value legend-clarifier">
            <paper-tooltip animation-delay="0" position="right" offset="0">
                <div class="custom-tooltip">[[t('multi_collection_tooltip')]]</div>
            </paper-tooltip>
          </div>
        </div>
        
        <!-- 小键盘图标按钮 -->
        <button class="keyboard-button" on-click="toggleShortcutPanel">
          <div class="keyboard-icon"></div>
          <paper-tooltip animation-delay="300" position="top" offset="0">
            [[t('shortcut_help')]]
          </paper-tooltip>
        </button>
        
        <!-- 快捷键弹框 -->
        <div class$="shortcut-panel [[_getPanelClass(showShortcutPanel)]]">
          <div class="shortcut-header">[[t('shortcut_help')]]</div>
          <div class="shortcut-item">
            <span class="shortcut-action">[[t('zoom_in_out')]]</span>
            <span class="shortcut-key">W / S</span>
          </div>
          <div class="shortcut-item">
            <span class="shortcut-action">[[t('pan_left_right')]]</span>
            <span class="shortcut-key">A / D</span>
          </div>
          <div class="shortcut-item">
            <span class="shortcut-action">[[t('scroll')]]</span>
            <span class="shortcut-key">[[t('mouse_scroll')]]</span>
          </div>
          <div class="shortcut-item">
            <span class="shortcut-action">[[t('drag')]]</span>
            <span class="shortcut-key">[[t('mouse_drag')]]</span>
          </div>
        </div>
      </div>
    `;
    
    @property({ type: Object })
    t: Function = (key) => i18next.t(key);
    
    @property({ type: Boolean })
    showShortcutPanel = false;

    constructor() {
      super();
      this.setupLanguageListener();
      this.setupDocumentClickListener();
    }

    setupLanguageListener() {
      i18next.on('languageChanged', () => {
        //更新语言后重新渲染
        const t = this.t;
        this.set('t', null);
        this.set('t', t);
      });
    }
    
    setupDocumentClickListener() {
      // 点击页面其他地方时关闭弹框
      document.addEventListener('click', (e) => {
        if (this.showShortcutPanel && !e.composedPath().includes(this.shadowRoot.host)) {
          this.showShortcutPanel = false;
        }
      });
    }
    
    toggleShortcutPanel(e) {
      e.stopPropagation();
      this.showShortcutPanel = !this.showShortcutPanel;
    }
    
    _getPanelClass(show) {
      return show ? 'show' : '';
    }
}